"""Game orders for the hackathon demo: order -> GameSpec -> play (human / AI) -> per-episode QA -> credits.

Episode contract (shared with the game server and the LLM player):
  runs/episodes/<game_id>/*.hdf5   collect_demos-format HDF5 (see agriphilo/game_qa.py)
    demo attrs (or file-level data attrs): player_id, player_kind ("human" | "ai")
  optional sidecar  <name>.json  or  <name>.hdf5.json:
    {"player_id": ..., "plan": [...], "reasoning": "...",
     "usage": {"input_tokens": n, "output_tokens": n, "cost_usd": x},       # applies to the whole file
     "episodes": {"demo_1": {"plan": ..., "reasoning": ..., "usage": {...}}}} # optional per-demo overrides

New or changed files are QA'd in a background thread: structural checks in-process, replay + _check_success in
the sim venv (sim/.venv/bin/python -m agriphilo.game_qa --replay). Each QA-passed episode captures its price from
the order's hold; AI episodes with usage also post a token_cost entry.
"""
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import h5py

from . import game_qa
from .agents.environment import GameEnvironmentTask
from .agents.payment import PRICING
from .credits import InsufficientCredits, Ledger
from .game_spec import GameSpec

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
CUSTOMER = "demo-customer"
GAME_PRICING = PRICING["game"]
SETTLE_S = 2.0          # a file must be unchanged this long before QA (writers may still be flushing)
REPLAY_TIMEOUT_S = 900


def _sidecar(path: Path) -> dict:
    for p in (path.with_suffix(".json"), path.with_name(path.name + ".json")):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except ValueError:
                return {"error": f"sidecar {p.name} is not valid JSON"}
    return {}


def _usage_cost(usage: dict) -> Optional[float]:
    if not usage:
        return None
    if usage.get("cost_usd") is not None:
        usd = float(usage["cost_usd"])
    else:
        rate = GAME_PRICING["llm_placeholder_usd_per_mtok"]
        usd = (usage.get("input_tokens", 0) * rate["input"] + usage.get("output_tokens", 0) * rate["output"]) / 1e6
    return round(usd / GAME_PRICING["credit_usd"], 6)


def _attr(obj, name, default=None):
    v = obj.attrs.get(name, default)
    return v.decode() if isinstance(v, bytes) else v


def _locked(fn):
    def wrap(self, *a, **k):
        with self.lock:
            return fn(self, *a, **k)
    wrap.__name__ = fn.__name__
    return wrap


class Game:
    def __init__(self, store: "GameStore", data: dict):
        self.store = store
        self.d = data
        self.lock = threading.RLock()  # request threads and the QA worker both touch self.d

    @property
    def id(self) -> str:
        return self.d["id"]

    @property
    def spec(self) -> Optional[GameSpec]:
        return GameSpec.model_validate(self.d["spec"]) if self.d.get("spec") else None

    @property
    def episode_dir(self) -> Path:
        return RUNS / "episodes" / self.id

    def save(self) -> None:
        with self.lock:
            p = RUNS / "games" / f"{self.id}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.name}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self.d, indent=1, ensure_ascii=False))
            tmp.replace(p)

    # ---- lifecycle ---------------------------------------------------------------------------
    def design(self) -> None:
        try:
            task = GameEnvironmentTask(self.d["order_text"], self.store.bb)
            spec = task.run()
            price = max(GAME_PRICING["episode_price_credits"], spec.credit_per_pass)
            self.d.update(spec=spec.model_dump(), summary=task.summary, status="awaiting_approval",
                          price_per_episode=price, hold_amount=round(price * spec.episodes_requested, 6))
        except Exception as e:
            self.d.update(status="failed", error=f"Game Environment Agent: {e}")
        self.save()

    @_locked
    def approve(self) -> None:
        if self.d["status"] != "awaiting_approval":
            raise RuntimeError(f"cannot approve a game in state {self.d['status']}")
        try:
            self.store.ledger.hold(CUSTOMER, self.id, self.d["hold_amount"])
        except InsufficientCredits as e:
            raise RuntimeError(str(e))
        self.d["status"] = "active"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.save()

    @_locked
    def close(self) -> None:
        if self.d["status"] not in ("active", "awaiting_approval"):
            raise RuntimeError(f"cannot close a game in state {self.d['status']}")
        if self.d["status"] == "active":
            self.store.ledger.release(CUSTOMER, self.id)
        self.d["status"] = "closed"
        self.save()

    # ---- episodes ----------------------------------------------------------------------------
    @_locked
    def scan(self) -> None:
        """Queue new/changed HDF5 files for QA."""
        if self.d["status"] not in ("active", "closed") or not self.episode_dir.exists():
            return
        now = time.time()
        for p in sorted(self.episode_dir.glob("*.hdf5")):
            st = p.stat()
            sig = f"{st.st_size}:{st.st_mtime_ns}"
            f = self.d["files"].get(p.name, {})
            if f.get("sig") == sig or now - st.st_mtime < SETTLE_S:
                continue
            self.d["files"][p.name] = {"sig": sig, "status": "queued"}
            self.store.qa_queue.put((self.id, p.name))
        self.save()

    def qa_file(self, name: str) -> None:
        path = self.episode_dir / name
        spec = self.spec
        self.d["files"][name]["status"] = "qa_running"
        self.save()
        cfg = game_qa.QAConfig()
        seen = set(self.d["seen_hashes"])
        try:
            verdicts = game_qa.qa_dataset(path, cfg=cfg, replay=False, seen_hashes=seen)
        except Exception as e:
            self.d["files"][name].update(status="error", error=f"could not read HDF5: {e}")
            return self.save()
        self.d["seen_hashes"] = sorted(seen)

        replay, replay_note = self._replay(path, cfg, [v.episode_id for v in verdicts if v.passed])
        side = _sidecar(path)
        with h5py.File(path, "r") as f:
            data = f["data"]
            meta = {n: (_attr(data[n], "player_id", _attr(data, "player_id", side.get("player_id", "unknown"))),
                        _attr(data[n], "player_kind", _attr(data, "player_kind", side.get("player_kind", "human"))),
                        len(data[n]["actions"]) if "actions" in data[n] else 0)
                    for n in data.keys()}

        with self.lock:  # billing reads status; close() may run concurrently
            file_usage_charged = False
            for v in verdicts:
                if v.passed and v.episode_id in replay:
                    v = replay[v.episode_id]
                player_id, kind, steps = meta[v.episode_id]
                if spec and steps > spec.max_steps:
                    v.passed = False
                    v.reasons.append(f"episode has {steps} steps > max_steps {spec.max_steps} (time limit)")
                if kind not in ("human", "ai"):
                    v.passed = False
                    v.reasons.append(f"unknown player_kind {kind!r}")
                ep_id = f"{path.stem}/{v.episode_id}"
                per = side.get("episodes", {}).get(v.episode_id, {})
                usage = per.get("usage") or (None if file_usage_charged else side.get("usage"))
                rec = {"episode_id": ep_id, "file": name, "demo": v.episode_id, "player_id": player_id,
                       "player_kind": kind, "steps": steps, "passed": v.passed, "reasons": v.reasons,
                       "metrics": v.metrics, "replay_checked": v.replay_checked, "replay_note": replay_note,
                       "plan": per.get("plan", side.get("plan")), "reasoning": per.get("reasoning", side.get("reasoning")),
                       "usage": usage, "billing": self._bill(ep_id, v.passed, player_id, kind, usage)}
                if usage and not per.get("usage"):
                    file_usage_charged = True
                self.d["episodes"][ep_id] = rec
            self.d["files"][name]["status"] = "done"
            self.save()

    def _replay(self, path: Path, cfg, ids: List[str]):
        """Replay in the sim venv. Returns ({demo: verdict}, note)."""
        if not ids:
            return {}, "no structurally valid episode to replay"
        if os.environ.get("AGRIPHILO_SKIP_REPLAY"):
            return {}, "replay skipped: AGRIPHILO_SKIP_REPLAY set (structural checks only)"
        if not game_qa.SIM_PYTHON.exists():
            return {}, "replay skipped: sim venv not available (structural checks only)"
        cmd = [str(game_qa.SIM_PYTHON), "-m", "agriphilo.game_qa", str(path), "--replay",
               "--config", json.dumps({"min_steps": cfg.min_steps, "require_success": cfg.require_success})]
        try:
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=REPLAY_TIMEOUT_S,
                               env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl")})
        except subprocess.TimeoutExpired:
            return {i: game_qa.EpisodeVerdict(i, False, ["replay timed out"]) for i in ids}, "replay timed out"
        out = {}
        for line in r.stdout.splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                out[d["episode_id"]] = game_qa.EpisodeVerdict(**d)
        if r.returncode != 0 or not out:
            err = (r.stderr.strip().splitlines() or ["no output"])[-1][:300]
            return {i: game_qa.EpisodeVerdict(i, False, [f"replay could not run: {err}"]) for i in ids}, "replay failed to run"
        return out, "replayed in sim venv"

    def _bill(self, ep_id: str, passed: bool, player_id: str, kind: str, usage: Optional[dict]) -> dict:
        led, spec, b = self.store.ledger, self.spec, {}
        cost = _usage_cost(usage) if kind == "ai" else None
        if cost is not None:
            led.token_cost(self.id, ep_id, player_id, cost, usage)
            b["token_cost"] = cost
        if not passed:
            b["charged"] = 0.0
            b["note"] = "QA failed: not charged"
        elif self.d["status"] != "active":
            b["charged"] = 0.0
            b["note"] = f"order {self.d['status']}: not charged"
        else:
            try:
                led.capture(self.id, ep_id, self.d["price_per_episode"], player_id, kind, spec.credit_per_pass)
                b.update(charged=self.d["price_per_episode"], player_share=spec.credit_per_pass,
                         platform_fee=round(self.d["price_per_episode"] - spec.credit_per_pass, 6))
            except InsufficientCredits as e:
                b.update(charged=0.0, note=f"not charged: {e}")
        return b

    @_locked
    def to_dict(self) -> dict:
        led = self.store.ledger
        eps = sorted(self.d["episodes"].values(), key=lambda e: e["episode_id"])
        passed = [e for e in eps if e["passed"]]
        return {**self.d, "seen_hashes": None, "episodes": eps,
                "progress": {"passed": len(passed), "requested": (self.d.get("spec") or {}).get("episodes_requested"),
                             "total": len(eps)},
                "hold_left": led.balance(f"hold:{self.id}"),
                "ledger": led.for_ref(self.id),
                "episode_dir": os.path.relpath(self.episode_dir, REPO)}

    def summary(self) -> dict:
        s = self.d.get("spec") or {}
        return {"id": self.id, "status": self.d["status"], "task": s.get("task"), "created_at": self.d["created_at"],
                "order_text": self.d["order_text"][:120]}


class GameStore:
    def __init__(self, bb, ledger: Optional[Ledger] = None):
        self.bb = bb
        self.ledger = ledger or Ledger(RUNS / "ledger.json")
        self.games: Dict[str, Game] = {}
        for p in sorted((RUNS / "games").glob("*.json")):
            d = json.loads(p.read_text())
            if d["status"] == "designing":
                d.update(status="failed", error="server restarted while designing")
            for f in d["files"].values():
                if f["status"] in ("queued", "qa_running"):
                    f["sig"] = None  # re-QA on next scan
            self.games[d["id"]] = Game(self, d)
        self.qa_queue: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._qa_worker, daemon=True).start()

    def _qa_worker(self) -> None:
        while True:
            gid, name = self.qa_queue.get()
            try:
                self.games[gid].qa_file(name)
            except Exception as e:  # keep the worker alive
                g = self.games[gid]
                g.d["files"][name].update(status="error", error=str(e))
                g.save()

    def create(self, order_text: str) -> Game:
        gid = uuid.uuid4().hex[:12]
        g = Game(self, {"id": gid, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "order_text": order_text,
                        "customer_id": CUSTOMER, "status": "designing", "summary": "", "error": None, "spec": None,
                        "price_per_episode": None, "hold_amount": None, "files": {}, "episodes": {},
                        "seen_hashes": []})
        g.save()
        self.games[gid] = g
        threading.Thread(target=g.design, daemon=True).start()
        return g

    def get(self, gid: str) -> Game:
        g = self.games[gid]
        g.scan()
        return g

    def all(self) -> List[Game]:
        return sorted(self.games.values(), key=lambda g: g.d["created_at"], reverse=True)

    def wallet(self) -> dict:
        led = self.ledger
        return {"customer_id": CUSTOMER, "available": led.balance(f"wallet:{CUSTOMER}"),
                "held": round(sum(v for a, v in led.balances("hold:").items()), 6),
                "payable": led.balances("payable:"), "ai_budget": led.balances("ai_budget:"),
                "platform_fee": led.balance("platform:fee"), "llm_spend": led.balance("llm:vendor"),
                "test_mode": True, "credit_usd": GAME_PRICING["credit_usd"],
                "entries": led.entries[-40:][::-1]}
