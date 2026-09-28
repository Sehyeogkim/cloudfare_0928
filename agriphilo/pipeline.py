"""Order pipeline: Customer -> (approve, pay the environment fee) -> Orchestrator -> Environment -> Marketplace.

Once the scene is built the order's game is published to the marketplace. Human players play it in the
browser game; every episode they submit is measured in code and judged by the Episode QA Agent; QA-passed
episodes earn the player a reward and become available for the requester to buy, a few credits each.

Brainbase agents make the judgment calls; code does everything that must be exact. Each order
lives in out/orders/<id>/ and its state is saved to order.json after every change.
"""
import json
import os
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from .agents.customer import CustomerSession
from .agents.environment import EnvironmentTask
from .agents.episode_qa import EpisodeQASession
from .agents.orchestrator import OrchestratorTask
from .agents.payment import PRICING
from .dataset_spec import TASK_STEPS, DatasetSpec
from .models import EnvironmentPlan, MarketEpisode, OrchestratorPlan, Purchase
from .sim.worker import SimWorker

ROOT = Path("out/orders")
REPO = Path(__file__).resolve().parent.parent
SIM_PYTHON = REPO / "sim" / ".venv" / "bin" / "python"
EPISODES = REPO / "runs" / "episodes"
CUSTOMER = "demo-customer"  # the requester's wallet (shared with the game orders in games.py)
ENVIRONMENT_FEE = PRICING["game"]["environment_fee_credits"]
EPISODE_PRICE = PRICING["game"]["episode_price_credits"]
PLAYER_SHARE = PRICING["game"]["player_share_credits"]
TITLE = "Serve a coffee at the cafe counter"
POLL_S = 3
CONTROL_HZ_DEFAULT = 20

STAGES = [
    ("spec", "Spec", "Customer Agent"),
    ("plan", "Plan", "Orchestrator Agent"),
    ("environment", "Scene", "Environment Agent"),
    ("marketplace", "Marketplace", "Players + QA Agent"),
]
DELIVERY_FILES = {
    "dataset.hdf5": "application/x-hdf5",
    "manifest.json": "application/json",
    "qa_report.json": "application/json",
}


def _stage(status: str = "pending") -> Dict[str, Any]:
    return {"status": status, "started_at": None, "finished_at": None, "detail": "", "error": "", "thread_id": None}


class Order:
    def __init__(self, text: str, auto: bool, bb, worker: SimWorker, order_id: Optional[str] = None, ledger=None):
        self.id = order_id or str(uuid.uuid4())
        self.text = text
        self.auto = auto
        self.bb = bb
        self.worker = worker  # kept for the web API's "worker" field
        self.ledger = ledger
        self.created_at = time.time()
        self.dir = ROOT / self.id
        self.lock = threading.RLock()
        self.stages: Dict[str, Dict[str, Any]] = {k: _stage() for k, _, _ in STAGES}
        self.awaiting: Optional[str] = None  # "answers" | "spec_approval"
        self.events: List[Dict[str, Any]] = []
        self.transcript: List[Dict[str, Any]] = [{"role": "customer", "text": text}]
        self.questions: List[str] = []
        self.spec: Optional[DatasetSpec] = None
        self.spec_summary = ""
        self.fee_paid = 0.0
        self.orchestrator_plan: Optional[OrchestratorPlan] = None
        self.orchestrator_summary = ""
        self.plan: Optional[EnvironmentPlan] = None
        self.plan_summary = ""
        self.has_scene_preview = False
        self.episodes: List[MarketEpisode] = []
        self.purchases: List[Purchase] = []
        self.session: Optional[CustomerSession] = None
        self._qa: Optional[EpisodeQASession] = None
        self._closed = threading.Event()

    # ---- state helpers ------------------------------------------------------------------

    def log(self, stage: str, text: str) -> None:
        with self.lock:
            self.events.append({"t": time.time(), "stage": stage, "text": text})
        self.save()

    def start(self, stage: str, detail: str = "") -> None:
        with self.lock:
            self.stages[stage].update(status="running", started_at=time.time(), finished_at=None, detail=detail, error="")
        self.save()

    def finish(self, stage: str, detail: str = "", status: str = "done") -> None:
        with self.lock:
            self.stages[stage].update(status=status, finished_at=time.time(), detail=detail)
        self.save()

    def fail(self, stage: str, error: str) -> None:
        with self.lock:
            self.stages[stage].update(status="failed", finished_at=time.time(), error=error)
            self.awaiting = None
        self.log(stage, f"Failed: {error}")

    @property
    def state(self) -> str:
        if any(s["status"] == "failed" for s in self.stages.values()):
            return "failed"
        m = self.stages["marketplace"]["status"]
        if m == "running":
            return "live"
        if m == "done":
            return "closed"
        if self.awaiting:
            return {"answers": "needs_reply", "spec_approval": "needs_approval"}[self.awaiting]
        return "running"

    # ---- customer stage -----------------------------------------------------------------

    def begin(self) -> None:
        self.start("spec", "Customer Agent is reading the order")
        self.log("spec", "Order received")
        self.session = CustomerSession(self.text, auto=self.auto, bb=self.bb)
        self.stages["spec"]["thread_id"] = self.session.thread_id
        self._spawn(self._run_customer)

    def _run_customer(self) -> None:
        s = self.session
        while s.poll() == "running":
            time.sleep(POLL_S)
        with self.lock:
            self.transcript = list(s.transcript)
        if s.state == "questions":
            with self.lock:
                self.questions = list(s.questions)
                self.awaiting = "answers"
                self.stages["spec"].update(status="waiting", detail="Waiting for your answers")
            self.log("spec", f"Customer Agent asked {len(s.questions)} question(s)")
            return
        s.close()
        if s.state != "ready":
            return self.fail("spec", s.error or "Customer Agent failed")
        with self.lock:
            self.spec, self.spec_summary = s.spec, s.summary
            self.awaiting = "spec_approval"
        self.finish("spec", f"{s.spec.mode} · {s.spec.episode_count} episodes wanted")
        self.log("spec", "Spec ready for approval")

    def answer(self, text: str) -> None:
        with self.lock:
            if self.awaiting != "answers":
                raise RuntimeError("this order is not waiting for answers")
            self.awaiting = None
            self.questions = []
            self.session.answer(text)
            self.transcript = list(self.session.transcript)
            self.stages["spec"].update(status="running", detail="Customer Agent is updating the spec")
        self.log("spec", "Answers sent")
        self._spawn(self._run_customer)

    # ---- environment production -----------------------------------------------------------

    def approve(self) -> None:
        with self.lock:
            if self.awaiting != "spec_approval":
                raise RuntimeError("this order is not waiting for spec approval")
            if self.ledger is not None:
                self.ledger.fee(CUSTOMER, self.id, ENVIRONMENT_FEE)  # raises InsufficientCredits
                self.fee_paid = ENVIRONMENT_FEE
            self.awaiting = None
        self.log("spec", f"Spec approved; environment fee {self.fee_paid:,.0f} credits paid")
        self._spawn(self._run_production)

    def _run_production(self) -> None:
        if not self.orchestrator_plan and not self._run_orchestrator():
            return
        if not self.plan and not self._run_environment():
            return
        self._run_marketplace()

    def _agent_task(self, stage: str, task_factory):
        """Run a Brainbase JsonTask, recording its thread id on the stage."""
        task = task_factory()
        with self.lock:
            self.stages[stage]["thread_id"] = task.thread_id
        self.save()
        return task.run()

    def _run_orchestrator(self) -> bool:
        self.start("plan", "Orchestrator Agent is writing the MuJoCo production plan")
        try:
            holder = {}

            def make():
                holder["t"] = OrchestratorTask(self.spec, self.text, self.bb)
                return holder["t"]

            self.orchestrator_plan = self._agent_task("plan", make)
            self.orchestrator_summary = holder["t"].summary
        except Exception as e:
            self.fail("plan", str(e))
            return False
        self.finish("plan", f"{len(self.orchestrator_plan.steps)} steps")
        self.log("plan", self.orchestrator_summary or "Production plan ready")
        return True

    def _run_environment(self) -> bool:
        self.start("environment", "Environment Agent is planning the scene")
        try:
            holder = {}

            def make():
                holder["t"] = EnvironmentTask(self.spec, self.bb, None, self.orchestrator_plan)
                return holder["t"]

            plan = self._agent_task("environment", make)
            self.plan_summary = holder["t"].summary
        except Exception as e:
            self.fail("environment", str(e))
            return False
        self.log("environment", self.plan_summary or "Scene planned")
        with self.lock:
            self.stages["environment"]["detail"] = "Building the scene in MuJoCo and rendering a preview"
        self.save()
        self.has_scene_preview = self._render_scene_preview(plan)
        with self.lock:
            self.plan = plan
        self.finish("environment", f"{len(plan.layout_ids)} layouts · {len(plan.style_ids)} styles · "
                                   f"{len(plan.cameras)} cameras" + ("" if self.has_scene_preview else " · no preview"))
        return True

    def _render_scene_preview(self, plan: EnvironmentPlan) -> bool:
        """Build the first planned layout/style in MuJoCo and save a render as scene.png."""
        if not SIM_PYTHON.exists():
            self.log("environment", "Scene preview skipped: sim/.venv is not installed")
            return False
        out = self.dir / "work" / "scene"
        cmd = [str(SIM_PYTHON), str(REPO / "sim" / "scripts" / "render_scene.py"), "--task", "CafeServeCoffee",
               "--layout", str(plan.layout_ids[0]), "--style", str(plan.style_ids[0]),
               "--out", str(out), "--size", "720", "--seed", str(plan.base_seed % 1000)]
        try:
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, MUJOCO_GL="cgl"))
            src = out / f"CafeServeCoffee_s{plan.style_ids[0]}_overview_a.png"
            if r.returncode != 0 or not src.exists():
                self.log("environment", f"Scene preview failed: {(r.stderr or r.stdout)[-300:]}")
                return False
            (self.dir / "scene.png").write_bytes(src.read_bytes())
            return True
        except Exception as e:
            self.log("environment", f"Scene preview failed: {e}")
            return False

    # ---- marketplace ------------------------------------------------------------------------

    def _run_marketplace(self) -> None:
        """Keep the listing live: pick up submitted episodes, measure them, and have the QA Agent judge them."""
        self._closed.clear()
        if self.stages["marketplace"]["status"] != "running":
            self.start("marketplace", "Live in the marketplace")
            self.log("marketplace", "Published to the marketplace")
        try:
            while not self._closed.wait(POLL_S):
                seen = {e.file for e in self.episodes}
                folder = EPISODES / self.id
                for path in sorted(folder.glob("*.hdf5")) if folder.exists() else []:
                    rel = os.path.relpath(path, REPO)
                    if rel not in seen and time.time() - path.stat().st_mtime > 1.0:
                        self._ingest(path, rel)
        finally:
            if self._qa is not None and self._qa.started:
                self._qa.close()

    def _ingest(self, path: Path, rel: str) -> None:
        try:
            facts = episode_facts(path)
        except Exception as e:
            with self.lock:
                self.episodes.insert(0, MarketEpisode(
                    episode_id=path.stem, file=rel, player_id="unknown", submitted_at=path.stat().st_mtime, steps=0,
                    duration_s=0, stages_completed=0, stages_total=len(TASK_STEPS), success=False,
                    qa="fail", quality=1, qa_reason=f"Unreadable file: {e}"))
            self.log("marketplace", f"Skipped {path.name}: unreadable episode ({e})")
            return
        ep = MarketEpisode(
            episode_id=path.stem, file=rel, player_id=facts["player_id"], submitted_at=path.stat().st_mtime,
            steps=facts["steps"], duration_s=facts["duration_s"], stages_completed=facts["task_stages_completed"],
            stages_total=facts["task_stages_total"], success=facts["simulator_success"],
            has_thumb=self._thumbnail(path, path.stem))
        with self.lock:
            self.episodes.insert(0, ep)
            self.stages["marketplace"]["detail"] = f"QA Agent is reviewing {path.name}"
        self.save()
        try:
            if self._qa is None:
                self._qa = EpisodeQASession(self.bb)
            review = self._qa.review({k: v for k, v in facts.items() if k != "player_id"})
            ep.qa, ep.quality, ep.qa_reason = review.verdict, review.quality, review.reason
        except Exception as e:
            ep.qa, ep.quality, ep.qa_reason = "fail", None, f"QA could not run: {e}"
            self._qa = None  # start a fresh thread for the next episode
        if ep.qa == "pass" and self.ledger is not None:
            try:
                self.ledger.reward(self.id, ep.episode_id, ep.player_id, PLAYER_SHARE)
                ep.reward_credits = PLAYER_SHARE
            except Exception as e:
                ep.qa_reason += f" (reward not paid: {e})"
        with self.lock:
            self.stages["marketplace"]["detail"] = self._listing_detail()
        self.log("marketplace", f"{ep.player_id} submitted {ep.episode_id}: QA {ep.qa}"
                                + (f" ({ep.quality}/5)" if ep.quality else "") + f" — {ep.qa_reason}")

    def _thumbnail(self, path: Path, episode_id: str) -> bool:
        """Strip of camera frames from the episode (front camera over wrist camera) as ep_<id>.png."""
        import h5py
        import numpy as np
        from PIL import Image

        try:
            with h5py.File(path, "r") as f:
                obs = f["data"]["demo_1"].get("obs")
                if obs is None or not len(obs.keys()):
                    return False
                rows = []
                for cam in sorted(obs.keys()):
                    frames = obs[cam][()]
                    picks = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
                    rows.append(np.concatenate([frames[i] for i in picks], axis=1))
            width = min(r.shape[1] for r in rows)
            self.dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.concatenate([r[:, :width] for r in rows], axis=0)).save(self.dir / f"ep_{episode_id}.png")
            return True
        except Exception:
            return False

    def _counts(self) -> Dict[str, int]:
        eps = self.episodes
        passed = [e for e in eps if e.qa == "pass"]
        return {"submitted": len(eps), "passed": len(passed), "failed": sum(1 for e in eps if e.qa == "fail"),
                "pending": sum(1 for e in eps if e.qa == "pending"),
                "available": sum(1 for e in passed if not e.purchased), "purchased": sum(1 for e in passed if e.purchased)}

    def _listing_detail(self) -> str:
        c = self._counts()
        return f"{c['submitted']} submitted · {c['passed']} passed · {c['available']} available · {c['purchased']} purchased"

    def purchase(self, count: int) -> None:
        """Buy `count` QA-passed episodes (oldest first) and rebuild the downloadable dataset."""
        with self.lock:
            if self.stages["marketplace"]["status"] not in ("running", "done"):
                raise RuntimeError("this order is not in the marketplace yet")
            available = sorted((e for e in self.episodes if e.qa == "pass" and not e.purchased),
                               key=lambda e: e.submitted_at)
            if count < 1 or count > len(available):
                raise RuntimeError(f"{len(available)} QA-passed episode(s) available; asked for {count}")
            picks = available[:count]
            credits = round(count * EPISODE_PRICE, 2)
            if self.ledger is not None:  # raises InsufficientCredits
                self.ledger.purchase(CUSTOMER, self.id, f"purchase:{self.id}:{len(self.purchases)}", credits,
                                     [e.episode_id for e in picks])
            for e in picks:
                e.purchased = True
            self.purchases.append(Purchase(at=time.time(), count=count, credits=credits,
                                           episode_ids=[e.episode_id for e in picks]))
            self.stages["marketplace"]["detail"] = self._listing_detail()
        self._build_delivery()
        self.log("marketplace", f"Bought {count} episode(s) for {credits:,.0f} credits")

    def _build_delivery(self) -> None:
        """dataset.hdf5 (collect_demos layout, one demo per purchased episode) + manifest + QA report."""
        import h5py

        bought = [e for e in sorted(self.episodes, key=lambda e: e.submitted_at) if e.purchased]
        out = self.dir / "delivery"
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "dataset.hdf5.tmp"
        with h5py.File(tmp, "w") as dst:
            data = dst.create_group("data")
            for i, e in enumerate(bought, 1):
                with h5py.File(REPO / e.file, "r") as src:
                    if i == 1:
                        data.attrs.update(dict(src["data"].attrs))
                    src.copy(src["data"]["demo_1"], data, name=f"demo_{i}")
                data[f"demo_{i}"].attrs.update(episode_id=e.episode_id, qa_quality=e.quality or 0, qa_reason=e.qa_reason)
        tmp.replace(out / "dataset.hdf5")
        (out / "manifest.json").write_text(json.dumps({
            "order_id": self.id, "task": self.plan.task if self.plan else "CafeServeCoffee",
            "robot": self.plan.robot if self.plan else "PandaOmron", "episodes": len(bought),
            "steps": sum(e.steps for e in bought), "players": sorted({e.player_id for e in bought}),
            "format": "robomimic/collect_demos HDF5 (states, actions, model_file, ep_meta, obs preview frames)",
            "simulator": "MuJoCo (RoboCasa)"}, indent=2))
        (out / "qa_report.json").write_text(json.dumps([
            {"episode_id": e.episode_id, "player_id": e.player_id, "verdict": e.qa, "quality": e.quality,
             "reason": e.qa_reason, "stages_completed": e.stages_completed, "stages_total": e.stages_total,
             "duration_s": e.duration_s} for e in bought], indent=2, ensure_ascii=False))

    def close_listing(self) -> None:
        with self.lock:
            if self.stages["marketplace"]["status"] != "running":
                raise RuntimeError("the listing is not live")
        self._closed.set()
        self.finish("marketplace", self._listing_detail() + " · closed")
        self.log("marketplace", "Listing closed")

    def reopen_listing(self) -> None:
        with self.lock:
            if self.stages["marketplace"]["status"] != "done":
                raise RuntimeError("the listing is not closed")
            self.stages["marketplace"].update(status="running", finished_at=None, detail=self._listing_detail())
        self.log("marketplace", "Listing reopened")
        self._spawn(self._run_marketplace)

    # ---- retry ------------------------------------------------------------------------------

    def retry(self) -> None:
        """Restart a failed order from the stage that failed."""
        with self.lock:
            failed = [k for k, s in self.stages.items() if s["status"] == "failed"]
            if not failed:
                raise RuntimeError("nothing to retry")
            stage = failed[0]
            if stage == "spec":
                reset, run = ["spec"], self._restart_customer
            else:
                reset = {"plan": ["plan", "environment", "marketplace"], "environment": ["environment", "marketplace"],
                         "marketplace": ["marketplace"]}[stage]
                if "plan" in reset:
                    self.orchestrator_plan = None
                if "environment" in reset:
                    self.plan = None
                run = self._run_production
            for k in reset:
                self.stages[k] = _stage()
        self.log(stage, "Retrying")
        self._spawn(run)

    def _restart_customer(self) -> None:
        """New Customer Agent conversation seeded with everything the customer has said so far."""
        said = [t["text"] for t in self.transcript if t["role"] == "customer" and t.get("text")]
        self.start("spec", "Customer Agent is reading the order")
        self.session = CustomerSession("\n\n".join(said), auto=self.auto, bb=self.bb)
        self.stages["spec"]["thread_id"] = self.session.thread_id
        self._run_customer()

    # ---- plumbing ---------------------------------------------------------------------------

    def _spawn(self, fn) -> None:
        def wrapped():
            try:
                fn()
            except Exception as e:
                traceback.print_exc()
                running = [k for k, s in self.stages.items() if s["status"] in ("running", "waiting")]
                self.fail(running[0] if running else "spec", str(e))
                if self.session and self.session.state == "running":
                    self.session.close()

        threading.Thread(target=wrapped, daemon=True).start()

    def file_path(self, name: str) -> Optional[Path]:
        if name == "scene.png" and self.has_scene_preview:
            return self.dir / "scene.png"
        if any(name == f"ep_{e.episode_id}.png" and e.has_thumb for e in self.episodes):
            return self.dir / name
        if name in DELIVERY_FILES and self.purchases:
            return self.dir / "delivery" / name
        return None

    def play_urls(self) -> Dict[str, str]:
        base = os.environ.get("AGRIPHILO_GAME_SERVER_URL", "")
        if not (base and self.plan):
            return {"player": "", "viewer": ""}
        q = {"game_id": self.id, "task": self.plan.task, "robot": self.plan.robot,
             "layout_ids": self.plan.layout_ids[0], "style_ids": self.plan.style_ids[0]}
        return {role: f"{base}?{urlencode(dict(q, role=role))}" for role in ("player", "viewer")}

    def _credits(self) -> Dict[str, Any]:
        balance = self.ledger.balance(f"wallet:{CUSTOMER}") if self.ledger is not None else None
        return {"environment_fee": ENVIRONMENT_FEE, "episode_price": EPISODE_PRICE, "balance": balance,
                "spent": round(self.fee_paid + sum(p.credits for p in self.purchases), 2)}

    def _listing(self) -> Optional[Dict[str, Any]]:
        if self.stages["marketplace"]["status"] not in ("running", "done"):
            return None
        urls = self.play_urls()
        return {"status": "live" if self.stages["marketplace"]["status"] == "running" else "closed",
                "play_url": urls["player"], "viewer_url": urls["viewer"], "reward_per_episode": PLAYER_SHARE,
                "episode_price": EPISODE_PRICE, "episodes_wanted": self.spec.episode_count if self.spec else 0,
                **self._counts()}

    def to_dict(self) -> Dict[str, Any]:
        dump = lambda m: m.model_dump() if m else None  # noqa: E731
        with self.lock:
            return {
                "id": self.id, "text": self.text, "auto": self.auto, "created_at": self.created_at,
                "state": self.state, "awaiting": self.awaiting,
                "stages": [{"key": k, "label": label, "owner": owner, **self.stages[k]} for k, label, owner in STAGES],
                "events": self.events, "transcript": self.transcript, "questions": self.questions,
                "spec": dump(self.spec), "spec_summary": self.spec_summary, "credits": self._credits(),
                "fee_paid": self.fee_paid,
                "orchestrator_plan": dump(self.orchestrator_plan), "orchestrator_summary": self.orchestrator_summary,
                "plan": dump(self.plan), "plan_summary": self.plan_summary, "has_scene_preview": self.has_scene_preview,
                "listing": self._listing(),
                "episodes": [e.model_dump() for e in self.episodes],
                "purchases": [p.model_dump() for p in self.purchases],
                "worker": self.worker.name,
                "files": [n for n in DELIVERY_FILES if self.purchases],
            }

    def marketplace_item(self) -> Optional[Dict[str, Any]]:
        """This order's game as a marketplace listing, once its scene is built."""
        listing = self._listing()
        if not listing or not self.plan:
            return None
        return {
            "order_id": self.id, "title": TITLE, "task": self.plan.task,
            "task_steps": list(self.spec.task_steps), "robot": self.plan.robot,
            "scene": f"{self.plan.task} · layout {self.plan.layout_ids[0]} · style {self.plan.style_ids[0]}",
            "scene_image_url": f"/api/orders/{self.id}/files/scene.png" if self.has_scene_preview else None,
            "requester": CUSTOMER, "status": listing["status"],
            "reward_per_episode": PLAYER_SHARE, "episode_price": EPISODE_PRICE,
            "episodes_wanted": listing["episodes_wanted"], "episodes_submitted": listing["submitted"],
            "episodes_passed": listing["passed"], "episodes_available": listing["available"],
            "play_url": listing["play_url"], "viewer_url": listing["viewer_url"], "created_at": self.created_at,
        }

    def summary(self) -> Dict[str, Any]:
        d = self.to_dict()
        for k in ("events", "transcript"):
            d.pop(k)
        return d

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"order.json.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False))
        tmp.replace(self.dir / "order.json")

    @classmethod
    def load(cls, path: Path, bb, worker: SimWorker, ledger=None) -> "Order":
        """Restore a saved order. Agent work in flight when the server stopped is marked failed;
        a live marketplace listing resumes."""
        d = json.loads(path.read_text())
        if {s["key"] for s in d["stages"]} != {k for k, _, _ in STAGES}:
            raise ValueError("order from an earlier pipeline version; not loaded")
        o = cls(d["text"], d["auto"], bb, worker, order_id=d["id"], ledger=ledger)
        o.created_at = d["created_at"]
        o.stages = {s["key"]: {k: s[k] for k in _stage()} for s in d["stages"]}
        o.awaiting, o.events, o.transcript = d["awaiting"], d["events"], d["transcript"]
        o.questions = d.get("questions", [])
        o.spec = DatasetSpec.model_validate(d["spec"]) if d["spec"] else None
        o.spec_summary, o.plan_summary = d["spec_summary"], d["plan_summary"]
        o.fee_paid = d.get("fee_paid", 0.0)
        o.orchestrator_plan = OrchestratorPlan.model_validate(d["orchestrator_plan"]) if d.get("orchestrator_plan") else None
        o.orchestrator_summary = d.get("orchestrator_summary", "")
        o.plan = EnvironmentPlan.model_validate(d["plan"]) if d["plan"] else None
        o.has_scene_preview = d.get("has_scene_preview", False)
        o.episodes = [MarketEpisode.model_validate(e) for e in d.get("episodes", [])]
        for e in o.episodes:
            if e.qa == "pending":  # its QA thread died with the server
                e.qa, e.qa_reason = "fail", "QA interrupted by a server restart"
        o.purchases = [Purchase.model_validate(p) for p in d.get("purchases", [])]
        in_flight = [k for k, s in o.stages.items() if k != "marketplace" and s["status"] in ("running", "waiting")]
        if o.awaiting == "answers" or in_flight:
            stuck = in_flight[0] if in_flight else "spec"
            o.stages[stuck].update(status="failed", error="Server restarted while this stage was in progress")
            o.awaiting = None
        elif o.stages["marketplace"]["status"] == "running":
            o._spawn(o._run_marketplace)
        return o


def episode_facts(path: Path) -> Dict[str, Any]:
    """Measure a submitted episode (collect_demos HDF5 written by the game server) for the QA Agent."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        data = f["data"]
        d = data["demo_1"]
        actions = np.asarray(d["actions"][()])
        env_info = json.loads(data.attrs.get("env_info", "{}"))
        hz = float(env_info.get("control_freq", CONTROL_HZ_DEFAULT))
        grip = actions[:, 6] if actions.ndim == 2 and actions.shape[1] > 6 else np.zeros(len(actions))
        arm = actions[:, :6] if actions.ndim == 2 and actions.shape[1] >= 6 else actions
        return {
            "player_id": str(d.attrs.get("player_id", "unknown")),
            "task": str(data.attrs.get("env", "CafeServeCoffee")),
            "task_steps": TASK_STEPS,
            "task_stages_completed": int(d.attrs.get("stage", 0)),
            "task_stages_total": len(TASK_STEPS),
            "simulator_success": bool(d.attrs.get("success", False)),
            "steps": int(len(actions)),
            "duration_s": round(len(actions) / hz, 1),
            "idle_ratio": round(float((np.abs(arm).max(axis=1) < 1e-3).mean()) if len(arm) else 1.0, 3),
            "max_abs_action": round(float(np.abs(actions).max()) if actions.size else 0.0, 3),
            "finite": bool(np.isfinite(actions).all()),
            "gripper_toggles": int((np.diff(np.sign(grip)) != 0).sum()) if len(grip) > 1 else 0,
            "has_camera_frames": "obs" in d,
        }


class OrderStore:
    def __init__(self, bb, worker: SimWorker, ledger=None):
        self.bb, self.worker, self.ledger = bb, worker, ledger
        self.orders: Dict[str, Order] = {}
        self.lock = threading.Lock()
        ROOT.mkdir(parents=True, exist_ok=True)
        for path in sorted(ROOT.glob("*/order.json")):
            try:
                o = Order.load(path, bb, worker, ledger)
                self.orders[o.id] = o
            except Exception as e:
                print(f"skipping {path}: {e}")

    def create(self, text: str, auto: bool) -> Order:
        o = Order(text, auto, self.bb, self.worker, ledger=self.ledger)
        with self.lock:
            self.orders[o.id] = o
        o.begin()
        return o

    def marketplace(self) -> List[Dict[str, Any]]:
        items = [i for i in (o.marketplace_item() for o in self.all()) if i]
        return sorted(items, key=lambda i: (i["status"] != "live", -i["created_at"]))

    def player(self, player_id: str) -> Dict[str, Any]:
        eps = [{"order_id": o.id, "title": TITLE, "episode_id": e.episode_id, "qa": e.qa, "quality": e.quality,
                "qa_reason": e.qa_reason, "reward_credits": e.reward_credits, "submitted_at": e.submitted_at}
               for o in self.all() for e in o.episodes if e.player_id == player_id]
        balance = self.ledger.balance(f"payable:{player_id}") if self.ledger is not None else 0.0
        return {"player_id": player_id, "balance": balance, "episodes": sorted(eps, key=lambda e: -e["submitted_at"])}

    def get(self, order_id: str) -> Order:
        return self.orders[order_id]

    def all(self) -> List[Order]:
        return sorted(self.orders.values(), key=lambda o: o.created_at, reverse=True)
