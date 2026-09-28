"""Code-side QA for RoboCasa game episodes (plan_game_platform §4, step D).

Reads the HDF5 written by robocasa/scripts/collect_demos.py:

    data (group)            attrs: env (task name), env_info (JSON robosuite.make config)
                            or, after dataset_states_to_obs, env_args (JSON {env_name, env_kwargs})
      demo_N (group)        attrs: model_file (MJCF xml), ep_meta (JSON)
        states   (T, S)     flattened mujoco state *before* action t
        actions  (T, A)
        actions_abs (T, B)  optional

Code measures; agents only interpret. robocasa is imported lazily so structural QA
and the ledger work without the simulator installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SIM_DIR = REPO / "sim"
# Repo dirs under sim/third_party (kept out of sim/ so they do not shadow the installed packages).
SIM_PATHS = [SIM_DIR / "third_party" / "robocasa", SIM_DIR / "third_party" / "robosuite"]
SIM_PYTHON = SIM_DIR / ".venv" / "bin" / "python"
DATASET_SCRIPTS = SIM_DIR / "third_party" / "robocasa" / "robocasa" / "scripts" / "dataset_scripts"


@dataclass
class QAConfig:
    min_steps: int = 20
    max_abs_action: float = 1.0 + 1e-6  # robosuite delta controllers clip to [-1, 1]
    idle_eps: float = 1e-3               # |action| below this (all dims) counts as idle
    max_idle_ratio: float = 0.6
    # Max |qpos| divergence (m or rad) between replay and recording. Velocities are not compared:
    # free objects settling on contact differ ~1e-2 in qvel between runs (MuJoCo warmstart is not
    # in the flattened state), while qpos stays ~1e-3. Tampered actions diverge by >1e-2 in a step.
    replay_qpos_tol: float = 1e-2
    require_success: bool = True


@dataclass
class EpisodeVerdict:
    episode_id: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    replay_checked: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---- structural checks ---------------------------------------------------------------------

def episode_hash(states: np.ndarray, actions: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (states, actions):
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def check_structure(grp: h5py.Group, episode_id: str, cfg: QAConfig,
                    seen_hashes: Optional[Set[str]] = None) -> EpisodeVerdict:
    v = EpisodeVerdict(episode_id=episode_id, passed=False)
    missing = [k for k in ("states", "actions") if k not in grp]
    if "model_file" not in grp.attrs:
        missing.append("attr:model_file")
    if missing:
        v.reasons.append(f"missing: {', '.join(missing)}")
        return v
    states = np.asarray(grp["states"][()])
    actions = np.asarray(grp["actions"][()])
    T = len(actions)
    v.metrics["steps"] = float(T)

    if states.ndim != 2 or actions.ndim != 2:
        v.reasons.append(f"bad rank: states {states.shape}, actions {actions.shape}")
        return v
    if len(states) != T:
        v.reasons.append(f"length mismatch: {len(states)} states vs {T} actions")
    if "actions_abs" in grp and len(grp["actions_abs"]) != T:
        v.reasons.append(f"length mismatch: {len(grp['actions_abs'])} actions_abs vs {T} actions")
    if not (np.isfinite(states).all() and np.isfinite(actions).all()):
        v.reasons.append("non-finite values in states/actions")
    if T < cfg.min_steps:
        v.reasons.append(f"too short: {T} < {cfg.min_steps} steps")

    if T and np.isfinite(actions).all():
        max_abs = float(np.abs(actions).max())
        idle = float((np.abs(actions).max(axis=1) < cfg.idle_eps).mean())
        v.metrics.update(max_abs_action=max_abs, idle_ratio=idle)
        if max_abs > cfg.max_abs_action:
            v.reasons.append(f"action out of range: max |a| = {max_abs:.3f} > {cfg.max_abs_action:.3f}")
        if idle > cfg.max_idle_ratio:
            v.reasons.append(f"too idle: {idle:.0%} of steps have no input (limit {cfg.max_idle_ratio:.0%})")

    if "ep_meta" in grp.attrs:
        try:
            json.loads(grp.attrs["ep_meta"])
        except ValueError:
            v.reasons.append("ep_meta is not valid JSON")

    if seen_hashes is not None and len(states) == T:
        hsh = episode_hash(states, actions)
        v.metrics["hash_prefix"] = int(hsh[:8], 16)
        if hsh in seen_hashes:
            v.reasons.append("duplicate of an earlier episode")
        seen_hashes.add(hsh)

    v.passed = not v.reasons
    return v


# ---- replay check ----------------------------------------------------------------------------

def _import_sim():
    for p in SIM_PATHS:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    import robosuite  # noqa: F401
    import robocasa  # noqa: F401  (registers kitchen envs)
    if str(SIM_DIR) not in sys.path:  # sim/cafe: custom tasks such as CafeServeCoffee
        sys.path.insert(0, str(SIM_DIR))
    import cafe.cafe_env  # noqa: F401
    return robosuite


def env_kwargs_from_file(f: h5py.File) -> dict:
    """robosuite.make kwargs from either raw collect_demos output or states_to_obs output."""
    attrs = f["data"].attrs
    if "env_args" in attrs:
        meta = json.loads(attrs["env_args"])
        kw = dict(meta["env_kwargs"])
        kw["env_name"] = meta["env_name"]
    elif "env_info" in attrs:
        kw = json.loads(attrs["env_info"])
        kw.setdefault("env_name", attrs.get("env"))
    else:
        raise KeyError("data has neither env_args nor env_info")
    kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
              ignore_done=True)
    kw.setdefault("control_freq", 20)
    return kw


def make_env(f: h5py.File):
    robosuite = _import_sim()
    return robosuite.make(**env_kwargs_from_file(f))


def _reset_to(env, model_xml: str, ep_meta: Optional[str], state0: np.ndarray) -> None:
    """Same sequence as robocasa playback_dataset.reset_to."""
    meta = json.loads(ep_meta) if ep_meta else {}
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(state0)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    elif hasattr(env, "update_state"):
        env.update_state()


def replay_check(env, grp: h5py.Group, verdict: EpisodeVerdict, cfg: QAConfig) -> EpisodeVerdict:
    """Replay recorded actions from states[0]; qpos in states[t+1] must match within tolerance.

    Flattened MuJoCo state layout is [time, qpos(nq), qvel(nv), act...].
    """
    states = np.asarray(grp["states"][()])
    actions = np.asarray(grp["actions"][()])
    ep_meta = grp.attrs.get("ep_meta")
    _reset_to(env, grp.attrs["model_file"], ep_meta, states[0])
    nq = env.sim.model.nq
    if states.shape[1] != len(env.sim.get_state().flatten()):
        verdict.replay_checked = True
        verdict.reasons.append(f"state size {states.shape[1]} does not match the rebuilt scene")
        verdict.passed = False
        return verdict
    max_err, max_full, first_bad = 0.0, 0.0, None
    for t in range(len(actions)):
        env.step(actions[t])
        if t < len(actions) - 1:
            now = env.sim.get_state().flatten()
            err = float(np.abs(states[t + 1][1:1 + nq] - now[1:1 + nq]).max())
            max_err = max(max_err, err)
            max_full = max(max_full, float(np.linalg.norm(states[t + 1] - now)))
            if first_bad is None and err > cfg.replay_qpos_tol:
                first_bad = t
    success = bool(env._check_success())
    verdict.replay_checked = True
    verdict.metrics.update(replay_max_qpos_err=max_err, replay_max_state_l2=max_full,
                           replay_success=float(success))
    if first_bad is not None:
        verdict.reasons.append(
            f"replay diverged at step {first_bad} (max qpos err {max_err:.2e} > {cfg.replay_qpos_tol:.0e})")
    if cfg.require_success and not success:
        verdict.reasons.append("task not successful on replay")
    verdict.passed = not verdict.reasons
    return verdict


# ---- dataset-level QA ----------------------------------------------------------------------

def qa_dataset(path: Path, cfg: Optional[QAConfig] = None, replay: bool = False,
               seen_hashes: Optional[Set[str]] = None) -> List[EpisodeVerdict]:
    """Structural QA on every demo; replay only the ones that pass structure (if replay=True)."""
    cfg = cfg or QAConfig()
    seen = seen_hashes if seen_hashes is not None else set()
    out: List[EpisodeVerdict] = []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
        env = None
        for name in demos:
            v = check_structure(f["data"][name], name, cfg, seen)
            if replay and v.passed:
                if env is None:
                    env = make_env(f)
                replay_check(env, f["data"][name], v, cfg)
            out.append(v)
        if env is not None:
            env.close()
    return out


# ---- extraction (subprocess into the sim venv) ----------------------------------------------

def _sim_python() -> str:
    return str(SIM_PYTHON) if SIM_PYTHON.exists() else sys.executable


def ensure_robomimic_meta(dataset: Path) -> None:
    """Add data.attrs env_args/total and per-demo num_samples, as robocasa's
    convert_to_robomimic_format does at the end of collect_demos. The dataset scripts need them."""
    with h5py.File(dataset, "a") as f:
        d = f["data"]
        if "env_args" not in d.attrs:
            if "env_info" not in d.attrs:
                raise KeyError("data has neither env_args nor env_info")
            kw = json.loads(d.attrs["env_info"])
            kw["translucent_robot"] = False
            d.attrs["env_args"] = json.dumps(dict(
                type=1, env_name=d.attrs.get("env", kw.get("env_name")),
                env_version=d.attrs.get("robocasa_version", "unknown"),
                robosuite_version=d.attrs.get("robosuite_version", "unknown"),
                mujoco_version=d.attrs.get("mujoco_version", "unknown"),
                env_kwargs=kw), indent=4)
        total = 0
        for ep in d:
            n = d[ep]["actions"].shape[0]
            d[ep].attrs["num_samples"] = n
            total += n
        d.attrs["total"] = total


def extract_obs(dataset: Path, output_name: str = "obs.hdf5", camera_names: Iterable[str] = (
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"),
        size: int = 128, n: Optional[int] = None, timeout: int = 3600) -> Path:
    """Render observations from saved states (robocasa dataset_states_to_obs). Output sits beside dataset."""
    cmd = [_sim_python(), str(DATASET_SCRIPTS / "dataset_states_to_obs.py"),
           "--dataset", str(dataset), "--output_name", output_name,
           "--camera_names", *camera_names, "--camera_height", str(size), "--camera_width", str(size)]
    if n:
        cmd += ["--n", str(n)]
    ensure_robomimic_meta(dataset)
    out = Path(dataset).parent / output_name
    r = _run(cmd, timeout)
    if not out.exists():  # the multiprocess script can exit 0 after a worker error
        raise RuntimeError(f"dataset_states_to_obs produced no {out.name}: {(r.stdout + r.stderr)[-2000:]}")
    return out


def convert_lerobot(dataset: Path, camera_names: Iterable[str] = (
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"),
        size: int = 256, timeout: int = 3600) -> Path:
    """Convert a collect_demos HDF5 to LeRobot format (robocasa convert_hdf5_lerobot).
    Output: <dataset dir>/lerobot (replaced if present)."""
    cmd = [_sim_python(), str(DATASET_SCRIPTS / "convert_hdf5_lerobot.py"),
           "--raw_dataset_path", str(dataset), "--camera_names", *camera_names,
           "--camera_height", str(size), "--camera_width", str(size)]
    ensure_robomimic_meta(dataset)
    out = Path(dataset).parent / "lerobot"
    r = _run(cmd, timeout)
    if not out.exists():
        raise RuntimeError(f"convert_hdf5_lerobot produced no lerobot/: {(r.stdout + r.stderr)[-2000:]}")
    return out


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in SIM_PATHS] + [env.get("PYTHONPATH", "")])
    env.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    # cwd must not be sim/ (sim/robocasa would shadow the package)
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(cmd[1]).name} failed ({r.returncode}): {r.stderr[-2000:]}")
    return r


# ---- credit ledger -------------------------------------------------------------------------

def record_credits(ledger_path: Path, verdicts: Iterable[EpisodeVerdict], player_id: str,
                   player_kind: str, credit_per_pass: float, dataset_id: str = "") -> List[dict]:
    """Append one entry per episode; only passing episodes earn credit. Re-recording an episode is a no-op."""
    if player_kind not in ("human", "ai"):
        raise ValueError("player_kind must be 'human' or 'ai'")
    ledger_path = Path(ledger_path)
    entries = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    known = {(e["dataset_id"], e["episode_id"]) for e in entries}
    added = []
    for v in verdicts:
        key = (dataset_id, v.episode_id)
        if key in known:
            continue
        e = {"dataset_id": dataset_id, "episode_id": v.episode_id, "player_id": player_id,
             "player_kind": player_kind, "verdict": "pass" if v.passed else "fail",
             "reasons": v.reasons, "credit": round(credit_per_pass, 6) if v.passed else 0.0,
             "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        entries.append(e)
        added.append(e)
        known.add(key)
    tmp = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    tmp.replace(ledger_path)
    return added


def player_balance(ledger_path: Path, player_id: str) -> float:
    p = Path(ledger_path)
    if not p.exists():
        return 0.0
    return round(sum(e["credit"] for e in json.loads(p.read_text()) if e["player_id"] == player_id), 6)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="QA a RoboCasa game-episode HDF5")
    ap.add_argument("dataset")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--config", default="{}", help='QAConfig overrides as JSON, e.g. {"min_steps": 10}')
    a = ap.parse_args()
    for v in qa_dataset(Path(a.dataset), cfg=QAConfig(**json.loads(a.config)), replay=a.replay):
        print(json.dumps(v.to_dict(), ensure_ascii=False))
