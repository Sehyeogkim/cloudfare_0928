"""QA checks computed from the dataset itself. The QA Agent interprets these numbers;
it never produces them. "pass" means the data meets the contract and the simulation success rule,
not that it improves a real robot.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from .collection import SENSOR_DATASETS
from .dataset_spec import TASK_STEPS, DatasetSpec
from .models import CoverageBin, EpisodeParams, EpisodeQA, QACheck, QAMetrics, SimJob
from .sim.worker import SimWorker

ACTION_LIMIT = 1.0 + 1e-6  # OSC delta actions are normalized to [-1, 1]
REPRO_SAMPLES = 3
N_STAGES = len(TASK_STEPS)


def _check_episode(g: h5py.Group, spec: DatasetSpec) -> EpisodeQA:
    reasons: List[str] = []
    required = ["sim_step", "sim_time", "ee_pose", "task_stage"] + [SENSOR_DATASETS[s] for s in spec.sensors]
    required += [f"rgb_{c.name}" for c in spec.data.cameras]
    required = list(dict.fromkeys(required))
    for name in required:
        if name not in g:
            reasons.append(f"missing {name}")
    present = [n for n in required if n in g]
    lengths = {n: len(g[n]) for n in present}
    if len(set(lengths.values())) > 1:
        short = min(lengths, key=lengths.get)
        reasons.append(f"frame count mismatch ({short} has {lengths[short]} of {max(lengths.values())})")
    for n in present:
        arr = g[n][()]
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            reasons.append(f"NaN/Inf in {n}")
    if "sim_time" in g and np.any(np.diff(g["sim_time"][()]) <= 0):
        reasons.append("time not monotonic")
    if "sim_step" in g and np.any(np.diff(g["sim_step"][()]) <= 0):
        reasons.append("sim_step not monotonic")
    if "action" in g:
        a = g["action"][()]
        if np.nanmax(np.abs(a)) > ACTION_LIMIT:
            reasons.append("action out of range")

    labeled = bool(g.attrs.get("success", False))
    success = labeled
    if "task_stage" in g:
        stages = g["task_stage"][()]
        if np.any(np.diff(stages.astype(int)) < 0):
            reasons.append("task stage went backwards")
        success = int(stages[-1]) >= N_STAGES
        if success != labeled:
            reasons.append("success label disagrees with the recorded task stages")
    return EpisodeQA(episode_id=g.name.split("/")[-1], valid=not reasons, success=success, reasons=reasons)


def _status(bad: int, total: int, warn_frac: float = 0.0, fail_frac: float = 0.05) -> str:
    frac = bad / max(total, 1)
    return "pass" if frac <= warn_frac else ("warn" if frac <= fail_frac else "fail")


def _categorical(values: np.ndarray, ok: np.ndarray, ids: List[int]) -> List[CoverageBin]:
    bins = []
    for i in ids:
        sel = values == i
        n = int(sel.sum())
        bins.append(CoverageBin(lo=float(i), hi=float(i), count=n,
                                success_rate=round(float(ok[sel].mean()), 3) if n else None))
    return bins


def _reproduce(dataset: Path, job: SimJob, worker: SimWorker, valid_ids: List[str], workdir: Path) -> Dict[str, int]:
    picks = valid_ids[:: max(1, len(valid_ids) // REPRO_SAMPLES)][:REPRO_SAMPLES]
    if not picks:
        return {"checked": 0, "matched": 0}
    by_id = {e.episode_id: e for e in job.episodes}
    sub = job.model_copy(update={"job_id": job.job_id + "-repro", "episodes": [by_id[i] for i in picks if i in by_id]})
    out = workdir / "repro.hdf5"
    worker.run(sub, out)
    matched = 0
    with h5py.File(dataset, "r") as a, h5py.File(out, "r") as b:
        for i in picks:
            ga, gb = a["episodes"][i], b["episodes"].get(i)
            if gb is None:
                continue
            same = all(
                n in gb and np.allclose(ga[n][()], gb[n][()], equal_nan=True, atol=1e-5)
                for n in ("joint_state", "action", "ee_pose", "task_stage")
            ) and bool(ga.attrs["success"]) == bool(gb.attrs["success"])
            matched += same
    out.unlink(missing_ok=True)
    return {"checked": len(picks), "matched": matched}


def run_checks(dataset: Path, job: SimJob, spec: DatasetSpec, worker: Optional[SimWorker], workdir: Path) -> QAMetrics:
    with h5py.File(dataset, "r") as f:
        groups = [f["episodes"][n] for n in sorted(f["episodes"])]
        eps = [_check_episode(g, spec) for g in groups]
        params = [EpisodeParams.model_validate_json(g.attrs["params"]) for g in groups]
        labeled_reasons = [str(g.attrs.get("failure_reason", "")) for g in groups]
        stages_done = np.array([int(g["task_stage"][-1]) if "task_stage" in g else 0 for g in groups])

    total = len(eps)
    valid = [e for e in eps if e.valid]
    ok = np.array([e.success for e in eps], bool)
    layouts = np.array([p.layout_id for p in params])
    styles = np.array([p.style_id for p in params])
    coverage = {
        "layout_id": _categorical(layouts, ok, list(job.plan.layout_ids)),
        "style_id": _categorical(styles, ok, list(job.plan.style_ids)),
        "stages_completed": _categorical(stages_done, ok, list(range(N_STAGES + 1))),
    }
    dup = total - len({p.seed for p in params})
    failure_reasons = Counter(
        reason.split(":")[0] for e, reason in zip(eps, labeled_reasons) if not e.success and reason
    )
    scene_bins = coverage["layout_id"] + coverage["style_id"]

    def count(pred) -> int:
        return sum(1 for e in eps if any(pred(x) for x in e.reasons))

    stage_rates = [float((stages_done >= k).mean()) if total else 0.0 for k in range(1, N_STAGES + 1)]
    checks = [
        QACheck(name="Requested streams present", category="structure",
                status=_status(count(lambda x: x.startswith("missing")), total),
                detail=f"{count(lambda x: x.startswith('missing'))} of {total} episodes missing a camera or sensor stream"),
        QACheck(name="Frame counts consistent", category="structure",
                status=_status(count(lambda x: "frame count" in x), total),
                detail=f"{count(lambda x: 'frame count' in x)} episodes with mismatched stream lengths (dropped frames)"),
        QACheck(name="No NaN/Inf values", category="structure",
                status=_status(count(lambda x: "NaN" in x), total),
                detail=f"{count(lambda x: 'NaN' in x)} episodes with NaN/Inf"),
        QACheck(name="Time monotonic", category="structure",
                status=_status(count(lambda x: "monotonic" in x), total),
                detail=f"{count(lambda x: 'monotonic' in x)} episodes with time going backwards"),
        QACheck(name="Actions within limits", category="structure",
                status=_status(count(lambda x: "action" in x), total),
                detail="normalized OSC actions within [-1, 1]"),
        QACheck(name="Success labels match task stages", category="task",
                status=_status(count(lambda x: "label" in x or "stage went" in x), total),
                detail="success recomputed from the recorded task stage (all task stages completed)"),
        QACheck(name="Success rate", category="task", status="pass",
                detail=(f"{ok.sum()} of {total} episodes complete all {N_STAGES} stages ({ok.mean():.0%}); per stage: "
                        + " · ".join(f"{i + 1}:{r:.0%}" for i, r in enumerate(stage_rates))) if total else "no episodes"),
        QACheck(name="Scene coverage (layouts × styles)", category="diversity",
                status="pass" if all(b.count for b in scene_bins) else "warn",
                detail="every planned layout and style has episodes" if all(b.count for b in scene_bins)
                else "some planned layouts/styles have no episodes"),
        QACheck(name="Unique seeds", category="diversity", status="pass" if dup == 0 else "fail",
                detail=f"{dup} duplicate seeds"),
    ]
    repro = _reproduce(dataset, job, worker, [e.episode_id for e in valid], workdir) if worker else {"checked": 0, "matched": 0}
    checks.append(QACheck(
        name="Reproducible from seed", category="reproducibility",
        status="pass" if repro["checked"] and repro["matched"] == repro["checked"] else ("warn" if not repro["checked"] else "fail"),
        detail=f"re-ran {repro['checked']} episodes with the same seed and scene; {repro['matched']} matched",
    ))
    return QAMetrics(
        episodes_total=total, episodes_valid=len(valid),
        success_rate=round(float(ok.mean()), 4) if total else 0.0,
        failure_reasons=dict(failure_reasons), coverage=coverage, duplicate_seeds=dup,
        repro_checked=repro["checked"], repro_matched=repro["matched"], checks=checks, episodes=eps,
    )
