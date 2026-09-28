"""Mock AI players: synthetic CafeServeCoffee episodes in the real HDF5 layout, no GPU needed.

Stands in for AI players (demo augmentation + policy rollouts) so every downstream stage runs end to end.
Frames are small top-down drawings of the cafe counter (coffee machine, mug, tray, gripper) and trajectories
are scripted through the task stages, so this data is for exercising the pipeline only. Every file is tagged worker="mock-ai-players". About 2% of episodes get a deliberate defect (NaN,
dropped frames, time reversal) so QA has something real to catch. Everything is a deterministic function of
the episode seed.
"""
import json
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from ..dataset_spec import TASK_STEPS
from ..models import EpisodeParams, EpisodeSummary, SimJob, SimResult
from .worker import Progress, SimWorker

H, W = 64, 96
FRAMES_PER_STAGE = 20
DEFECT_RATE = 0.02
SIM_VERSION = "mock-ai-players-0.3"

# Counter-top coordinates in meters (x along the counter, y away from the robot), matching sim/cafe/cafe_env.py.
HOME = np.array([-0.15, -0.25])
CUP = np.array([-0.17, -0.05])
MACHINE = np.array([0.0, 0.08])
TRAY = np.array([-0.4, 0.0])
STAGE_TARGETS = [CUP, TRAY]  # where the gripper ends each stage
FAILURES = [
    "missed the grasp on the mug",
    "mug placed outside the tray or tipped on the rim",
]


def _palette(style_id: int):
    rng = np.random.default_rng(style_id)
    counter = rng.uniform(0.55, 0.95, 3)
    wall = rng.uniform(0.2, 0.5, 3)
    return counter.astype(np.float32), wall.astype(np.float32)


def _to_px(p):
    return int(W / 2 + 12 + p[0] * 90), int(H * 0.62 - p[1] * 90)


def _box(img, c, half, color):
    x, y = c
    img[max(0, y - half[1]):max(0, y + half[1]), max(0, x - half[0]):max(0, x + half[0])] = color


def _frame(p: EpisodeParams, ee, cup, holding: bool):
    counter, wall = _palette(p.style_id)
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = counter
    rgb[: int(H * 0.25)] = wall
    depth = np.full((H, W), 1.0, np.float32)
    _box(rgb, _to_px(TRAY), (12, 8), [0.1, 0.15, 0.55])                    # tray
    _box(rgb, _to_px(MACHINE + [0, 0.1]), (8, 9), [0.15, 0.15, 0.17])      # coffee machine
    _box(rgb, _to_px(cup), (3, 3), [0.92, 0.92, 0.9])                       # mug
    _box(rgb, _to_px(cup), (2, 2), [0.35, 0.2, 0.1])                        # coffee
    ex, ey = _to_px(ee)
    _box(rgb, (ex, ey - 5), (2, 2), [1.0, 0.8, 0.2] if holding else [0.95, 0.95, 0.95])
    depth[max(0, ey - 8):ey - 2, max(0, ex - 3):ex + 3] = 0.6
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth


def _success_prob(p: EpisodeParams) -> float:
    prob = 0.92 - 4.0 * max(0.0, p.placement_jitter_m - 0.02) - 0.01 * (p.layout_id % 7)
    return float(np.clip(prob, 0.1, 0.98))


def simulate_episode(p: EpisodeParams, hz: int, cameras) -> dict:
    rng = np.random.default_rng(p.seed)
    n_stages = len(STAGE_TARGETS)
    jitter = rng.uniform(-p.placement_jitter_m, p.placement_jitter_m, (n_stages, 2))
    targets = [t + j for t, j in zip(STAGE_TARGETS, jitter)]
    success = rng.random() < _success_prob(p)
    stages_done = n_stages if success else int(rng.integers(0, n_stages))
    ee, cup = HOME.copy(), targets[0].copy()
    frames = {c: [] for c in cameras}
    depth, ee_pose, joints, grip, actions, obj, stage_log = [], [], [], [], [], [], []
    starts = [HOME] + targets[:-1]
    for s in range(min(n_stages, stages_done + 1)):  # a failed episode ends during the stage it failed
        a, b = starts[s], targets[s]
        failing = s == stages_done
        for k in range(FRAMES_PER_STAGE):
            u = (k + 1) / FRAMES_PER_STAGE
            if failing:
                u *= rng.uniform(0.4, 0.8)
            pos = a + (b - a) * u + rng.normal(0, 0.003, 2)
            vel = (pos - ee) * hz
            ee = pos
            holding = s == 1 and not failing
            if holding:
                cup = ee.copy()
            if failing and s == 1:
                cup = ee + [0.1, 0.0]
            rgb, d = _frame(p, ee, cup, holding)
            for c in cameras:
                frames[c].append(rgb)
            depth.append(d)
            z = 0.95 + 0.08 * np.sin(np.pi * u) * (s == 1)
            ee_pose.append([ee[0], ee[1], z, 0.0, 1.0, 0.0, 0.0])
            joints.append(np.concatenate([np.tanh([ee[0], ee[1], z, ee[0] * ee[1], u, -u, 0.3]), [0.04, -0.04]]))
            g = 1.0 if holding or (s == 0 and u > 0.9 and not failing) else -1.0
            grip.append([g])
            act = np.zeros(12)
            act[0:2] = np.clip(vel / 0.5, -1, 1)
            act[6] = g
            act[11] = -1.0
            actions.append(act)
            obj.append([cup[0], cup[1], 0.93, 1.0, 0.0, 0.0, 0.0])
            stage_log.append(s)
        if not failing:
            stage_log[-1] = s + 1
    T = len(ee_pose)
    return dict(
        rgb={c: np.asarray(v, np.uint8) for c, v in frames.items()},
        depth=np.asarray(depth, np.float32),
        ee_pose=np.asarray(ee_pose, np.float32),
        joint_state=np.asarray(joints, np.float32),
        gripper_state=np.asarray(grip, np.float32),
        action=np.asarray(actions, np.float32),
        object_pose=np.asarray(obj, np.float32),
        task_stage=np.maximum.accumulate(np.asarray(stage_log, np.int8)),
        sim_step=np.arange(T, dtype=np.int64),
        sim_time=np.arange(T, dtype=np.float64) / hz,
        success=bool(success),
        stages_completed=int(stages_done),
        failure_reason=None if success else f"stage {stages_done + 1} ({TASK_STEPS[stages_done]}): {FAILURES[stages_done]}",
    )


def _defect(rng, ep: dict) -> Optional[str]:
    if rng.random() >= DEFECT_RATE:
        return None
    kind = rng.choice(["nan", "dropped_frames", "time_reversal"])
    if kind == "nan":
        ep["ee_pose"][len(ep["ee_pose"]) // 2, 0] = np.nan
    elif kind == "dropped_frames":
        for c in ep["rgb"]:
            ep["rgb"][c] = ep["rgb"][c][:-3]
    else:
        ep["sim_time"][5:8] = ep["sim_time"][5:8][::-1]
    return str(kind)


class MockSimWorker(SimWorker):
    name = "mock-ai-players"

    def run(self, job: SimJob, out_path: Path, progress: Optional[Progress] = None) -> SimResult:
        t0 = time.time()
        hz = job.plan.control_hz
        cameras = [c.name for c in job.plan.cameras]
        sensors = set(job.sensors)
        summaries = []
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs.update(worker=self.name, simulator_version=SIM_VERSION, scene_template=job.plan.scene_template,
                           task=job.plan.task, robot=job.plan.robot, cameras=json.dumps(cameras))
            eps = f.create_group("episodes")
            for i, p in enumerate(job.episodes):
                ep = simulate_episode(p, hz, cameras)
                defect = _defect(np.random.default_rng(p.seed + 7), ep)
                g = eps.create_group(p.episode_id)
                g.create_dataset("sim_step", data=ep["sim_step"])
                g.create_dataset("sim_time", data=ep["sim_time"])
                g.create_dataset("ee_pose", data=ep["ee_pose"])
                g.create_dataset("task_stage", data=ep["task_stage"])
                for name in ("joint_state", "gripper_state", "action", "object_pose", "depth"):
                    if name in sensors or name in ("joint_state", "action"):
                        g.create_dataset(name, data=ep[name], compression="gzip")
                # the first camera is also stored as "rgb" so single-camera tools keep working
                g.create_dataset("rgb", data=ep["rgb"][cameras[0]], compression="gzip")
                for c in cameras:
                    g.create_dataset(f"rgb_{c}", data=ep["rgb"][c], compression="gzip")
                g.attrs.update(success=ep["success"], stages_completed=ep["stages_completed"],
                               failure_reason=ep["failure_reason"] or "", params=p.model_dump_json(),
                               player="ai", defect=defect or "")
                summaries.append(EpisodeSummary(
                    episode_id=p.episode_id, seed=p.seed, success=ep["success"], failure_reason=ep["failure_reason"],
                    steps=len(ep["sim_step"]), stages_completed=ep["stages_completed"]))
                if progress and (i % 10 == 9 or i == len(job.episodes) - 1):
                    progress(i + 1, len(job.episodes))
        return SimResult(job_id=job.job_id, worker=self.name, simulator_version=SIM_VERSION,
                         dataset_path=str(out_path), episodes=summaries, compute_seconds=time.time() - t0)
