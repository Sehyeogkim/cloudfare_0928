"""Data Collection stage: package simulator output into the delivered dataset.

Works on any HDF5 that follows docs/sim_worker.md, whichever worker produced it.
"""
import hashlib
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
from PIL import Image

from .models import Manifest

# DatasetSpec sensor name -> dataset name inside each episode group
SENSOR_DATASETS = {
    "rgb": "rgb",
    "depth": "depth",
    "joint_state": "joint_state",
    "eef_pose": "ee_pose",
    "gripper_state": "gripper_state",
    "action": "action",
    "object_pose": "object_pose",
    "task_stage": "task_stage",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_into(main: Path, extra: Path) -> None:
    """Copy every episode group of `extra` into `main` (used for top-up runs)."""
    with h5py.File(main, "a") as dst, h5py.File(extra, "r") as src:
        for name in src["episodes"]:
            if name in dst["episodes"]:
                raise ValueError(f"episode {name} already exists in {main}")
            src.copy(src["episodes"][name], dst["episodes"], name=name)


def export(src_path: Path, dst_path: Path, keep: Iterable[str]) -> None:
    """Write a copy of the dataset holding only the `keep` episodes."""
    keep = set(keep)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        dst.attrs.update(dict(src.attrs))
        eps = dst.create_group("episodes")
        for name in sorted(src["episodes"]):
            if name in keep:
                src.copy(src["episodes"][name], eps, name=name)


def build_manifest(order_id: str, path: Path, sensors: List[str]) -> Manifest:
    frames = 0
    missing = set()
    with h5py.File(path, "r") as f:
        episodes = f["episodes"]
        for name in episodes:
            g = episodes[name]
            frames += len(g["sim_step"]) if "sim_step" in g else 0
            missing.update(s for s in sensors if SENSOR_DATASETS[s] not in g)
        attrs = dict(f.attrs)
        count = len(episodes)
    return Manifest(
        order_id=order_id,
        worker=str(attrs.get("worker", "unknown")),
        simulator_version=str(attrs.get("simulator_version", "unknown")),
        scene_template=str(attrs.get("scene_template", "unknown")),
        episodes=count,
        frames=frames,
        sensors=sensors,
        size_bytes=path.stat().st_size,
        sha256=sha256(path),
        missing_sensors=sorted(missing),
    )


def preview(path: Path, out: Path, n: int = 6) -> bool:
    """Contact sheet of first/last RGB frames for n episodes spread across the dataset."""
    with h5py.File(path, "r") as f:
        names = sorted(f["episodes"])
        if not names or "rgb" not in f["episodes"][names[0]]:
            return False
        picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, min(n, len(names)))]
        tiles = []
        for name in picks:
            rgb = f["episodes"][name]["rgb"]
            tiles.append(np.concatenate([rgb[0], rgb[len(rgb) - 1]], axis=0))
    sheet = np.concatenate(tiles, axis=1)
    img = Image.fromarray(sheet)
    scale = max(1, 768 // img.width)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return True
