"""RoboCasa env construction and episode recording in the collect_demos HDF5 format.

The recorder follows robocasa/scripts/collect_demos.py + robosuite DataCollectionWrapper so that
agriphilo.game_qa can replay the file: data.attrs env/env_info, per-demo model_file (after reset),
ep_meta, states recorded BEFORE each action, and actions.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# Repo dirs, not sim/: sim/robocasa would shadow the real package (sim/robocasa/robocasa).
# The checkouts moved to sim/third_party/ on 2026-09-28; both layouts are searched.
SIM_PATHS = [REPO / "sim" / base / name for base in ("third_party", ".") for name in ("robocasa", "robosuite")]


def import_sim():
    for p in reversed(SIM_PATHS):
        if (p / p.name).is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    import robosuite  # noqa: F401
    import robocasa  # noqa: F401  (registers kitchen envs)
    return robosuite


def env_config(task: str, robot: str, layout_id: int, style_id: int) -> dict:
    """The robosuite.make kwargs that get stored as env_info (renderer flags excluded)."""
    import_sim()
    from robosuite.controllers import load_composite_controller_config

    return {
        "env_name": task,
        "robots": robot,
        "controller_configs": load_composite_controller_config(controller=None, robot=robot),
        "layout_ids": layout_id,
        "style_ids": style_id,
    }


def make_env(config: dict, seed: int, render: bool):
    robosuite = import_sim()
    return robosuite.make(**config, has_renderer=False, has_offscreen_renderer=render,
                          use_camera_obs=False, ignore_done=True, control_freq=20, seed=seed)


class Recorder:
    """Wraps env.step: records the flattened sim state before each action, plus the action.

    It never renders: offscreen rendering mid-episode perturbs MuJoCo's warmstart and breaks
    bit-exact replay. Video is rendered afterwards from the saved states (render_video).
    """

    def __init__(self, env):
        self.env = env
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.model_xml = ""
        self.ep_meta: Dict = {}

    def start(self) -> None:
        """Same sequence as DataCollectionWrapper._start_new_episode."""
        env = self.env
        env.reset()
        self.model_xml = env.sim.model.get_xml()
        self.ep_meta = env.get_ep_meta()
        state = np.array(env.sim.get_state().flatten())
        env.set_ep_meta(self.ep_meta)
        env.reset_from_xml_string(self.model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(state)  # the xml alone does not hold object placements
        env.sim.forward()
        if hasattr(env, "update_sites"):
            env.update_sites()

    def step(self, action: np.ndarray):
        self.states.append(np.array(self.env.sim.get_state().flatten()))
        a = np.asarray(action, dtype=np.float64)
        self.actions.append(a)
        return self.env.step(a)

    def save(self, path: Path, env_info: dict, demo_attrs: Dict) -> Path:
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            d = f.create_group("data")
            d.attrs["env"] = env_info["env_name"]
            d.attrs["env_info"] = json.dumps(env_info)
            g = d.create_group("demo_1")
            g.attrs["model_file"] = self.model_xml
            g.attrs["ep_meta"] = json.dumps(self.ep_meta)
            for k, v in demo_attrs.items():
                g.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
            g.create_dataset("states", data=np.array(self.states))
            g.create_dataset("actions", data=np.array(self.actions))
        return path



def render_video(hdf5: Path, out: Path, camera: str = "robot0_agentview_center", every: int = 3,
                 size: int = 384, fps: int = 20) -> Optional[Path]:
    """Render frames from the saved states (not live), so recording stays replayable."""
    import h5py
    import imageio

    import_sim()
    from .. import game_qa

    with h5py.File(hdf5, "r") as f:
        kw = game_qa.env_kwargs_from_file(f)
        kw["has_offscreen_renderer"] = True
        env = import_sim().make(**kw)
        g = f["data/demo_1"]
        states = g["states"][()]
        game_qa._reset_to(env, g.attrs["model_file"], g.attrs.get("ep_meta"), states[0])
        frames = []
        for t in list(range(0, len(states), every)) + [len(states) - 1]:
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            frames.append(np.asarray(env.sim.render(camera_name=camera, width=size, height=size))[::-1])
        env.close()
    try:
        imageio.mimsave(str(out), frames, fps=max(1, fps // every))
    except Exception:
        out = out.with_suffix(".gif")
        imageio.mimsave(str(out), frames, duration=every / fps)
    from PIL import Image
    for name, fr in (("first", frames[0]), ("last", frames[-1])):
        Image.fromarray(fr).save(out.with_name(f"{out.stem}_{name}.png"))
    return out
