"""Render a RoboCasa coffee scene to PNGs for a quick look at the background/robot."""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "cgl")  # macOS offscreen

import numpy as np
from PIL import Image

from robocasa.utils.env_utils import create_env
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cafe.cafe_env  # noqa: F401  registers CafeServeCoffee

p = argparse.ArgumentParser()
p.add_argument("--task", default="CoffeeSetupMug")
p.add_argument("--robot", default="PandaOmron")
p.add_argument("--layout", type=int, default=None)
p.add_argument("--style", type=int, default=None)
p.add_argument("--size", type=int, default=768)
p.add_argument("--out", default="renders")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

env = create_env(
    a.task,
    robots=a.robot,
    camera_names=["robot0_agentview_left"],
    camera_widths=64,
    camera_heights=64,
    seed=a.seed,
    layout_ids=a.layout,
    style_ids=a.style,
)
env.reset()
print("layout/style:", getattr(env, "layout_id", None), getattr(env, "style_id", None))
print("lang:", env.get_ep_meta().get("lang"))

os.makedirs(a.out, exist_ok=True)
m = env.sim.model
cams = [m.camera(i).name for i in range(m.ncam)]
print("cameras:", cams)
for cam in cams:
    img = env.sim.render(camera_name=cam, width=a.size, height=a.size)[::-1]
    path = os.path.join(a.out, f"{a.task}_{cam}.png")
    Image.fromarray(np.asarray(img)).save(path)
    print("saved", path)

# free-camera overviews centered on the task object, so the arm doesn't block the view
import mujoco

obj_pos = np.array(env.coffee_machine.pos, dtype=float) if hasattr(env, "coffee_machine") else env.sim.data.body_xpos[env.obj_body_id["obj"]].copy()
mjm = env.sim.model._model
w = int(a.size * 16 / 9)
mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, w)
mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, a.size)
renderer = mujoco.Renderer(mjm, a.size, w)
opt = mujoco.MjvOption()
opt.geomgroup[0] = 0  # hide collision geoms
base_pos = env.sim.data.body_xpos[mjm.body("mobilebase0_base").id] if mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_base") >= 0 else env.robots[0].base_pos
d = obj_pos[:2] - np.asarray(base_pos)[:2]
az0 = np.degrees(np.arctan2(d[1], d[0]))  # direction the robot faces the counter
cam = mujoco.MjvCamera()
cam.lookat[:] = obj_pos + np.array([0, 0, 0.1])
for name, daz, el, dist in [("overview_a", 30, -30, 1.25), ("overview_b", -30, -30, 1.25), ("overview_top", 0, -55, 1.9)]:
    cam.azimuth, cam.elevation, cam.distance = az0 + daz, el, dist
    renderer.update_scene(env.sim.data._data, cam, opt)
    path = os.path.join(a.out, f"{a.task}_s{a.style}_{name}.png")
    Image.fromarray(renderer.render()).save(path)
    print("saved", path)
env.close()
