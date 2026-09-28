"""Game server for the CafeServeCoffee scene (Human play for agriphilo /games).

One shared MuJoCo sim streams JPEG frames over WebSockets to every client:
  ?role=player  keyboard control + submit episode
  ?role=viewer  watch only (embedded in the data needer's order page)
Other query params come from agriphilo's humanPanel: game_id, task, robot, layout_ids, style_ids.

Submitted episodes are collect_demos-format HDF5 written to runs/episodes/<game_id>/ (see agriphilo/game_qa.py),
which agriphilo picks up for QA and billing.

Run:  sim/.venv/bin/python sim/web/server.py   ->  http://localhost:8010/?role=player
"""
import asyncio
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")

import h5py
import mujoco
import numpy as np
import robosuite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image
from robosuite.controllers import load_composite_controller_config

SIM_DIR = Path(__file__).resolve().parent.parent
REPO = SIM_DIR.parent
sys.path.insert(0, str(SIM_DIR))  # for cafe/; the robosuite/robocasa repos live in sim/third_party
import cafe.cafe_env  # noqa: E402,F401  registers CafeServeCoffee

EPISODES = REPO / "runs" / "episodes"
W, H, PIP = 960, 540, 240
CONTROL_HZ = 20
OBS_CAMS = ["robot0_agentview_center", "robot0_eye_in_hand"]
OBS_RES, OBS_EVERY = 128, 2  # camera frames saved with the episode, at CONTROL_HZ / OBS_EVERY

# key -> (index in [dx, dy, dz, droll, dpitch, dyaw], sign); arm frame is the robot base
ARM_KEYS = {
    "w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "r": (2, 1), "f": (2, -1),
    "q": (5, 1), "e": (5, -1), "z": (4, 1), "c": (4, -1),
}
BASE_KEYS = {"i": (0, 1), "k": (0, -1), "j": (1, 1), "l": (1, -1), "u": (2, 1), "o": (2, -1)}


def game_config(params) -> dict:
    first = lambda v, d: int(str(v).split(",")[0]) if v else d
    return dict(
        game_id=params.get("game_id") or "local",
        task=params.get("task") or "CafeServeCoffee",
        robot=params.get("robot") or "PandaOmron",
        layout=first(params.get("layout_ids"), 9),
        style=first(params.get("style_ids"), 3),
    )


class Sim:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.env_info = dict(
            env_name=cfg["task"],
            robots=cfg["robot"],
            controller_configs=load_composite_controller_config(controller=None, robot=cfg["robot"]),
            layout_ids=cfg["layout"],
            style_ids=cfg["style"],
            control_freq=CONTROL_HZ,
            translucent_robot=False,
        )
        self.env = robosuite.make(
            **self.env_info, has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, ignore_done=True, seed=0,
        )
        self.keys, self.grip, self.view = set(), -1.0, "overview"
        self.player_id = "human-demo"
        self.last_saved = None
        self.reset()

    # ---- episode lifecycle -------------------------------------------------------------
    def reset(self):
        self.env.reset()
        self.robot = self.env.robots[0]
        m = self.env.sim.model._model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
        self.main = mujoco.Renderer(m, H, W)
        self.pip = mujoco.Renderer(m, PIP, PIP)
        self.obs_r = mujoco.Renderer(m, OBS_RES, OBS_RES)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0  # hide collision geoms
        self.cam = mujoco.MjvCamera()
        cm = np.array(self.env.coffee_machine.pos, dtype=float)
        d = cm[:2] - self.env.sim.data.body_xpos[m.body("mobilebase0_base").id][:2]
        self.az0 = float(np.degrees(np.arctan2(d[1], d[0])))
        self.cam.lookat[:] = cm + np.array([0, 0, 0.05])
        self.set_view(self.view)
        self.grip, self.nframe, self.last_pip = -1.0, 0, None
        self.t0 = time.time()
        self.model_xml = self.env.sim.model.get_xml()
        self.ep_meta = json.dumps(self.env.get_ep_meta())
        self.states, self.actions = [], []
        self.obs = {c: [] for c in OBS_CAMS}

    def set_view(self, v):
        presets = {"overview": (0, -40, 1.5), "left": (40, -28, 1.3), "right": (-40, -28, 1.3), "top": (0, -70, 1.6)}
        self.view = v if v in presets else "overview"
        az, el, dist = presets[self.view]
        self.cam.azimuth, self.cam.elevation, self.cam.distance = self.az0 + az, el, dist

    def action(self):
        arm, base = np.zeros(6), np.zeros(3)
        for k in self.keys:
            if k in ARM_KEYS:
                i, s = ARM_KEYS[k]
                arm[i] += s
            if k in BASE_KEYS:
                i, s = BASE_KEYS[k]
                base[i] += s
        arm *= 0.2 if "shift" in self.keys else 0.6  # shift = fine control
        moving_base = bool(np.any(base))
        ac = {
            "right": np.zeros(6) if moving_base else arm,
            "right_gripper": np.array([self.grip]),
            "base": base * 0.6,
            "base_mode": np.array([1 if moving_base else -1]),
        }
        if "torso" in self.robot._action_split_indexes:
            a, b = self.robot._action_split_indexes["torso"]
            ac["torso"] = np.zeros(b - a)
        return self.robot.create_action_vector(ac)

    def step(self):
        a = self.action()
        self.states.append(self.env.sim.get_state().flatten())  # state before action t
        self.actions.append(a)
        self.env.step(a)  # stages advance inside CafeServeCoffee._post_action
        if len(self.actions) % OBS_EVERY == 1:
            for c in OBS_CAMS:
                self.obs[c].append(self._render(self.obs_r, camera=c, fx=False))

    def submit(self) -> str:
        """Write the episode as collect_demos-format HDF5 into runs/episodes/<game_id>/."""
        out_dir = EPISODES / self.cfg["game_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"human_{datetime.now():%Y%m%d_%H%M%S}.hdf5"
        tmp = path.with_suffix(".hdf5.tmp")
        with h5py.File(tmp, "w") as f:
            data = f.create_group("data")
            data.attrs.update(
                env=self.cfg["task"], env_info=json.dumps(self.env_info),
                date=datetime.now().strftime("%Y-%m-%d"), time=datetime.now().strftime("%H:%M:%S"),
                robosuite_version=getattr(robosuite, "__version__", "master"),
                mujoco_version=mujoco.__version__,
            )
            demo = data.create_group("demo_1")
            demo.create_dataset("states", data=np.asarray(self.states))
            demo.create_dataset("actions", data=np.asarray(self.actions))
            if self.obs[OBS_CAMS[0]]:
                obs = demo.create_group("obs")  # preview frames; QA ignores this group
                for c in OBS_CAMS:
                    obs.create_dataset(c, data=np.asarray(self.obs[c]), compression="gzip")
                obs.attrs.update(every_n_steps=OBS_EVERY)
            demo.attrs.update(
                model_file=self.model_xml, ep_meta=self.ep_meta,
                player_id=self.player_id, player_kind="human",
                success=bool(self.env._check_success()), stage=int(self.env.stage),
            )
        tmp.rename(path)  # agriphilo only picks up *.hdf5, so it never sees a half-written file
        self.last_saved = os.path.relpath(path, REPO)
        return self.last_saved

    # ---- rendering -----------------------------------------------------------------------
    def _render(self, r, cam=None, camera=None, fx=True):
        d = self.env.sim.data._data
        if cam is not None:
            r.update_scene(d, cam, self.opt)
        else:
            r.update_scene(d, camera=camera, scene_option=self.opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = fx
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = fx
        return r.render()

    def frame(self) -> bytes:
        out = self._render(self.main, cam=self.cam).copy()
        self.nframe += 1
        if self.nframe % 2 == 1 or self.last_pip is None:
            self.last_pip = self._render(self.pip, camera="robot0_eye_in_hand", fx=False)
        out[H - PIP - 12 : H - 12, W - PIP - 12 : W - 12] = self.last_pip
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=75)
        return buf.getvalue()

    def status(self) -> dict:
        return dict(
            game_id=self.cfg["game_id"], task=self.cfg["task"], layout=self.cfg["layout"], style=self.cfg["style"],
            stage=int(self.env.stage), stages=self.env.STAGES, grip="closed" if self.grip > 0 else "open",
            steps=len(self.actions), view=self.view, saved=self.last_saved,
            elapsed=round(time.time() - self.t0, 1),
        )


app = FastAPI()
sim: Sim = None
clients = {}  # WebSocket -> role
loop_task = None


async def sim_loop():
    """Step while a player is connected; broadcast frames to everyone."""
    period = 1.0 / CONTROL_HZ
    while clients:
        t = time.perf_counter()
        if "player" in clients.values():
            sim.step()
        else:
            sim.keys = set()
        frame, status = sim.frame(), json.dumps(sim.status())
        for ws in list(clients):
            try:
                await ws.send_bytes(frame)
                await ws.send_text(status)
            except Exception:
                clients.pop(ws, None)
        await asyncio.sleep(max(0.005, period - (time.perf_counter() - t)))


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(sock: WebSocket):
    global sim, loop_task
    await sock.accept()
    params = dict(sock.query_params)
    role = "player" if params.get("role") == "player" else "viewer"
    cfg = game_config(params)
    if sim is None or (role == "player" and cfg != sim.cfg):
        sim = Sim(cfg)  # same thread as rendering: the CGL context is thread-bound
    if role == "player" and params.get("player_id"):
        sim.player_id = params["player_id"]
    clients[sock] = role
    if loop_task is None or loop_task.done():
        loop_task = asyncio.create_task(sim_loop())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if role != "player":
                continue
            t = msg.get("type")
            if t == "keys":
                sim.keys = set(msg["keys"])
            elif t == "grip":
                sim.grip = -sim.grip
            elif t == "view":
                sim.set_view(msg["view"])
            elif t == "reset":
                sim.reset()
            elif t == "submit":
                sim.submit()
                sim.reset()
    except WebSocketDisconnect:
        pass
    finally:
        clients.pop(sock, None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8010)))
