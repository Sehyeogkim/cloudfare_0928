"""Skill primitives: scripted low-level controllers that turn one plan step into robosuite actions.

A Scene adapter maps task-level entity names ("mug", "coffee_dispenser") to poses read from the sim
state, so the same primitives can run on another RoboCasa task by adding an adapter.
Actions follow the PandaOmron HYBRID_MOBILE_BASE layout: [arm OSC delta(6, base frame), gripper(1),
base(3), torso(1), mode(1)]; the base and torso stay still and mode=-1 (arm mode).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

POS_SCALE = 0.05   # OSC output_max for position: action 1.0 == 5 cm goal offset
ROT_SCALE = 0.5    # OSC output_max for rotation (rad)
MAX_STEP_M = 0.015  # cap on commanded move per control step (20 Hz -> 0.3 m/s)
OPEN, CLOSE = -1.0, 1.0
import os as _os


def _band(env_name: str, default):
    v = _os.environ.get(env_name)
    return tuple(float(x) for x in v.split(",")) if v else default


# (min, desired, max) forward distance base->point before grasping / placing. The arm can't reach
# low points near the base. A fixed 0.45 m for both beat looser bands in a sweep on CoffeeSetupMug
# (8/8 seeds vs 1-4/8). Override via env (AGRI_REACH="lo,desired,hi") for experiments.
REACH_M = _band("AGRI_REACH", (0.44, 0.45, 0.46))
PLACE_REACH_M = _band("AGRI_PLACE_REACH", (0.44, 0.45, 0.46))
XY_ALIGN_M = 0.025  # contact with fixtures leaves ~1-2 cm; RoboCasa pour check allows 4 cm


@dataclass
class SkillResult:
    ok: bool
    message: str
    steps: int


def _axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang = np.arccos(cos)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return v * ang


# ---- scene adapters --------------------------------------------------------------------------

class Scene:
    """Task adapter: entity names the planner may use and how to read their poses."""

    task = ""
    objects: Dict[str, str] = {}   # planner name -> env.objects key (graspable)
    targets: Dict[str, str] = {}   # planner name -> description (place targets)

    def __init__(self, env):
        self.env = env

    def object_pos(self, name: str) -> np.ndarray:
        key = self.objects[name]
        return np.array(self.env.sim.data.body_xpos[self.env.obj_body_id[key]])

    def grasp_point(self, name: str, robot_xy: np.ndarray):
        """(eef position, horizontal finger-closing direction) for grasping `name`."""
        return self.object_pos(name), None

    def target_pos(self, name: str) -> np.ndarray:
        """Where the held object's bottom-centre should end up (z = the supporting surface)."""
        raise NotImplementedError

    def object_half_height(self, name: str) -> float:
        m = self.env.sim.model
        key = self.objects[name]
        try:
            return float(m.geom_size[m.geom_name2id(f"{key}_reg_bbox")][2])
        except Exception:
            return 0.05

    def object_bottom(self, name: str) -> np.ndarray:
        return self.object_pos(name) - np.array([0, 0, self.object_half_height(name)])

    def success(self) -> bool:
        return bool(self.env._check_success())

    def describe(self) -> Dict:
        r = lambda v: [round(float(x), 3) for x in v]
        return {
            "task": self.task,
            "instruction": self.env.get_ep_meta().get("lang", ""),
            "objects": {n: {"position_m": r(self.object_pos(n))} for n in self.objects},
            "targets": {n: {"position_m": r(self.target_pos(n)), "about": d} for n, d in self.targets.items()},
        }


class CoffeeSetupMugScene(Scene):
    task = "CoffeeSetupMug"
    objects = {"mug": "obj"}
    targets = {"coffee_dispenser": ("drip tray under the coffee machine nozzle where the mug must stand. The "
                                    "nozzle overhang is only ~11 cm above the tray: carry the mug with its "
                                    "bottom 3-6 cm above the tray, never higher.")}
    # RoboCasa counts the mug as placed when its centre is within 4 cm (xy) of the pour site. The Panda
    # hand does not fit under the nozzle overhang, so the mug is set down 2.5 cm short of the site,
    # on the robot's side.
    STANDOFF_M = 0.025

    def _site(self) -> np.ndarray:
        cm = self.env.coffee_machine
        sid = self.env.sim.model.site_name2id(cm.naming_prefix + "receptacle_place_site")
        return np.array(self.env.sim.data.site_xpos[sid])

    def _tray_top(self, site: np.ndarray) -> float:
        """Top of the highest machine collision geom under the site (the drip tray grate)."""
        m, d = self.env.sim.model, self.env.sim.data
        best = site[2]
        top_seen = -np.inf
        for i in range(m.ngeom):
            if not m.geom_contype[i] or "coffee" not in m.body(m.geom_bodyid[i]).name:
                continue
            R = d.geom_xmat[i].reshape(3, 3)
            c = d.geom_xpos[i] + R @ m.geom_aabb[i][:3]
            h = np.abs(R) @ m.geom_aabb[i][3:]
            inside = np.all(np.abs(site[:2] - c[:2]) <= h[:2] + 0.02)
            top = c[2] + h[2]
            if inside and top < site[2] + 0.03 and top > top_seen:
                top_seen = top
        return float(top_seen) if np.isfinite(top_seen) else float(best)

    def target_pos(self, name: str) -> np.ndarray:
        site = self._site()
        robot_xy = np.array(self.env.robots[0].part_controllers["right"].origin_pos)[:2]
        u = robot_xy - site[:2]
        u /= np.linalg.norm(u) + 1e-9
        # The success check uses the mug body origin (its bbox centre, handle included).
        return np.array([*(site[:2] + self.STANDOFF_M * u), self._tray_top(site)])

    def grasp_point(self, name: str, robot_xy: np.ndarray):
        """Side-rim grasp: one finger inside the cup, one outside, on the side away from the handle.

        The fingers close along their current axis, so the hand's long side stays parallel to the
        machine front and clear of the nozzle overhang."""
        m, d = self.env.sim.model, self.env.sim.data
        key = self.objects[name]
        body = self.object_pos(name)
        try:
            gid = m.geom_name2id(f"{key}_reg_int")
            return np.array(d.geom_xpos[gid]), float(m.geom_size[gid][0])
        except Exception:
            pass
        try:  # no interior region: assume the cup fills the bbox, handle ignored
            gid = m.geom_name2id(f"{key}_reg_bbox")
            half = m.geom_size[gid]
            return np.array(d.geom_xpos[gid]), float(min(half[0], half[1])) - 0.012
        except Exception:
            return body, None


SCENES = {"CoffeeSetupMug": CoffeeSetupMugScene}


# ---- primitives ------------------------------------------------------------------------------

class Skills:
    PRIMITIVES = ("move_above", "grasp", "lift", "carry_above", "lower_onto", "release", "retreat")

    def __init__(self, env, recorder, scene: Scene):
        self.env, self.rec, self.scene = env, recorder, scene
        self.robot = env.robots[0]
        self.arm = self.robot.part_controllers["right"]
        self.eef_site = self.robot.eef_site_id["right"]
        self.gripper_cmd = OPEN
        self.hold_ori = self._eef_ori().copy()
        self.held: Optional[str] = None
        self.log: List[Dict] = []

    # -- state
    def _eef(self) -> np.ndarray:
        return np.array(self.env.sim.data.site_xpos[self.eef_site])

    def _eef_ori(self) -> np.ndarray:
        return np.array(self.env.sim.data.site_xmat[self.eef_site]).reshape(3, 3)

    def _finger_axis(self) -> np.ndarray:
        m, d = self.env.sim.model, self.env.sim.data
        a = d.geom_xpos[m.geom_name2id("gripper0_right_finger1_pad_collision")]
        b = d.geom_xpos[m.geom_name2id("gripper0_right_finger2_pad_collision")]
        return np.array(a - b)

    def _grasping(self, name: str) -> bool:
        obj = self.env.objects[self.scene.objects[name]]
        return bool(self.env._check_grasp(gripper=self.robot.gripper["right"], object_geoms=obj))

    # -- low level
    def _action(self, dpos_world: np.ndarray) -> np.ndarray:
        R_base = np.array(self.arm.origin_ori)
        a = np.zeros(self.env.action_dim)
        a[0:3] = np.clip(R_base.T @ dpos_world / POS_SCALE, -1, 1)
        rot_err_world = _axis_angle(self.hold_ori @ self._eef_ori().T)
        a[3:6] = np.clip(R_base.T @ rot_err_world / ROT_SCALE, -1, 1)
        a[6] = self.gripper_cmd
        a[-1] = -1.0
        return a

    def _servo(self, target: np.ndarray, max_steps: int = 200, tol: float = 0.008,
               track: Optional[str] = None) -> int:
        """Drive the eef, or a held object (track=name, closed loop on its live pose), to target.

        Returns steps used, or -1 if the tolerance was not reached."""
        for i in range(max_steps):
            cur = self.scene.object_pos(track) if track else self._eef()
            err = target - cur
            if np.linalg.norm(err) < tol:
                return i
            n = np.linalg.norm(err)
            step = err if n <= MAX_STEP_M else err / n * MAX_STEP_M
            self.rec.step(self._action(step * 1.6))
        return -1

    def _forward_dist(self, point: np.ndarray) -> float:
        fwd = np.array(self.arm.origin_ori)[:, 0]
        return float(np.dot(point - np.array(self.arm.origin_pos), fwd))

    def _base_forward_to(self, point: np.ndarray, reach=REACH_M, tol: float = 0.01) -> None:
        """If `point` is outside the (min, desired, max) forward reach band, drive the mobile base
        along its forward axis so the point is `desired` ahead of it.

        The arm cannot reach low points closer than ~0.35 m to the base, and a far reach tips the
        wrist into overhangs, so grasp and carry call this first. Base mode (last action > 0) keeps
        the arm's goal fixed to the base."""
        lo, desired_m, hi = reach
        if lo <= self._forward_dist(point) <= hi:
            return
        fwd = np.array(self.arm.origin_ori)[:, 0]
        for _ in range(120):
            e = float(np.dot(point - np.array(self.arm.origin_pos), fwd)) - desired_m
            if abs(e) < tol:
                break
            self.rec.step(self._base_action(np.clip(e / 0.024, -0.5, 0.5)))
        for _ in range(12):  # come to rest before switching back to arm mode
            self.rec.step(self._base_action(0.0))

    def _base_action(self, forward: float) -> np.ndarray:
        a = np.zeros(self.env.action_dim)
        a[6] = self.gripper_cmd
        a[7] = forward
        a[-1] = 1.0
        return a

    def _hold(self, steps: int) -> None:
        for _ in range(steps):
            self.rec.step(self._action(np.zeros(3)))

    def _require(self, cond: bool, msg: str, steps: int) -> SkillResult:
        return SkillResult(True, "ok", steps) if cond else SkillResult(False, msg, steps)

    # -- primitives (the planner's vocabulary)
    def move_above(self, target: str, height_m: float = 0.15) -> SkillResult:
        p = self._pos_of(target) + np.array([0, 0, height_m])
        n = self._servo(p)
        return self._require(n >= 0, f"could not reach above {target}", max(n, 0))

    def grasp(self, target: str) -> SkillResult:
        """Rim grasp with up to three tries: far side of the handle, then the opposite side, then deeper."""
        if target not in self.scene.objects:
            return SkillResult(False, f"{target} is not a graspable object", 0)
        n0 = len(self.rec.actions)
        self._base_forward_to(self.scene.object_pos(target))
        for side, depth in ((1, 0.025), (-1, 0.025), (1, 0.035)):
            g = self._grasp_pose(target, side, depth)
            self.gripper_cmd = OPEN
            self._hold(4)
            self._servo(g + np.array([0, 0, 0.10]))
            self._servo(g, tol=0.005, max_steps=120)
            self.gripper_cmd = CLOSE
            self._hold(15)
            if self._grasping(target):
                self.held = target
                return SkillResult(True, "ok", len(self.rec.actions) - n0)
            self.gripper_cmd = OPEN
            self._hold(8)
            self._servo(self._eef() + np.array([0, 0, 0.10]), tol=0.02, max_steps=60)
        return SkillResult(False, f"gripper closed but {target} is not held (3 tries)", len(self.rec.actions) - n0)

    def _grasp_pose(self, target: str, side: int, depth: float) -> np.ndarray:
        c, r_in = self.scene.grasp_point(target, np.array(self.arm.origin_pos)[:2])
        g = np.array(c, dtype=float)
        if r_in is None:
            return g
        u = self._finger_axis()
        u = np.array([u[0], u[1], 0.0]) / (np.linalg.norm(u[:2]) + 1e-9)
        if np.dot(u, self.scene.object_pos(target) - c) > 0:  # away from the handle first
            u = -u
        g = c + side * (r_in + 0.006) * u
        g[2] = self.scene.object_pos(target)[2] + self.scene.object_half_height(target) - depth
        return g

    def lift(self, height_m: float = 0.15) -> SkillResult:
        n = self._servo(self._eef() + np.array([0, 0, height_m]))
        if self.held and not self._grasping(self.held):
            self.held = None
            return SkillResult(False, "object slipped out while lifting", max(n, 0))
        return self._require(n >= 0, "lift did not finish", max(n, 0))

    def carry_above(self, target: str, height_m: float = 0.05) -> SkillResult:
        """Move the held object so its bottom is height_m above the target surface."""
        if not self.held:
            return SkillResult(False, "nothing is held", 0)
        self._base_forward_to(self.scene.target_pos(target), PLACE_REACH_M)
        hh = self.scene.object_half_height(self.held)
        goal = self.scene.target_pos(target) + np.array([0, 0, height_m + hh])
        travel = goal.copy()
        travel[2] = max(goal[2], self.scene.object_pos(self.held)[2])  # never dip while travelling
        n = self._servo(travel, track=self.held, max_steps=300)
        n2 = self._servo(goal, track=self.held, max_steps=120, tol=0.006)
        steps = max(n, 0) + max(n2, 0)
        if not self._grasping(self.held):
            return SkillResult(False, "object slipped out while carrying", steps)
        xy = np.linalg.norm((self.scene.object_pos(self.held) - goal)[:2])
        return self._require(xy < XY_ALIGN_M, f"held object is {xy * 100:.1f} cm off {target}", steps)

    def lower_onto(self, target: str, clearance_m: float = 0.0) -> SkillResult:
        """Lower the held object until it rests on the target (stops when it stops descending)."""
        if not self.held:
            return SkillResult(False, "nothing is held", 0)
        surface = self.scene.target_pos(target)
        n0, last_z, stall = len(self.rec.actions), None, 0
        for _ in range(150):
            bottom = self.scene.object_bottom(self.held)
            if bottom[2] - surface[2] <= clearance_m + 0.002:
                break
            z = bottom[2]
            stall = stall + 1 if last_z is not None and last_z - z < 0.0005 else 0
            if stall >= 6:
                break
            last_z = z
            self.rec.step(self._action(np.array([0.0, 0.0, -0.006])))
        gap = self.scene.object_bottom(self.held)[2] - surface[2]
        return self._require(gap < 0.02, f"held object stopped {gap * 100:.1f} cm above {target}",
                             len(self.rec.actions) - n0)

    def release(self) -> SkillResult:
        self.gripper_cmd = OPEN
        self._hold(15)
        self.held = None
        return SkillResult(True, "ok", 15)

    def retreat(self, height_m: float = 0.05) -> SkillResult:
        """Rise a little, then back the base away so the gripper ends >25 cm from the object.

        Rising far is not an option under wall cabinets; backing up always clears the workspace."""
        self._servo(self._eef() + np.array([0, 0, min(height_m, 0.08)]), tol=0.015, max_steps=40)
        start = np.array(self.arm.origin_pos)
        fwd = np.array(self.arm.origin_ori)[:, 0]
        for _ in range(60):
            if np.dot(start - np.array(self.arm.origin_pos), fwd) > 0.28:
                break
            self.rec.step(self._base_action(-0.5))
        for _ in range(12):
            self.rec.step(self._base_action(0.0))
        return SkillResult(True, "ok", 0)

    def _pos_of(self, name: str) -> np.ndarray:
        if name in self.scene.objects:
            return self.scene.object_pos(name)
        if name in self.scene.targets:
            return self.scene.target_pos(name)
        raise KeyError(f"unknown entity {name}")

    def run(self, step) -> SkillResult:
        fn = getattr(self, step.primitive)
        kw = {k: v for k, v in step.model_dump(exclude={"primitive", "reason"}).items() if v is not None}
        n0 = len(self.rec.actions)
        try:
            res = fn(**kw)
        except KeyError as e:
            res = SkillResult(False, str(e), 0)
        res.steps = len(self.rec.actions) - n0  # count actions actually sent, whatever path ran
        self.log.append({"primitive": step.primitive, "args": kw, "reason": step.reason,
                         "ok": res.ok, "message": res.message, "steps": res.steps,
                         "t_end": len(self.rec.actions)})
        return res
