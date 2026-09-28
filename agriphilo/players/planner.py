"""Plans: a validated list of skill primitives, from the LLM (Brainbase) or a scripted fallback."""
import json
import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Primitive = Literal["move_above", "grasp", "lift", "carry_above", "lower_onto", "release", "retreat"]
NEEDS_TARGET = {"move_above": "any", "grasp": "object", "carry_above": "target", "lower_onto": "target"}
ARGS = {  # primitive -> allowed numeric args (m)
    "move_above": {"height_m"}, "lift": {"height_m"}, "carry_above": {"height_m"},
    "lower_onto": {"clearance_m"}, "retreat": {"height_m"}, "grasp": set(), "release": set(),
}


class PlanStep(BaseModel):
    primitive: Primitive
    target: Optional[str] = None
    height_m: Optional[float] = Field(default=None, ge=0.0, le=0.4)
    clearance_m: Optional[float] = Field(default=None, ge=0.0, le=0.1)
    reason: str = Field(min_length=1, max_length=300)


class Plan(BaseModel):
    steps: List[PlanStep] = Field(min_length=1, max_length=20)
    summary: str = ""


def check_plan(plan: Plan, objects: List[str], targets: List[str]) -> None:
    """Raise ValueError if a step names an unknown entity or an argument the primitive ignores."""
    for i, s in enumerate(plan.steps):
        need = NEEDS_TARGET.get(s.primitive)
        allowed = {"any": objects + targets, "object": objects, "target": targets}.get(need, [])
        if need and s.target not in allowed:
            raise ValueError(f"step {i} {s.primitive}: target must be one of {allowed}, got {s.target!r}")
        if not need and s.target is not None:
            raise ValueError(f"step {i} {s.primitive} takes no target")
        extra = {k for k in ("height_m", "clearance_m") if getattr(s, k) is not None} - ARGS[s.primitive]
        if extra:
            raise ValueError(f"step {i} {s.primitive} does not take {sorted(extra)}")


PRIMITIVE_DOC = """Primitives (the only actions you can use):
- move_above(target, height_m=0.15): move the open gripper above an object or target.
- grasp(target): approach a graspable object from above and close the gripper on it. Fails if not held.
- lift(height_m=0.15): raise the gripper (and anything held) straight up.
- carry_above(target, height_m=0.05): move the held object so its bottom hangs height_m above a place
  target's surface. Respect any height limit given in the target description.
- lower_onto(target): lower the held object until it rests on the target.
- release(): open the gripper.
- retreat(height_m=0.05): rise slightly and back the robot away so the gripper is clear of the object
  (tasks usually only count as done when the gripper is away)."""


def scripted_plan(scene_desc: Dict) -> Plan:
    """Deterministic fallback: pick the first object, place it on the first target."""
    obj = next(iter(scene_desc["objects"]))
    tgt = next(iter(scene_desc["targets"]))
    S = PlanStep
    return Plan(summary=f"Pick up the {obj} and place it at {tgt}.", steps=[
        S(primitive="move_above", target=obj, height_m=0.15, reason=f"Get over the {obj} before descending."),
        S(primitive="grasp", target=obj, reason=f"Close the gripper on the {obj} rim."),
        S(primitive="lift", height_m=0.08, reason="Lift clear of the counter and the raised tray lip."),
        S(primitive="carry_above", target=tgt, height_m=0.04, reason=f"Slide the {obj} in low, right over {tgt}."),
        S(primitive="lower_onto", target=tgt, reason=f"Set the {obj} down on {tgt}."),
        S(primitive="release", reason=f"Let go of the {obj}."),
        S(primitive="retreat", height_m=0.05, reason="Move away so the task counts as complete."),
    ])


# ---- Brainbase (LLM) planner -----------------------------------------------------------------

INSTRUCTIONS = f"""You control a simulated mobile manipulator (Franka Panda arm with a parallel gripper)
in a RoboCasa kitchen. You never output joint values. You write a short plan of skill primitives;
scripted controllers execute each one. Positions are world coordinates in metres (z is up).

{PRIMITIVE_DOC}

Reply format: {{"status": "ready", "plan": {{"summary": "...", "steps": [{{"primitive": "...",
"target": "...", "height_m": 0.1, "reason": "one short sentence shown to the audience"}}]}},
"summary": "..."}}. Omit arguments you do not need. Use only the entity names given in the scene."""


def brainbase_plan(scene_desc: Dict, history: Optional[List[Dict]] = None, timeout_s: float = 600):
    """One Brainbase agent turn -> (Plan, usage dict). history holds earlier attempts + failures."""
    from ..agents.base import JSON_RULES, JsonTask, agent_spec
    from ..brainbase import Brainbase

    objects, targets = list(scene_desc["objects"]), list(scene_desc["targets"])

    class PlanTask(JsonTask):
        agent = agent_spec("Robot Player", INSTRUCTIONS + "\n\n" + JSON_RULES)
        key = "plan"
        model = Plan

        def check(self, obj):
            check_plan(obj, objects, targets)

    msg = "Scene:\n" + json.dumps(scene_desc, indent=1)
    if history:
        msg += ("\n\nEarlier attempts in this episode (the robot state has changed since; plan only the "
                "remaining steps from the current state):\n" + json.dumps(history, indent=1))
    t0 = time.time()
    task = PlanTask(msg, Brainbase())
    plan = task.run(timeout_s=timeout_s)
    usage = {"thread_id": task.thread_id, "turns": task.turns, "seconds": round(time.time() - t0, 1),
             "model": task.agent["model"]}
    return plan, usage
