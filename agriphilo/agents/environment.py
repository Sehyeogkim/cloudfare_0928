"""Environment Agent: chooses how to build the cafe scene for a DatasetSpec.

The agent picks from a fixed menu (layouts, styles, cameras, object variety, timing). Turning the
plan into per-episode parameters is deterministic code, so the same plan always yields the same job.
"""
import json
from typing import List, Optional

import numpy as np

from ..dataset_spec import DatasetSpec
from ..game_spec import MAX_LAYOUT_ID, MAX_STYLE_ID, ROBOCASA_CAMERAS, ROBOCASA_ROBOTS, ROBOCASA_TASKS, GameSpec
from ..models import EnvironmentPlan, EpisodeParams, OrchestratorPlan, SimJob
from .base import JSON_RULES, JsonTask, agent_spec

INSTRUCTIONS = f"""You are the Environment Agent of WEMINE, an on-demand robot simulation data service.
You receive a validated DatasetSpec, the Orchestrator's production plan (and sometimes QA feedback from a previous
run) and decide how the cafe-counter scene is set up in MuJoCo (RoboCasa). You choose from a fixed menu; you do not
write simulator code. The scene always holds a cup of coffee on the counter in front of the robot, a tray next to it, and a coffee machine behind.

Decide:
- layout_ids (1-{MAX_LAYOUT_ID}) and style_ids (1-{MAX_STYLE_ID}): counter layouts and visual styles to sample from.
  single -> 1 id, few -> 3-5 ids, many -> 10-20 ids (follow DatasetSpec.environment.layout_variety / style_variety).
  Layouts with a long straight counter work best: prefer 9, 5, 7 first. Styles 3 (modern, white counter) and
  8 (wood) look most like a cafe.
- cameras: copy DatasetSpec.data.cameras unless there is a reason to add one (say it in the rationale).
- control_hz: DatasetSpec.data.control_hz. max_steps must be at least success_rule.time_limit_s * control_hz.
- object_instance_randomization: true when the customer wants visual variety of the objects.
- placement_jitter_m (0-0.05): how much object positions vary between episodes.
- base_seed: any integer in [0, 2^31).
- focus: short plain-language notes on what the scene should stress.
- rationale: one or two sentences explaining the choices, for the customer.
If QA feedback is given, change the plan to address it and say how in the rationale.

{JSON_RULES}
Shape: {{"status": "ready", "environment_plan": <EnvironmentPlan>, "summary": "<one sentence>"}}

EnvironmentPlan JSON Schema:
{json.dumps(EnvironmentPlan.model_json_schema(), ensure_ascii=False)}
"""

AGENT = agent_spec("Environment Agent", INSTRUCTIONS)


class EnvironmentTask(JsonTask):
    agent = AGENT
    key = "environment_plan"
    model = EnvironmentPlan

    def __init__(self, spec: DatasetSpec, bb, qa_feedback: Optional[str] = None,
                 orchestrator_plan: Optional[OrchestratorPlan] = None):
        self.spec = spec
        payload = {"dataset_spec": spec.model_dump()}
        if orchestrator_plan is not None:
            payload["orchestrator_plan"] = orchestrator_plan.model_dump()
        if qa_feedback:
            payload["qa_feedback"] = qa_feedback
        super().__init__(json.dumps(payload, ensure_ascii=False), bb)

    def check(self, plan: EnvironmentPlan) -> None:
        needed = self.spec.success_rule.time_limit_s * plan.control_hz
        if plan.max_steps < needed:
            raise ValueError(f"max_steps {plan.max_steps} < time_limit_s * control_hz = {needed:.0f}")


GAME_INSTRUCTIONS = f"""You are the Game Environment Agent of WEMINE. WEMINE turns a data order into a playable
robot game: human or AI players control a simulated robot in a RoboCasa kitchen, and the episodes that pass QA
become the customer's dataset. You receive the customer's order text and, when available, a validated DatasetSpec.
You pick the game setup from fixed menus; you do not write simulator code.

Decide:
- task: the RoboCasa atomic task closest to what the customer wants. One of: {", ".join(ROBOCASA_TASKS)}.
  If nothing fits well, pick the closest one and say what differs in the rationale.
- layout_ids (1-{MAX_LAYOUT_ID}) and style_ids (1-{MAX_STYLE_ID}): kitchen layouts and styles to sample from. Use more of
  them when the customer wants visual or scene variety; a few when they want a narrow, repeatable setup.
- robot: one of {", ".join(ROBOCASA_ROBOTS)}. PandaOmron is a single-arm mobile Panda; GR1FixedLowerBody is a
  humanoid upper body. The customer's real robot is usually not available; choose the closest proxy and say so.
- cameras: 1-3 of {", ".join(ROBOCASA_CAMERAS)} with width/height (64-1024) and fps (5-30).
  Include robot0_eye_in_hand for manipulation data.
- control_hz (10-30), success_rule.time_limit_s, and max_steps >= time_limit_s * control_hz.
- episodes_requested: the number of QA-passing episodes the customer needs (from the DatasetSpec when given).
- credit_per_pass: credits paid to a player per QA-passing episode; harder or longer tasks earn more (0.5-5 typical).
- base_seed: any integer in [0, 2^31).
- rationale: one or two sentences explaining the choices, for the customer.

{JSON_RULES}
Shape: {{"status": "ready", "game_spec": <GameSpec>, "summary": "<one sentence>"}}

GameSpec JSON Schema:
{json.dumps(GameSpec.model_json_schema(), ensure_ascii=False)}
"""

GAME_AGENT = agent_spec("Game Environment Agent", GAME_INSTRUCTIONS)


class GameEnvironmentTask(JsonTask):
    """Order (+ optional DatasetSpec) -> GameSpec for the RoboCasa Game Server. Runs alongside EnvironmentTask."""

    agent = GAME_AGENT
    key = "game_spec"
    model = GameSpec

    def __init__(self, order_text: str, bb, spec: Optional[DatasetSpec] = None):
        self.spec = spec
        payload = {"order_text": order_text}
        if spec is not None:
            payload["dataset_spec"] = spec.model_dump()
        super().__init__(json.dumps(payload, ensure_ascii=False), bb)

    def check(self, game: GameSpec) -> None:
        if self.spec is not None and game.episodes_requested != self.spec.episode_count:
            raise ValueError(
                f"episodes_requested {game.episodes_requested} != dataset_spec.episode_count {self.spec.episode_count}"
            )


def build_sim_job(
    job_id: str, order_id: str, spec: DatasetSpec, plan: EnvironmentPlan,
    count: Optional[int] = None, seed_offset: int = 0, id_offset: int = 0,
) -> SimJob:
    """Expand the plan into per-episode parameters. Deterministic for a given plan and offsets.

    Layouts and styles are cycled in a shuffled order so every one of them gets episodes.
    """
    n = count if count is not None else spec.episode_count
    rng = np.random.default_rng(plan.base_seed + seed_offset)
    layouts = rng.permutation(np.resize(plan.layout_ids, n))
    styles = rng.permutation(np.resize(plan.style_ids, n))
    jitter = rng.uniform(0, plan.placement_jitter_m, n)
    seeds = rng.choice(2**31 - 1, size=n, replace=False)
    episodes: List[EpisodeParams] = [
        EpisodeParams(
            episode_id=f"ep_{id_offset + i:04d}", seed=int(seeds[i]),
            layout_id=int(layouts[i]), style_id=int(styles[i]),
            placement_jitter_m=round(float(jitter[i]), 4), player="ai",
        )
        for i in range(n)
    ]
    return SimJob(
        job_id=job_id, order_id=order_id, spec=spec, plan=plan, episodes=episodes,
        sensors=spec.sensors, success_rule=spec.success_rule,
    )
