"""Orchestrator Agent: turns an approved DatasetSpec into the production plan for MuJoCo.

The plan says which agent does what, how the cafe scene is assembled, how each task stage is checked,
where the episodes come from (human demo + AI players) and what QA will look at. Later agents get it as context.
"""
import json

from ..dataset_spec import DatasetSpec
from ..models import OrchestratorPlan
from .base import JSON_RULES, JsonTask, agent_spec

INSTRUCTIONS = f"""You are the Orchestrator Agent of WEMINE, an on-demand robot simulation data service.
You receive an approved DatasetSpec and write the production plan: how WEMINE builds and runs it in MuJoCo.

Fixed facts you plan around (do not invent other tools):
- Simulator: MuJoCo 3 with RoboCasa/robosuite. Scene: a RoboCasa kitchen counter styled as a cafe, holding
  a cup of coffee (mug) on the open counter, a tray next to it and a coffee machine as backdrop. Robot: PandaOmron (mobile Franka Panda), OSC end-effector control.
- Task environment CafeServeCoffee checks the task stages in order with simulator state (the mug is grasped, then
  the mug rests inside the tray and is released).
- Stages and their owners, in order:
  environment (Environment Agent: layouts, styles, cameras, object variety; renders a preview),
  play (Human player: plays the task once or a few times in the browser game; the customer watches live),
  simulation (AI players: produce the bulk of the episodes from the human demos),
  collection (Data Collection: packages episodes into the requested schema),
  qa (QA Agent: structure, task success per stage, diversity, replay reproducibility),
  quote (Payment Agent).
- data_sourcing.ai_episodes + data_sourcing.human_demos must be at least the DatasetSpec episode_count.
Write concretely, in MuJoCo/RoboCasa terms, short lines. Write in the language of the DatasetSpec's free text.

{JSON_RULES}
Shape: {{"status": "ready", "orchestrator_plan": <OrchestratorPlan>, "summary": "<one sentence for the customer>"}}

OrchestratorPlan JSON Schema:
{json.dumps(OrchestratorPlan.model_json_schema(), ensure_ascii=False)}
"""

AGENT = agent_spec("Orchestrator Agent", INSTRUCTIONS)


class OrchestratorTask(JsonTask):
    agent = AGENT
    key = "orchestrator_plan"
    model = OrchestratorPlan

    def __init__(self, spec: DatasetSpec, order_text: str, bb):
        self.spec = spec
        payload = {"order_text": order_text, "dataset_spec": spec.model_dump()}
        super().__init__(json.dumps(payload, ensure_ascii=False), bb)

    def check(self, plan: OrchestratorPlan) -> None:
        got = plan.data_sourcing.ai_episodes + plan.data_sourcing.human_demos
        if got < self.spec.episode_count:
            raise ValueError(f"data_sourcing covers {got} episodes < episode_count {self.spec.episode_count}")
        stages = [s.stage for s in plan.steps]
        for needed in ("environment", "play", "simulation", "qa"):
            if needed not in stages:
                raise ValueError(f"steps must include the {needed} stage")
