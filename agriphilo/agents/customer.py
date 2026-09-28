"""Customer Agent: turns a robot team's natural-language order into a validated DatasetSpec.

The agent either asks clarifying questions or returns a DatasetSpec.

    python3 -m agriphilo.agents.customer examples/failure_order.txt
    python3 -m agriphilo.agents.customer examples/generation_order.txt --auto
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from ..brainbase import Brainbase
from ..dataset_spec import TASK_STEPS, DatasetSpec
from .base import JSON_RULES, AgentThread, agent_spec

INSTRUCTIONS = f"""You are the Customer Agent of WEMINE, an on-demand robot simulation data service.
A robot team describes, in natural language (Korean or English), either
  - a FAILURE they saw (video/log description) that they want reproduced, evaluated and covered with more data -> mode "failure", or
  - NEW data they need for their robot/task -> mode "generation".
Your job is to turn the order into a DatasetSpec.

Pilot scope (do not promise more):
- One cafe-counter scene (environment.scene_template "robocasa_cafe_counter"), simulated in MuJoCo with RoboCasa.
  Counter layouts and visual styles can vary (environment.layout_variety / style_variety).
- Simulated robot is always a mobile Franka Panda (robot.sim_model "PandaOmron"), a proxy for the customer's hardware.
  Record their real robot in robot.customer_robot.
- Task is serving a coffee ("CafeServeCoffee"), exactly these task_steps in order: {json.dumps(TASK_STEPS)}.
- Episodes come from a human demo played in a browser game plus AI players; the customer does not need to choose.
- Data: cameras from {{robot0_agentview_center, robot0_agentview_left, robot0_agentview_right, robot0_frontview,
  robot0_eye_in_hand}} with resolution and fps, sensors, control rate, hdf5 or lerobot format.
- NOT covered: real liquid physics (the coffee in the cup is a visual; brewing and pouring are not simulated), other robots,
  real-robot performance guarantees, other tasks or scenes. Put such requests in out_of_scope; never silently drop them.

For failure mode, describe what the customer saw in reference_failure and bias the environment toward it.

Ask clarifying questions only when an answer would materially change the spec (no idea of scale, data format unclear).
Ask at most 3 questions per turn, in the customer's language. Otherwise choose sensible defaults and list each one in "assumptions".
Sensible defaults: mode generation, episode_count 200, cameras robot0_agentview_center and robot0_eye_in_hand at 128x128 20 fps,
sensors [rgb, joint_state, eef_pose, gripper_state, action, object_pose, task_stage], control_hz 20, format hdf5,
layout_variety few, style_variety few, success_rule time_limit_s 180.

{JSON_RULES}
Put your questions in the JSON reply too; the customer never sees tool arguments. Use one of two shapes:
{{"status": "need_info", "questions": ["..."]}}
{{"status": "ready", "dataset_spec": <DatasetSpec>, "summary": "<one-sentence summary for the customer, in their language>"}}

DatasetSpec JSON Schema:
{json.dumps(DatasetSpec.model_json_schema(), ensure_ascii=False)}
"""

AGENT = agent_spec("Customer Agent", INSTRUCTIONS)

AUTO_ANSWER = (
    "The customer is not available to answer. Do not ask more questions: choose sensible defaults, "
    "record each in assumptions, and return status ready."
)


def ask_stdin(questions) -> str:
    print("\n[Customer Agent questions]")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    print("Type your answer (end with an empty line):")
    lines = []
    for line in sys.stdin:
        if not line.strip():
            break
        lines.append(line.rstrip())
    return "\n".join(lines) or AUTO_ANSWER


class CustomerSession(AgentThread):
    """The Customer Agent conversation for one order.

    state: "running" -> "questions" (answer() resumes it) -> ... -> "ready" | "error".
    """

    agent = AGENT

    def __init__(self, order: str, auto: bool = False, bb: Optional[Brainbase] = None):
        self.auto = auto
        self.order = order
        self.transcript: List[Dict[str, Any]] = [{"role": "customer", "text": order}]
        self.questions: List[str] = []
        self.spec: Optional[DatasetSpec] = None
        self.summary = ""
        super().__init__(order, bb or Brainbase())

    def handle(self, reply: Dict[str, Any]) -> None:
        if reply.get("status") == "need_info":
            self.questions = reply.get("questions") or []
            if self.auto:
                return self.send(AUTO_ANSWER)
            self.transcript.append({"role": "agent", "questions": self.questions})
            self.state = "questions"
            return
        try:
            self.spec = DatasetSpec.model_validate(reply.get("dataset_spec"))
        except ValidationError as e:
            return self.validation_failed("DatasetSpec", e)
        self.summary = reply.get("summary", "")
        self.transcript.append({"role": "agent", "text": self.summary or "Spec ready."})
        self.state = "ready"

    def answer(self, text: str) -> None:
        if self.state != "questions":
            raise RuntimeError(f"cannot answer in state {self.state}")
        self.questions = []
        self.transcript.append({"role": "customer", "text": text.strip()})
        self.send(text.strip() or AUTO_ANSWER)


def run(order: str, answer: Optional[Callable] = ask_stdin, cleanup: bool = True) -> CustomerSession:
    """Blocking CLI driver. answer=None means auto mode."""
    session = CustomerSession(order, auto=answer is None)
    print(f"thread {session.thread_id} (agent {session.agent_id})", file=sys.stderr)
    try:
        while session.state in ("running", "questions"):
            if session.state == "questions":
                session.answer(answer(session.questions))
            time.sleep(3)
            session.poll()
        if session.state == "error":
            raise RuntimeError(f"{session.error} (thread {session.thread_id})")
        return session
    finally:
        if cleanup:
            session.close()
            print("deleted sandbox machine", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("order", help="order text, or a path to a file containing it")
    p.add_argument("--auto", action="store_true", help="never ask the human; agent fills defaults")
    p.add_argument("--keep-machine", action="store_true", help="do not delete the sandbox afterwards")
    p.add_argument("--out", default="out/specs", help="directory to write the spec JSON")
    args = p.parse_args()

    order = Path(args.order).read_text() if Path(args.order).is_file() else args.order
    result = run(order, answer=None if args.auto else ask_stdin, cleanup=not args.keep_machine)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result.spec.mode}_{result.thread_id[:8]}.json"
    path.write_text(result.spec.model_dump_json(indent=2))
    print(f"\n[Summary] {result.summary}")
    print(result.spec.model_dump_json(indent=2))
    print(f"\nsaved {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
