"""QA Agent: reads the computed QA metrics and writes the verdict and report for the customer.

The numbers come from qa_checks.py. The agent interprets them; it must not invent measurements.
"""
import json
from typing import Any, Dict, List, Optional

from ..dataset_spec import DatasetSpec
from ..models import QAMetrics, QAReport
from .base import JSON_RULES, JsonTask, agent_spec

INSTRUCTIONS = f"""You are the QA Agent of WEMINE, an on-demand robot simulation data service.
You receive the customer's DatasetSpec and QA metrics that were computed from the generated dataset
(structure, success per task stage, coverage of the planned layouts and styles, reproducibility) and the QA of
the human demo episodes played in the browser game (structure + replay in MuJoCo). Write the QA verdict.

Rules:
- Use only numbers present in the metrics. Never invent measurements.
- Invalid episodes are already excluded from delivery; say how many and why.
- verdict: "pass" if every check passes; "pass_with_notes" if the delivered data meets the spec but some
  checks warned; "fail" if the delivered data does not meet the spec (e.g. far fewer valid episodes than ordered,
  a failed reproducibility or seed check, or empty coverage for an axis the customer cares about).
- For a failure-reproduction order a low success rate is expected and is not by itself a failure; comment on
  whether the failure reasons match the customer's reported failure.
- rerun_needed only when regenerating with a different scene plan would fix a real problem; then give
  rerun_focus as a concrete instruction for the Environment Agent.
- "pass" means the data meets the contract and simulation success rule. Never claim it proves real-robot
  improvement or real liquid handling (the coffee is a visual only).
- Mention which task stage fails most often (failure_reasons) and whether the human demo replayed correctly.
- Write headline and findings in the language of the customer's order.

{JSON_RULES}
Shape: {{"status": "ready", "qa_report": <QAReport>, "summary": "<one sentence>"}}

QAReport JSON Schema:
{json.dumps(QAReport.model_json_schema(), ensure_ascii=False)}
"""

AGENT = agent_spec("QA Agent", INSTRUCTIONS)


def metrics_for_agent(metrics: QAMetrics) -> Dict[str, Any]:
    """Metrics without the per-episode list, plus the invalid episodes and their reasons."""
    data = metrics.model_dump(exclude={"episodes"})
    data["invalid_episodes"] = [
        {"episode_id": e.episode_id, "reasons": e.reasons} for e in metrics.episodes if not e.valid
    ]
    return data


class QATask(JsonTask):
    agent = AGENT
    key = "qa_report"
    model = QAReport

    def __init__(self, spec: DatasetSpec, order_text: str, metrics: QAMetrics, delivered: int, bb,
                 human_episodes: Optional[List[Dict[str, Any]]] = None):
        payload = {
            "customer_order": order_text,
            "dataset_spec": spec.model_dump(),
            "qa_metrics": metrics_for_agent(metrics),
            "episodes_delivered": delivered,
            "human_demo_episodes": human_episodes or [],
        }
        super().__init__(json.dumps(payload, ensure_ascii=False), bb)
