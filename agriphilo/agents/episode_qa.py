"""Episode QA Agent: a rough, natural-language quality check of each episode a player submits.

One Brainbase thread per marketplace listing reviews episodes one after another, so later episodes
skip the sandbox start-up. Code measures the episode (length, task stages, idle time, action range);
the agent reads those facts and decides pass/fail with a quality score and a one-sentence reason.
"""
import json
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .base import JSON_RULES, POLL_S, AgentThread, agent_spec

INSTRUCTIONS = f"""You are the Episode QA Agent of WEMINE, a robot data marketplace. Human players control a
simulated robot in a MuJoCo browser game; each episode they submit may be sold to a requester as training data.
For every episode you receive measured facts (task, task stages completed, whether the simulator marked it a
success, length, idle time, action range, gripper use). Judge it roughly, like a data reviewer skimming it:

- pass: the task was completed (all stages) and nothing suggests garbage data (e.g. mostly idle, absurdly short
  or long, no gripper use for a pick task, non-finite values).
- fail: the task was not completed, or the data looks unusable. Say why in plain words.
- quality 1-5: 5 = clean, efficient demonstration; 3 = completed but slow or wobbly; 1 = unusable.
- reason: one short sentence a requester can read.
Use only the facts given. Do not invent measurements.

{JSON_RULES}
Shape: {{"status": "ready", "episode_review": {{"verdict": "pass" | "fail", "quality": 1-5, "reason": "..."}}}}
"""

AGENT = agent_spec("Episode QA Agent", INSTRUCTIONS)


class EpisodeReview(BaseModel):
    verdict: str = Field(pattern="^(pass|fail)$")
    quality: int = Field(ge=1, le=5)
    reason: str


class EpisodeQASession(AgentThread):
    """Reviews episodes one at a time on a single thread. review() blocks until the verdict arrives."""

    agent = AGENT
    max_turns = 400  # one turn per episode plus corrections

    def __init__(self, bb):
        self.bb = bb
        self.started = False
        self.result: Optional[EpisodeReview] = None

    def handle(self, reply: Dict[str, Any]) -> None:
        try:
            self.result = EpisodeReview.model_validate(reply.get("episode_review"))
        except Exception as e:
            return self.validation_failed('"episode_review"', e)
        self.state = "ready"

    def review(self, facts: Dict[str, Any], timeout_s: float = 300) -> EpisodeReview:
        text = "Review this episode:\n" + json.dumps(facts, ensure_ascii=False)
        self.result = None
        if not self.started:
            super().__init__(text, self.bb)
            self.started = True
        else:
            self.send(text)
        deadline = time.time() + timeout_s
        while self.poll() == "running":
            if time.time() > deadline:
                raise TimeoutError(f"Episode QA Agent timed out after {timeout_s:.0f}s")
            time.sleep(POLL_S)
        if self.state != "ready" or self.result is None:
            raise RuntimeError(f"Episode QA Agent: {self.error or 'no verdict'}")
        return self.result
