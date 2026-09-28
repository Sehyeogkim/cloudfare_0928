"""Shared machinery for Brainbase-hosted agents that reply in JSON.

Each agent runs as a Brainbase thread (Claude Code harness on a Cloudflare sandbox). Replies are
parsed and validated locally; invalid replies go back to the agent for correction.
"""
import json
import re
import time
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from ..brainbase import Brainbase

SILENT_GRACE_S = 15
POLL_S = 3

JSON_RULES = """Do not use tools, files or the shell. The pipeline only sees your text reply, never tool arguments:
if your runtime makes you call request_user_input to end the turn, first write the JSON reply as your message.
Reply with exactly one JSON object and nothing else."""


def agent_spec(title: str, instructions: str) -> Dict[str, Any]:
    return {
        "harness": "claude_code",
        "model": "claude-sonnet-5",
        "machine_kind": "cloudflare",
        "title": f"WEMINE · {title}",
        "instructions": instructions,
    }


def parse_reply(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    return json.loads(raw)


class AgentThread:
    """One Brainbase thread advanced one poll at a time.

    state: "running" -> ("questions" ->) "ready" | "error". Subclasses implement handle(reply).
    """

    agent: Dict[str, Any] = {}
    max_turns = 6

    def __init__(self, first_input: str, bb: Brainbase):
        self.bb = bb
        self.created_at = time.time()
        self.turn_started_at = self.created_at
        created = self.bb.create_thread(self.agent, first_input)
        self.thread_id: str = created["thread_id"]
        self.agent_id: str = created["agent_id"]
        self.state = "running"
        self.error = ""
        self.turns = 0
        self.replies_seen = 0
        self.idles_seen = 0
        self.silent_since: Optional[float] = None
        self.machine_id: Optional[str] = None

    def handle(self, reply: Dict[str, Any]) -> None:
        raise NotImplementedError

    def poll(self) -> str:
        """Check the thread once; process a new agent reply if one has settled."""
        if self.state != "running":
            return self.state
        thread = self.bb.get_thread(self.thread_id)
        self.machine_id = thread.get("machine_id") or self.machine_id
        if thread["status"] == "running":
            return self.state
        replies = self.bb.assistant_reply_count(self.thread_id)
        if replies > self.replies_seen:
            self.replies_seen = replies
            self.silent_since = None
            self.turns += 1
            try:
                reply = parse_reply(self.bb.last_assistant_text(self.thread_id))
            except ValueError:
                self.send("Your reply was not valid JSON. Reply with exactly one JSON object.")
                return self.state
            self.handle(reply)
        elif self._idle_count() > self.idles_seen:
            # The turn ended with no text yet. The reply can land a few seconds after the
            # idle event, so only nudge after a grace period.
            if self.silent_since is None:
                self.silent_since = time.time()
            elif time.time() - self.silent_since > SILENT_GRACE_S:
                self.silent_since = None
                self.turns += 1
                self.send("Your turn ended without a text reply. Reply now with exactly one JSON object as instructed.")
        return self.state

    def _idle_count(self) -> int:
        return sum(1 for e in self.bb.events(self.thread_id) if e.get("type") == "idle")

    def send(self, content: str) -> None:
        if self.turns >= self.max_turns:
            self.state, self.error = "error", f"no valid reply after {self.max_turns} turns"
            return
        self.idles_seen = self._idle_count()  # idle events up to now belong to earlier turns
        self.bb.post_message(self.thread_id, content)
        self.turn_started_at = time.time()
        self.state = "running"

    def validation_failed(self, what: str, e: Exception) -> None:
        self.send(f"{what} failed validation, fix and resend the whole reply:\n{e}")

    def close(self) -> None:
        """Delete the sandbox machine so it stops running."""
        try:
            machine_id = self.machine_id or self.bb.get_thread(self.thread_id).get("machine_id")
            if machine_id:
                self.bb.delete_machine(machine_id)
                self.machine_id = None
        except Exception:
            pass


class JsonTask(AgentThread):
    """Single-shot agent: input in, one validated object out.

    Expected reply: {"status": "ready", "<key>": {...}, "summary": "..."}.
    An optional check(obj) raising ValueError adds pipeline-specific validation.
    """

    key = "result"
    model: Type[BaseModel] = BaseModel

    def __init__(self, first_input: str, bb: Brainbase):
        super().__init__(first_input, bb)
        self.result: Optional[BaseModel] = None
        self.summary = ""

    def check(self, obj: BaseModel) -> None:
        pass

    def handle(self, reply: Dict[str, Any]) -> None:
        try:
            obj = self.model.model_validate(reply.get(self.key))
            self.check(obj)
        except (ValidationError, ValueError) as e:
            return self.validation_failed(f'"{self.key}"', e)
        self.result, self.summary = obj, reply.get("summary", "")
        self.state = "ready"

    def run(self, timeout_s: float = 600) -> BaseModel:
        """Block until the agent returns a valid object; always frees the sandbox."""
        deadline = time.time() + timeout_s
        try:
            while self.poll() == "running":
                if time.time() > deadline:
                    raise TimeoutError(f"{self.agent['title']} timed out after {timeout_s:.0f}s")
                time.sleep(POLL_S)
            if self.state != "ready":
                raise RuntimeError(f"{self.agent['title']}: {self.error}")
            return self.result
        finally:
            self.close()
