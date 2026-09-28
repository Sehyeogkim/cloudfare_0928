"""Minimal Brainbase Universal Harness API client (https://docs.brainbaselabs.com/api)."""
import os
import time
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "https://api.brainbaselabs.com/v2"
RETRIES = 4
RETRY_STATUS = {429, 500, 502, 503, 504}


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class Brainbase:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        key = api_key or os.environ.get("BRAINBASE_API_KEY")
        if not key:
            raise RuntimeError("BRAINBASE_API_KEY is not set")
        self.http = httpx.Client(
            base_url=BASE_URL, headers={"Authorization": f"Bearer {key}"}, timeout=60
        )

    def _req(self, method: str, path: str, **kw) -> Any:
        """Send a request, retrying transient failures.

        Reads and deletes are retried on any network error, 429 or 5xx. POSTs are retried only when the
        connection never opened, so a message is never sent twice.
        """
        idempotent = method in ("GET", "DELETE")
        for attempt in range(RETRIES + 1):
            last = attempt == RETRIES
            try:
                r = self.http.request(method, path, **kw)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if last:
                    raise
            except httpx.TransportError:
                if last or not idempotent:
                    raise
            else:
                if r.status_code in RETRY_STATUS and idempotent and not last:
                    pass
                else:
                    r.raise_for_status()
                    return r.json()
            time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("unreachable")

    def create_thread(self, agent: Dict[str, Any], input: str) -> Dict[str, Any]:
        return self._req("POST", "/threads", json={"agent": agent, "input": input})

    def get_thread(self, thread_id: str) -> Dict[str, Any]:
        return self._req("GET", f"/threads/{thread_id}")

    def post_message(self, thread_id: str, content: str) -> Dict[str, Any]:
        body = {"messages": [{"role": "user", "content": content}], "run": True}
        return self._req("POST", f"/threads/{thread_id}/messages", json=body)

    def messages(self, thread_id: str) -> List[Dict[str, Any]]:
        return self._req("GET", f"/threads/{thread_id}/messages")["items"]

    def assistant_reply_count(self, thread_id: str) -> int:
        return sum(1 for m in self.messages(thread_id) if m["role"] == "assistant" and m.get("content"))

    def last_assistant_text(self, thread_id: str) -> str:
        for m in reversed(self.messages(thread_id)):
            if m["role"] == "assistant" and m.get("content"):
                return m["content"]
        raise RuntimeError(f"thread {thread_id} has no assistant reply")

    def events(self, thread_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self._req("GET", f"/threads/{thread_id}/events", params={"limit": limit})["items"]

    def delete_machine(self, machine_id: str) -> Dict[str, Any]:
        return self._req("DELETE", f"/machines/{machine_id}")
