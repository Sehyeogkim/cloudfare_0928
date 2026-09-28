"""Simulator workers: turn a SimJob into an HDF5 dataset. See docs/sim_worker.md for the contract.

- MockSimWorker (sim/mock.py): synthetic episodes, no GPU. Default, so the pipeline runs end to end.
- HttpSimWorker: the Isaac Sim worker on RunPod, reached over HTTP. Set AGRIPHILO_SIM_WORKER_URL.
"""
import os
import time
from pathlib import Path
from typing import Callable, Optional

import httpx

from ..models import SimJob, SimResult

Progress = Callable[[int, int], None]


class SimWorker:
    name = "base"

    def run(self, job: SimJob, out_path: Path, progress: Optional[Progress] = None) -> SimResult:
        raise NotImplementedError


class HttpSimWorker(SimWorker):
    """Client for a remote simulator implementing the HTTP contract in docs/sim_worker.md."""

    def __init__(self, base_url: str, token: Optional[str] = None, poll_s: float = 5, timeout_s: float = 6 * 3600):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(base_url=self.base_url, headers=headers, timeout=120)
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.name = f"http:{self.base_url}"

    def run(self, job: SimJob, out_path: Path, progress: Optional[Progress] = None) -> SimResult:
        r = self.http.post("/jobs", json=job.model_dump())
        r.raise_for_status()
        remote_id = r.json()["job_id"]
        deadline = time.time() + self.timeout_s
        while True:
            s = self.http.get(f"/jobs/{remote_id}")
            s.raise_for_status()
            body = s.json()
            p = body.get("progress") or {}
            if progress and "done" in p:
                progress(int(p["done"]), int(p.get("total", len(job.episodes))))
            if body["status"] == "done":
                break
            if body["status"] == "failed":
                raise RuntimeError(f"simulator job failed: {body.get('error')}")
            if time.time() > deadline:
                raise TimeoutError(f"simulator job {remote_id} still {body['status']} after {self.timeout_s}s")
            time.sleep(self.poll_s)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.http.stream("GET", f"/jobs/{remote_id}/dataset") as d:
            d.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in d.iter_bytes():
                    f.write(chunk)
        result = dict(body["result"], job_id=job.job_id, dataset_path=str(out_path))
        result.setdefault("worker", self.name)
        return SimResult.model_validate(result)


def default_worker() -> SimWorker:
    url = os.environ.get("AGRIPHILO_SIM_WORKER_URL")
    if url:
        return HttpSimWorker(url, os.environ.get("AGRIPHILO_SIM_WORKER_TOKEN"))
    from .mock import MockSimWorker

    return MockSimWorker()
