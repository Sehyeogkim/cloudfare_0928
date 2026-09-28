"""Order pipeline: Customer -> (approve, hold credits) -> Orchestrator -> Environment -> Play (human demo, browser game)
-> AI players (simulation) -> Collection -> QA -> Quote -> (pay) -> Delivery.

Brainbase agents make the judgment calls; code does everything that must be exact. Each order
lives in out/orders/<id>/ and its state is saved to order.json after every change.
"""
import json
import os
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents.customer import CustomerSession
from .agents.environment import EnvironmentTask, build_sim_job
from .agents.orchestrator import OrchestratorTask
from .agents.payment import PRICING, PaymentTask
from .agents.qa import QATask
from .collection import build_manifest, export, merge_into, preview
from .dataset_spec import TASK_STEPS, DatasetSpec
from .models import EnvironmentPlan, HumanEpisode, Manifest, OrchestratorPlan, Payment, QAMetrics, QAReport, Quote, SimJob
from .qa_checks import run_checks
from .sim.worker import SimWorker

ROOT = Path("out/orders")
REPO = Path(__file__).resolve().parent.parent
SIM_PYTHON = REPO / "sim" / ".venv" / "bin" / "python"
EPISODES = REPO / "runs" / "episodes"
CUSTOMER = "demo-customer"  # same wallet as the game orders (games.py)
CREDITS_PER_EPISODE = PRICING["game"]["episode_price_credits"]
PLAYER_SHARE = PRICING["game"]["player_share_credits"]
POLL_S = 3
MAX_TOPUPS = 1
MAX_RERUNS = 1

STAGES = [
    ("spec", "Spec", "Customer Agent"),
    ("plan", "Plan", "Orchestrator Agent"),
    ("environment", "Scene", "Environment Agent"),
    ("play", "Play", "Human player"),
    ("simulation", "AI players", "AI players"),
    ("collection", "Data", "Data Collection"),
    ("qa", "QA", "QA Agent"),
    ("quote", "Quote", "Payment Agent"),
    ("delivery", "Delivery", "Checkout"),
]
DELIVERY_FILES = {
    "dataset.hdf5": "application/x-hdf5",
    "manifest.json": "application/json",
    "qa_report.json": "application/json",
    "qa_report.md": "text/markdown",
    "spec.json": "application/json",
}


def _stage(status: str = "pending") -> Dict[str, Any]:
    return {"status": status, "started_at": None, "finished_at": None, "detail": "", "error": "", "thread_id": None}


class Order:
    def __init__(self, text: str, auto: bool, bb, worker: SimWorker, order_id: Optional[str] = None, ledger=None):
        self.id = order_id or str(uuid.uuid4())
        self.text = text
        self.auto = auto
        self.bb = bb
        self.worker = worker
        self.created_at = time.time()
        self.dir = ROOT / self.id
        self.lock = threading.RLock()
        self.stages: Dict[str, Dict[str, Any]] = {k: _stage() for k, _, _ in STAGES}
        self.ledger = ledger
        self.credits_held = 0.0
        self.awaiting: Optional[str] = None  # "answers" | "spec_approval" | "play" | "payment"
        self.events: List[Dict[str, Any]] = []
        self.transcript: List[Dict[str, Any]] = [{"role": "customer", "text": text}]
        self.questions: List[str] = []
        self.spec: Optional[DatasetSpec] = None
        self.spec_summary = ""
        self.orchestrator_plan: Optional[OrchestratorPlan] = None
        self.orchestrator_summary = ""
        self.plan: Optional[EnvironmentPlan] = None
        self.has_scene_preview = False
        self.human_episodes: List[HumanEpisode] = []
        self.has_human_preview = False
        self._play_done = threading.Event()
        self.plan_summary = ""
        self.job: Optional[SimJob] = None
        self.progress: Dict[str, int] = {}
        self.compute_seconds = 0.0
        self.manifest: Optional[Manifest] = None
        self.qa_metrics: Optional[QAMetrics] = None
        self.qa_report: Optional[QAReport] = None
        self.quote: Optional[Quote] = None
        self.quote_summary = ""
        self.payment: Optional[Payment] = None
        self.has_preview = False
        self.session: Optional[CustomerSession] = None

    # ---- state helpers ------------------------------------------------------------------

    def log(self, stage: str, text: str) -> None:
        with self.lock:
            self.events.append({"t": time.time(), "stage": stage, "text": text})
        self.save()

    def start(self, stage: str, detail: str = "") -> None:
        with self.lock:
            s = self.stages[stage]
            s.update(status="running", started_at=time.time(), finished_at=None, detail=detail, error="")
        self.save()

    def finish(self, stage: str, detail: str = "", status: str = "done") -> None:
        with self.lock:
            self.stages[stage].update(status=status, finished_at=time.time(), detail=detail)
        self.save()

    def fail(self, stage: str, error: str) -> None:
        with self.lock:
            self.stages[stage].update(status="failed", finished_at=time.time(), error=error)
            self.awaiting = None
        self.log(stage, f"Failed: {error}")

    @property
    def state(self) -> str:
        if any(s["status"] == "failed" for s in self.stages.values()):
            return "failed"
        if self.stages["delivery"]["status"] == "done":
            return "delivered"
        if self.awaiting:
            return {"answers": "needs_reply", "spec_approval": "needs_approval", "play": "needs_play",
                    "payment": "awaiting_payment"}[self.awaiting]
        return "running"

    # ---- customer stage -----------------------------------------------------------------

    def begin(self) -> None:
        self.start("spec", "Customer Agent is reading the order")
        self.log("spec", "Order received")
        self.session = CustomerSession(self.text, auto=self.auto, bb=self.bb)
        self.stages["spec"]["thread_id"] = self.session.thread_id
        self._spawn(self._run_customer)

    def _run_customer(self) -> None:
        s = self.session
        while s.poll() == "running":
            time.sleep(POLL_S)
        with self.lock:
            self.transcript = list(s.transcript)
        if s.state == "questions":
            with self.lock:
                self.questions = list(s.questions)
                self.awaiting = "answers"
                self.stages["spec"].update(status="waiting", detail="Waiting for your answers")
            self.log("spec", f"Customer Agent asked {len(s.questions)} question(s)")
            return
        s.close()
        if s.state != "ready":
            return self.fail("spec", s.error or "Customer Agent failed")
        with self.lock:
            self.spec, self.spec_summary = s.spec, s.summary
            self.awaiting = "spec_approval"
        self.finish("spec", f"{s.spec.mode} · {s.spec.episode_count} episodes")
        self.log("spec", "Spec ready for approval")

    def answer(self, text: str) -> None:
        with self.lock:
            if self.awaiting != "answers":
                raise RuntimeError("this order is not waiting for answers")
            self.awaiting = None
            self.questions = []
            self.session.answer(text)
            self.transcript = list(self.session.transcript)
            self.stages["spec"].update(status="running", detail="Customer Agent is updating the spec")
        self.log("spec", "Answers sent")
        self._spawn(self._run_customer)

    # ---- production -----------------------------------------------------------------------

    @property
    def credit_estimate(self) -> float:
        return round(self.spec.episode_count * CREDITS_PER_EPISODE, 2) if self.spec else 0.0

    def approve(self) -> None:
        with self.lock:
            if self.awaiting != "spec_approval":
                raise RuntimeError("this order is not waiting for spec approval")
            if self.ledger is not None:
                self.ledger.hold(CUSTOMER, self.id, self.credit_estimate)  # raises InsufficientCredits
                self.credits_held = self.credit_estimate
            self.awaiting = None
        self.log("spec", f"Spec approved; {self.credits_held:,.0f} credits held" if self.credits_held else "Spec approved")
        self._spawn(self._run_production)

    def _run_production(self) -> None:
        if not self.orchestrator_plan and not self._run_orchestrator():
            return
        feedback = None
        for rerun in range(MAX_RERUNS + 1):
            if not self._run_environment(feedback):
                return
            if rerun == 0 and not self._run_play():
                return
            raw = self.dir / "work" / f"raw_{rerun}.hdf5"
            if not self._run_simulation(raw):
                return
            if not self._run_collection_and_checks(raw):
                return
            if not self._run_qa_agent():
                return
            if self.qa_report.rerun_needed and rerun < MAX_RERUNS:
                feedback = self.qa_report.rerun_focus or "; ".join(self.qa_report.recommended_fixes)
                self.log("qa", f"QA Agent requested a rerun: {feedback}")
                for k in ("environment", "simulation", "collection", "qa"):
                    self.stages[k] = _stage()
                continue
            break
        self._deliver_files(raw)
        self._run_quote()

    def _agent_task(self, stage: str, task_factory):
        """Run a Brainbase JsonTask, recording its thread id on the stage."""
        task = task_factory()
        with self.lock:
            self.stages[stage]["thread_id"] = task.thread_id
        self.save()
        return task.run()

    def _run_orchestrator(self) -> bool:
        self.start("plan", "Orchestrator Agent is writing the MuJoCo production plan")
        try:
            holder = {}

            def make():
                holder["t"] = OrchestratorTask(self.spec, self.text, self.bb)
                return holder["t"]

            self.orchestrator_plan = self._agent_task("plan", make)
            self.orchestrator_summary = holder["t"].summary
        except Exception as e:
            self.fail("plan", str(e))
            return False
        ds = self.orchestrator_plan.data_sourcing
        self.finish("plan", f"{len(self.orchestrator_plan.steps)} steps · {ds.human_demos} human + {ds.ai_episodes} AI episodes")
        self.log("plan", self.orchestrator_summary or "Production plan ready")
        return True

    def _run_environment(self, feedback: Optional[str]) -> bool:
        self.start("environment", "Environment Agent is planning the scene")
        try:
            task_holder = {}

            def make():
                task_holder["t"] = EnvironmentTask(self.spec, self.bb, feedback, self.orchestrator_plan)
                return task_holder["t"]

            self.plan = self._agent_task("environment", make)
            self.plan_summary = task_holder["t"].summary
            self.job = build_sim_job(f"{self.id[:8]}-job", self.id, self.spec, self.plan)
        except Exception as e:
            self.fail("environment", str(e))
            return False
        self.log("environment", self.plan_summary or "Scene planned")
        with self.lock:
            self.stages["environment"]["detail"] = "Building the scene in MuJoCo and rendering a preview"
        self.save()
        self.has_scene_preview = self._render_scene_preview()
        self.finish("environment", f"{len(self.plan.layout_ids)} layouts · {len(self.plan.style_ids)} styles · "
                                   f"{len(self.plan.cameras)} cameras" + ("" if self.has_scene_preview else " · no preview"))
        return True

    def _render_scene_preview(self) -> bool:
        """Build the first planned layout/style in MuJoCo and save a render as scene.png."""
        if not SIM_PYTHON.exists():
            self.log("environment", "Scene preview skipped: sim/.venv is not installed")
            return False
        out = self.dir / "work" / "scene"
        cmd = [str(SIM_PYTHON), str(REPO / "sim" / "scripts" / "render_scene.py"), "--task", "CafeServeCoffee",
               "--layout", str(self.plan.layout_ids[0]), "--style", str(self.plan.style_ids[0]),
               "--out", str(out), "--size", "720", "--seed", str(self.plan.base_seed % 1000)]
        try:
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, MUJOCO_GL="cgl"))
            src = out / f"CafeServeCoffee_s{self.plan.style_ids[0]}_overview_a.png"
            if r.returncode != 0 or not src.exists():
                self.log("environment", f"Scene preview failed: {(r.stderr or r.stdout)[-300:]}")
                return False
            (self.dir / "scene.png").write_bytes(src.read_bytes())
            return True
        except Exception as e:
            self.log("environment", f"Scene preview failed: {e}")
            return False

    # ---- human play -------------------------------------------------------------------------

    def _run_play(self) -> bool:
        """Wait while a human plays the task in the browser game; QA each submitted episode as it arrives."""
        self._play_done.clear()
        self.start("play", "Waiting for a human demo in the browser game")
        with self.lock:
            self.awaiting = "play"
        self.save()
        seen = {h.file for h in self.human_episodes}
        while not self._play_done.wait(POLL_S):
            for path in sorted((EPISODES / self.id).glob("*.hdf5")) if (EPISODES / self.id).exists() else []:
                if path.name in seen:
                    continue
                seen.add(path.name)
                self._qa_human_episode(path)
        with self.lock:
            self.awaiting = None
        n = len(self.human_episodes)
        passed = sum(1 for h in self.human_episodes if h.qa_passed)
        self.finish("play", f"{n} human episode(s), {passed} passed QA" if n else "skipped: no human episode")
        self.log("play", f"Human play closed with {n} episode(s)")
        return True

    def continue_play(self) -> None:
        with self.lock:
            if self.awaiting != "play":
                raise RuntimeError("this order is not waiting for human play")
        self._play_done.set()

    def _qa_human_episode(self, path: Path) -> None:
        import h5py

        with self.lock:
            self.stages["play"]["detail"] = f"QA on {path.name}: structure + replay in MuJoCo"
        self.save()
        rel = os.path.relpath(path, REPO)
        with h5py.File(path, "r") as f:
            d = f["data"]["demo_1"]
            steps = len(d["actions"])
            stage = int(d.attrs.get("stage", 0))
            success = bool(d.attrs.get("success", False))
            player = str(d.attrs.get("player_id", "human"))
            has_frames = "obs" in d and len(d["obs"].keys()) > 0
        ep = HumanEpisode(file=rel, player_id=player, steps=steps, stages_completed=stage, success=success,
                          has_frames=has_frames, at=time.time())
        try:
            r = subprocess.run([str(SIM_PYTHON), "-m", "agriphilo.game_qa", str(path), "--replay"], cwd=str(REPO),
                               capture_output=True, text=True, timeout=900, env=dict(os.environ, MUJOCO_GL="cgl"))
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if not lines:
                raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1][:300] if (r.stderr or r.stdout).strip() else f"exit {r.returncode}")
            verdict = json.loads(lines[-1])
            ep.qa_passed = bool(verdict["passed"])
            ep.qa_reasons = list(verdict["reasons"])
            ep.replay_max_qpos_err = verdict["metrics"].get("replay_max_qpos_err")
        except Exception as e:
            ep.qa_passed, ep.qa_reasons = False, [f"QA could not run: {e}"]
        if ep.qa_passed and self.ledger is not None:
            # pay the player from this order's hold: episode price -> player share + platform fee
            try:
                self.ledger.capture(self.id, path.name, CREDITS_PER_EPISODE, player, "human", PLAYER_SHARE)
                ep.reward_credits = PLAYER_SHARE
            except Exception as e:
                ep.qa_reasons.append(f"not paid: {e}")
        with self.lock:
            self.human_episodes.append(ep)
            self.stages["play"]["detail"] = f"{len(self.human_episodes)} human episode(s) received"
        if has_frames:
            self.has_human_preview = self._human_preview(path)
        self.log("play", f"Human episode {path.name}: {stage}/{len(TASK_STEPS)} stages, QA {'pass' if ep.qa_passed else 'fail'}")

    def _human_preview(self, path: Path) -> bool:
        import h5py
        import numpy as np
        from PIL import Image

        with h5py.File(path, "r") as f:
            obs = f["data"]["demo_1"]["obs"]
            rows = []
            for cam in sorted(obs.keys()):
                frames = obs[cam][()]
                picks = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
                rows.append(np.concatenate([frames[i] for i in picks], axis=1))
        width = min(r.shape[1] for r in rows)
        img = Image.fromarray(np.concatenate([r[:, :width] for r in rows], axis=0))
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
        img.save(self.dir / "human_preview.png")
        return True

    def _simulate(self, job: SimJob, out: Path, stage: str, label: str = "") -> None:
        def progress(done: int, total: int) -> None:
            with self.lock:
                self.progress = {"done": done, "total": total}
                self.stages[stage]["detail"] = f"{label}{done}/{total} episodes"
            self.save()

        result = self.worker.run(job, out, progress)
        self.compute_seconds += result.compute_seconds

    def _run_simulation(self, raw: Path) -> bool:
        self.start("simulation", f"Submitting {len(self.job.episodes)} episodes to {self.worker.name}")
        self.compute_seconds = 0.0
        try:
            self._simulate(self.job, raw, "simulation")
        except Exception as e:
            self.fail("simulation", f"{self.worker.name}: {e}")
            return False
        self.finish("simulation", f"{len(self.job.episodes)} episodes on {self.worker.name}")
        self.log("simulation", f"Simulated {len(self.job.episodes)} episodes ({self.worker.name})")
        return True

    def _run_collection_and_checks(self, raw: Path) -> bool:
        self.start("collection", "Packaging episodes")
        try:
            self.manifest = build_manifest(self.id, raw, self.spec.sensors)
            self.has_preview = preview(raw, self.dir / "preview.png")
            self.finish("collection", f"{self.manifest.episodes} episodes · {self.manifest.frames} frames")
            self.start("qa", "Running structure, task, diversity and reproducibility checks")
            metrics = run_checks(raw, self.job, self.spec, self.worker, self.dir / "work")
            for topup in range(MAX_TOPUPS):
                invalid = metrics.episodes_total - metrics.episodes_valid
                if not invalid:
                    break
                self.log("qa", f"{invalid} episodes failed checks; generating {invalid} replacements")
                extra = build_sim_job(f"{self.id[:8]}-topup{topup}", self.id, self.spec, self.plan, count=invalid,
                                      seed_offset=1000 + topup, id_offset=len(self.job.episodes))
                extra_path = self.dir / "work" / f"topup_{topup}.hdf5"
                self._simulate(extra, extra_path, "qa", "Simulating replacements: ")
                merge_into(raw, extra_path)
                extra_path.unlink()
                self.job = self.job.model_copy(update={"episodes": self.job.episodes + extra.episodes})
                metrics = run_checks(raw, self.job, self.spec, self.worker, self.dir / "work")
            self.qa_metrics = metrics
            self.manifest = build_manifest(self.id, raw, self.spec.sensors)
        except Exception as e:
            traceback.print_exc()
            self.fail("qa" if self.stages["qa"]["status"] == "running" else "collection", str(e))
            return False
        self.log("qa", f"Checks done: {metrics.episodes_valid}/{metrics.episodes_total} episodes valid")
        return True

    def _run_qa_agent(self) -> bool:
        with self.lock:
            self.stages["qa"]["detail"] = "QA Agent is writing the report"
        try:
            self.qa_report = self._agent_task("qa", lambda: QATask(
                self.spec, self.text, self.qa_metrics, self.qa_metrics.episodes_valid, self.bb,
                [h.model_dump() for h in self.human_episodes]))
        except Exception as e:
            self.fail("qa", str(e))
            return False
        self.finish("qa", f"{self.qa_report.verdict} · {self.qa_metrics.episodes_valid} episodes delivered")
        self.log("qa", self.qa_report.headline)
        return True

    def _deliver_files(self, raw: Path) -> None:
        out = self.dir / "delivery"
        keep = [e.episode_id for e in self.qa_metrics.episodes if e.valid]
        export(raw, out / "dataset.hdf5", keep)
        manifest = build_manifest(self.id, out / "dataset.hdf5", self.spec.sensors)
        (out / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        (out / "spec.json").write_text(self.spec.model_dump_json(indent=2))
        (out / "qa_report.json").write_text(json.dumps({
            "report": self.qa_report.model_dump(), "metrics": self.qa_metrics.model_dump()}, indent=2, ensure_ascii=False))
        (out / "qa_report.md").write_text(self._qa_markdown(manifest))
        with self.lock:
            self.manifest = manifest

    def _qa_markdown(self, manifest: Manifest) -> str:
        r, m = self.qa_report, self.qa_metrics
        lines = [f"# QA report · order {self.id[:8]}", "", f"**Verdict:** {r.verdict}", "", r.headline, "",
                 "## Findings", *[f"- {x}" for x in r.findings], ""]
        if r.recommended_fixes:
            lines += ["## Recommended fixes", *[f"- {x}" for x in r.recommended_fixes], ""]
        lines += ["## Checks", "", "| Check | Category | Status | Detail |", "| --- | --- | --- | --- |",
                  *[f"| {c.name} | {c.category} | {c.status} | {c.detail} |" for c in m.checks], "",
                  f"Delivered {manifest.episodes} episodes ({manifest.frames} frames) from {manifest.worker} "
                  f"({manifest.simulator_version}). sha256 `{manifest.sha256}`.", "",
                  "A pass means the data meets the dataset contract and the simulation success rule. "
                  "It does not show improvement on a real robot."]
        return "\n".join(lines) + "\n"

    def _run_quote(self) -> None:
        self.start("quote", "Payment Agent is preparing the quote")
        facts = {
            "order_text": self.text,
            "mode": self.spec.mode,
            "episodes_ordered": self.spec.episode_count,
            "episodes_delivered": self.manifest.episodes,
            "human_demo_episodes": len(self.human_episodes),
            "gpu_hours": round(self.compute_seconds / 3600, 2),
            "qa_verdict": self.qa_report.verdict,
            "simulator": self.manifest.worker,
        }
        try:
            holder = {}

            def make():
                holder["t"] = PaymentTask(facts, self.bb)
                return holder["t"]

            self.quote = self._agent_task("quote", make)
            self.quote_summary = holder["t"].summary
        except Exception as e:
            return self.fail("quote", str(e))
        with self.lock:
            self.awaiting = "payment"
        self.finish("quote", f"${self.quote.total_usd:,.2f}")
        self.log("quote", f"Quote ready: ${self.quote.total_usd:,.2f}")

    def pay(self) -> None:
        with self.lock:
            if self.awaiting != "payment":
                raise RuntimeError("this order is not awaiting payment")
            self.awaiting = None
            self.payment = Payment(status="paid", amount_usd=self.quote.total_usd,
                                   reference="TEST-" + uuid.uuid4().hex[:8].upper(), paid_at=time.time())
            self.stages["delivery"].update(status="done", started_at=time.time(), finished_at=time.time(),
                                           detail="Files available")
        self.log("delivery", f"Paid ${self.payment.amount_usd:,.2f} (test mode, {self.payment.reference}); files unlocked")

    def start_checkout(self, origin: str) -> str:
        """Create a Stripe Checkout Session (test mode) for the quote; returns the hosted checkout URL."""
        from . import stripe_checkout

        with self.lock:
            if self.awaiting != "payment":
                raise RuntimeError("this order is not awaiting payment")
            total = self.quote.total_usd
        s = stripe_checkout.create_session(
            self.id, total, f"{self.manifest.episodes if self.manifest else 0} QA-passed CafeServeCoffee episodes",
            success_url=f"{origin}/requester?order={self.id}&checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{origin}/requester?order={self.id}&checkout=cancel")
        self.log("delivery", f"Stripe checkout opened ({s['id'][:18]}…)")
        return s["url"]

    def confirm_checkout(self, session_id: str) -> None:
        """Mark the order paid only if Stripe says this session paid the full quote for this order."""
        from . import stripe_checkout

        s = stripe_checkout.retrieve_session(session_id)
        with self.lock:
            if self.payment:
                return
            if self.awaiting != "payment":
                raise RuntimeError("this order is not awaiting payment")
            if (s.get("metadata") or {}).get("order_id") != self.id:
                raise RuntimeError("checkout session belongs to another order")
            if s.get("payment_status") != "paid":
                raise RuntimeError(f"Stripe reports payment_status={s.get('payment_status')}")
            if s.get("amount_total") != int(round(self.quote.total_usd * 100)):
                raise RuntimeError("paid amount does not match the quote")
            self.awaiting = None
            self.payment = Payment(status="paid", mode="stripe_test", amount_usd=s["amount_total"] / 100,
                                   reference=s.get("payment_intent") or s["id"], paid_at=time.time())
            self.stages["delivery"].update(status="done", started_at=time.time(), finished_at=time.time(),
                                           detail="Files available")
        self.log("delivery", f"Paid ${self.payment.amount_usd:,.2f} via Stripe (test mode, {self.payment.reference}); files unlocked")

    def retry(self) -> None:
        """Restart a failed order from the stage that failed."""
        with self.lock:
            failed = [k for k, s in self.stages.items() if s["status"] == "failed"]
            if not failed:
                raise RuntimeError("nothing to retry")
            stage = failed[0]
            if stage == "spec":
                reset, run = ["spec"], self._restart_customer
            elif stage == "quote":
                reset, run = ["quote"], self._run_quote
            elif stage == "delivery":
                raise RuntimeError("delivery cannot fail")
            elif stage == "play":
                reset, run = ["play", "simulation", "collection", "qa", "quote"], self._run_production
            else:
                reset, run = ["plan", "environment", "simulation", "collection", "qa", "quote"], self._run_production
            for k in reset:
                self.stages[k] = _stage()
        self.log(stage, "Retrying")
        self._spawn(run)

    def _restart_customer(self) -> None:
        """New Customer Agent conversation seeded with everything the customer has said so far."""
        said = [t["text"] for t in self.transcript if t["role"] == "customer" and t.get("text")]
        self.start("spec", "Customer Agent is reading the order")
        self.session = CustomerSession("\n\n".join(said), auto=self.auto, bb=self.bb)
        self.stages["spec"]["thread_id"] = self.session.thread_id
        self._run_customer()

    # ---- plumbing ---------------------------------------------------------------------------

    def _spawn(self, fn) -> None:
        def wrapped():
            try:
                fn()
            except Exception as e:
                traceback.print_exc()
                running = [k for k, s in self.stages.items() if s["status"] in ("running", "waiting")]
                self.fail(running[0] if running else "spec", str(e))
                if self.session and self.session.state == "running":
                    self.session.close()

        threading.Thread(target=wrapped, daemon=True).start()

    def file_path(self, name: str) -> Optional[Path]:
        if name == "preview.png" and self.has_preview:
            return self.dir / "preview.png"
        if name == "scene.png" and self.has_scene_preview:
            return self.dir / "scene.png"
        if name == "human_preview.png" and self.has_human_preview:
            return self.dir / "human_preview.png"
        if name in DELIVERY_FILES and self.payment:
            return self.dir / "delivery" / name
        return None

    def to_dict(self) -> Dict[str, Any]:
        dump = lambda m: m.model_dump() if m else None  # noqa: E731
        with self.lock:
            return {
                "id": self.id, "text": self.text, "auto": self.auto, "created_at": self.created_at,
                "state": self.state, "awaiting": self.awaiting,
                "stages": [{"key": k, "label": label, "owner": owner, **self.stages[k]} for k, label, owner in STAGES],
                "events": self.events, "transcript": self.transcript, "questions": self.questions,
                "spec": dump(self.spec), "spec_summary": self.spec_summary,
                "credits": self._credits(),
                "orchestrator_plan": dump(self.orchestrator_plan), "orchestrator_summary": self.orchestrator_summary,
                "plan": dump(self.plan), "plan_summary": self.plan_summary, "has_scene_preview": self.has_scene_preview,
                "play": self._play_info(),
                "job": {"job_id": self.job.job_id, "episodes": [e.model_dump() for e in self.job.episodes]} if self.job else None,
                "progress": self.progress, "compute_seconds": self.compute_seconds,
                "manifest": dump(self.manifest), "qa_metrics": dump(self.qa_metrics), "qa_report": dump(self.qa_report),
                "quote": dump(self.quote), "quote_summary": self.quote_summary, "payment": dump(self.payment),
                "has_preview": self.has_preview, "worker": self.worker.name,
                "files": [n for n in DELIVERY_FILES if self.payment],
            }

    def _credits(self) -> Dict[str, Any]:
        balance = self.ledger.balance(f"wallet:{CUSTOMER}") if self.ledger is not None else None
        return {"estimate": self.credit_estimate, "held": self.credits_held, "balance": balance}

    def _play_info(self) -> Dict[str, Any]:
        base = os.environ.get("AGRIPHILO_GAME_SERVER_URL", "")
        urls = {}
        if base and self.plan:
            from urllib.parse import urlencode

            q = {"game_id": self.id, "task": self.plan.task, "robot": self.plan.robot,
                 "layout_ids": self.plan.layout_ids[0], "style_ids": self.plan.style_ids[0]}
            urls = {role: f"{base}?{urlencode(dict(q, role=role))}" for role in ("player", "viewer")}
        return {"game_server_url": base, "player_url": urls.get("player", ""), "viewer_url": urls.get("viewer", ""),
                "human_episodes": [h.model_dump() for h in self.human_episodes],
                "has_human_preview": self.has_human_preview}

    def marketplace_item(self) -> Optional[Dict[str, Any]]:
        """This order's game as a marketplace listing, once its environment is built."""
        if not self.plan or self.stages["play"]["status"] == "pending":
            return None
        eps = self.human_episodes
        return {
            "order_id": self.id, "title": "Serve a coffee at the cafe counter", "task": self.plan.task,
            "task_steps": list(self.spec.task_steps), "robot": self.plan.robot,
            "scene": f"{self.plan.task} · layout {self.plan.layout_ids[0]} · style {self.plan.style_ids[0]}",
            "scene_image_url": f"/api/orders/{self.id}/files/scene.png" if self.has_scene_preview else None,
            "requester": CUSTOMER, "status": "open" if self.awaiting == "play" else "closed",
            "reward_per_episode": PLAYER_SHARE,
            "episodes_wanted": max(1, self.orchestrator_plan.data_sourcing.human_demos) if self.orchestrator_plan else 1,
            "episodes_submitted": len(eps), "episodes_passed": sum(1 for h in eps if h.qa_passed),
            "play_url": self._play_info()["player_url"], "created_at": self.created_at,
        }

    def summary(self) -> Dict[str, Any]:
        d = self.to_dict()
        for k in ("job", "qa_metrics", "events", "transcript"):
            d.pop(k)
        return d

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"order.json.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False))
        tmp.replace(self.dir / "order.json")

    @classmethod
    def load(cls, path: Path, bb, worker: SimWorker, ledger=None) -> "Order":
        """Restore a saved order. Work that was in flight when the server stopped is marked failed."""
        d = json.loads(path.read_text())
        if {s["key"] for s in d["stages"]} != {k for k, _, _ in STAGES}:
            raise ValueError("order from the earlier greenhouse pipeline; not loaded")
        o = cls(d["text"], d["auto"], bb, worker, order_id=d["id"], ledger=ledger)
        o.created_at = d["created_at"]
        o.stages = {s["key"]: {k: s[k] for k in _stage()} for s in d["stages"]}
        o.awaiting, o.events, o.transcript = d["awaiting"], d["events"], d["transcript"]
        o.spec = DatasetSpec.model_validate(d["spec"]) if d["spec"] else None
        o.spec_summary, o.plan_summary, o.quote_summary = d["spec_summary"], d["plan_summary"], d["quote_summary"]
        o.plan = EnvironmentPlan.model_validate(d["plan"]) if d["plan"] else None
        o.orchestrator_plan = OrchestratorPlan.model_validate(d["orchestrator_plan"]) if d.get("orchestrator_plan") else None
        o.orchestrator_summary = d.get("orchestrator_summary", "")
        o.has_scene_preview = d.get("has_scene_preview", False)
        o.credits_held = (d.get("credits") or {}).get("held", 0.0)
        play = d.get("play") or {}
        o.human_episodes = [HumanEpisode.model_validate(h) for h in play.get("human_episodes", [])]
        o.has_human_preview = play.get("has_human_preview", False)
        o.manifest = Manifest.model_validate(d["manifest"]) if d["manifest"] else None
        o.qa_metrics = QAMetrics.model_validate(d["qa_metrics"]) if d["qa_metrics"] else None
        o.qa_report = QAReport.model_validate(d["qa_report"]) if d["qa_report"] else None
        o.quote = Quote.model_validate(d["quote"]) if d["quote"] else None
        o.payment = Payment.model_validate(d["payment"]) if d["payment"] else None
        o.progress, o.compute_seconds, o.has_preview = d["progress"], d["compute_seconds"], d["has_preview"]
        if d["job"] and o.spec and o.plan:
            o.job = SimJob(job_id=d["job"]["job_id"], order_id=o.id, spec=o.spec, plan=o.plan,
                           episodes=d["job"]["episodes"], sensors=o.spec.sensors, success_rule=o.spec.success_rule)
        # A live Brainbase conversation or background run cannot resume across a restart.
        if o.awaiting in ("answers", "play") or any(s["status"] in ("running", "waiting") for s in o.stages.values()):
            stuck = next((k for k, s in o.stages.items() if s["status"] in ("running", "waiting")), "spec")
            o.stages[stuck].update(status="failed", error="Server restarted while this stage was in progress")
            o.awaiting = None
        return o


class OrderStore:
    def __init__(self, bb, worker: SimWorker, ledger=None):
        self.bb, self.worker, self.ledger = bb, worker, ledger
        self.orders: Dict[str, Order] = {}
        self.lock = threading.Lock()
        ROOT.mkdir(parents=True, exist_ok=True)
        for path in sorted(ROOT.glob("*/order.json")):
            try:
                o = Order.load(path, bb, worker, ledger)
                self.orders[o.id] = o
            except Exception as e:
                print(f"skipping {path}: {e}")

    def create(self, text: str, auto: bool) -> Order:
        o = Order(text, auto, self.bb, self.worker, ledger=self.ledger)
        with self.lock:
            self.orders[o.id] = o
        o.begin()
        return o

    def marketplace(self) -> List[Dict[str, Any]]:
        items = [i for i in (o.marketplace_item() for o in self.all()) if i]
        return sorted(items, key=lambda i: (i["status"] != "open", -i["created_at"]))

    def player(self, player_id: str) -> Dict[str, Any]:
        eps = []
        for o in self.all():
            for h in o.human_episodes:
                if h.player_id == player_id:
                    eps.append({"order_id": o.id, "title": "Serve a coffee at the cafe counter",
                                "file": Path(h.file).name, "stages_completed": h.stages_completed,
                                "success": h.success, "qa_passed": h.qa_passed, "qa_reasons": h.qa_reasons,
                                "reward_credits": h.reward_credits, "at": h.at})
        balance = self.ledger.balance(f"payable:{player_id}") if self.ledger is not None else 0.0
        return {"player_id": player_id, "balance": balance,
                "episodes": sorted(eps, key=lambda e: -(e["at"] or 0))}

    def get(self, order_id: str) -> Order:
        return self.orders[order_id]

    def all(self) -> List[Order]:
        return sorted(self.orders.values(), key=lambda o: o.created_at, reverse=True)
