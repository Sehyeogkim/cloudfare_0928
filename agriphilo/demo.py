"""Offline stand-in for the Brainbase client, for trying the app without an API key or credits.

Each agent answers with a scripted reply built from its input, after TURN_S seconds:
- Customer: asks one round of questions (unless auto), then returns a canned spec from examples/.
- Environment / QA / Payment: derive their reply from the payload they are given.
Replies are marked "[Demo]" so they are never mistaken for real agent output.
"""
import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

EXAMPLES = Path(__file__).parent.parent / "examples"
TURN_S = 5


def _customer(t: Dict[str, Any]) -> Dict[str, Any]:
    text = t["input"]
    wants_defaults = "Do not ask more questions" in text or "fill in defaults" in text.lower()
    if not t["replies"] and not wants_defaults:
        return {"status": "need_info", "questions": [
            "How many episodes do you need, roughly?",
            "Which robot and gripper do you use in the field?",
        ]}
    failure = any(w in text.lower() for w in ("fail", "실패", "재현", "reproduce"))
    spec = json.loads((EXAMPLES / f"sample_cafe{'_failure' if failure else ''}_spec.json").read_text())
    stated = re.search(r"(\d+)\s*(?:successful\s+)?(?:episodes|demos|에피소드|개)", text)
    if stated:
        spec["episode_count"] = int(stated.group(1))
    return {"status": "ready", "dataset_spec": spec, "summary": "[Demo] Canned cafe spec; no agent was called."}


def _orchestrator(t: Dict[str, Any]) -> Dict[str, Any]:
    spec = json.loads(t["input"])["dataset_spec"]
    n = spec["episode_count"]
    cams = ", ".join(c["name"] for c in spec["data"]["cameras"])
    return {"status": "ready", "summary": "[Demo] Scripted production plan.", "orchestrator_plan": {
        "simulator": "mujoco", "framework": "robocasa",
        "steps": [
            {"stage": "environment", "agent": "Environment Agent",
             "action": "Pick cafe-like RoboCasa layouts/styles, cameras and object variety; build the scene and render it",
             "output": "EnvironmentPlan + scene preview"},
            {"stage": "play", "agent": "Human player",
             "action": "Play CafeServeCoffee once in the browser game (OSC keyboard teleop); customer watches live",
             "output": "1 human demo (HDF5 states/actions) with replay QA"},
            {"stage": "simulation", "agent": "AI players",
             "action": f"Generate {n} episodes from the human demo across the planned layouts and styles",
             "output": f"{n} AI episodes in the requested schema"},
            {"stage": "collection", "agent": "Data Collection",
             "action": f"Package {cams} frames, joint states, eef pose and actions at {spec['data']['control_hz']} Hz",
             "output": "dataset.hdf5 + manifest"},
            {"stage": "qa", "agent": "QA Agent",
             "action": "Check structure, success per task stage, layout/style coverage and seed replay",
             "output": "QA report; failing episodes excluded"},
            {"stage": "quote", "agent": "Payment Agent", "action": "Price QA-passed episodes from the rate card",
             "output": "Quote"},
        ],
        "scene_build": [
            "Load a RoboCasa kitchen layout with a long straight counter and a coffee machine fixture",
            "Put a mug of coffee on the open counter in front of the robot and a tray right next to it",
            "Spawn PandaOmron in front of the coffee machine with an OSC_POSE arm controller",
            "Register the 5 task stage checks in CafeServeCoffee",
        ],
        "task_stages": [
            {"name": s_, "success_check": c} for s_, c in zip(spec["task_steps"], [
                "the mug is grasped (gripper contact on both fingers)",
                "the mug rests inside the tray and the gripper has released it",
            ])],
        "data_sourcing": {"human_demos": 1, "ai_episodes": n,
                          "method": "Demo augmentation across layouts/styles plus scripted AI-player rollouts"},
        "data_mapping": [f"rgb <- {cams} renders", "joint_state <- robot0_joint_pos + gripper qpos",
                         "eef_pose <- robot0_eef_pos/quat", "action <- 12-D OSC + gripper + base command",
                         "task_stage <- CafeServeCoffee.update_stages()"],
        "qa_criteria": ["all requested streams present, equal length, finite", "time monotonic, actions in [-1, 1]",
                        "success = both task stages completed within the time limit",
                        "every planned layout and style covered", "human demo replays within 1e-2 qpos"],
        "risks": ["The mug must be carried upright so it does not tip on the tray rim", "Liquid is not simulated"],
        "rationale": "[Demo] One human demo seeds AI players; QA checks every stage.",
    }}


def _environment(t: Dict[str, Any]) -> Dict[str, Any]:
    spec = json.loads(t["input"].split("\n", 1)[0])["dataset_spec"]
    many = {"single": 1, "few": 3, "many": 10}
    layouts = [9, 5, 7, 1, 2, 3, 4, 6, 8, 10][: many[spec["environment"]["layout_variety"]]]
    styles = [3, 8, 1, 2, 4, 5, 6, 7, 9, 10][: many[spec["environment"]["style_variety"]]]
    hz = spec["data"]["control_hz"]
    return {"status": "ready", "summary": "[Demo] Scripted cafe scene plan.", "environment_plan": {
        "layout_ids": layouts, "style_ids": styles, "cameras": spec["data"]["cameras"], "control_hz": hz,
        "max_steps": int(math.ceil(spec["success_rule"]["time_limit_s"] * hz)),
        "object_instance_randomization": True, "placement_jitter_m": 0.02, "base_seed": 20260928,
        "focus": ["[Demo] mug tipping on the tray rim" if spec["mode"] == "failure" else "[Demo] even layout/style coverage"],
        "rationale": "[Demo] Straight-counter layouts and cafe-like styles; mug and tray models vary per episode.",
    }}


# Keyword -> RoboCasa task for the scripted Game Environment Agent. First match wins; the real agent reasons instead.
GAME_TASK_KEYWORDS = [
    (("coffee", "커피", "mug", "머그"), "CoffeeSetupMug"),
    (("sink", "싱크", "개수대"), "PickPlaceCounterToSink"),
    (("cabinet", "찬장", "수납장"), "PickPlaceCounterToCabinet"),
    (("microwave", "전자레인지"), "PickPlaceCounterToMicrowave"),
    (("drawer", "서랍"), "OpenDrawer"),
    (("pick", "place", "집어", "옮기"), "PickPlaceCounterToCabinet"),
]


def _game_environment(t: Dict[str, Any]) -> Dict[str, Any]:
    p = json.loads(t["input"].split("\n", 1)[0])
    text = p["order_text"].lower()
    task = next((task for words, task in GAME_TASK_KEYWORDS if any(w in text for w in words)), "PickPlaceCounterToCabinet")
    spec = p.get("dataset_spec")
    stated = re.search(r"(\d+)\s*(?:successful\s+)?(?:episodes|demos|에피소드|개)", text)
    count = spec["episode_count"] if spec else (int(stated.group(1)) if stated else 50)
    variety = any(w in text for w in ("variety", "다양", "diverse", "many kitchens"))
    hz, limit_s = 20, (60.0 if task.startswith("PickPlace") else 45.0)
    return {"status": "ready", "summary": f"[Demo] Scripted game: {task}.", "game_spec": {
        "task": task,
        "layout_ids": list(range(1, 11)) if variety else [1, 2, 3],
        "style_ids": list(range(1, 11)) if variety else [1, 2, 3],
        "robot": "PandaOmron",
        "cameras": [
            {"name": "robot0_agentview_left", "width": 256, "height": 256, "fps": 20},
            {"name": "robot0_eye_in_hand", "width": 256, "height": 256, "fps": 20},
        ],
        "control_hz": hz, "max_steps": int(math.ceil(limit_s * hz)),
        "episodes_requested": count,
        "success_rule": {"type": "robocasa_check_success", "time_limit_s": limit_s},
        "credit_per_pass": 2.0 if task.startswith("PickPlace") else 1.5,
        "base_seed": 20260928,
        "rationale": f"[Demo] Keyword match to {task}; PandaOmron proxy robot; agent view + wrist camera.",
    }}


def _qa(t: Dict[str, Any]) -> Dict[str, Any]:
    p = json.loads(t["input"])
    m = p["qa_metrics"]
    warn = [c["name"] for c in m["checks"] if c["status"] != "pass"]
    failed = [c["name"] for c in m["checks"] if c["status"] == "fail"]
    verdict = "fail" if failed else ("pass_with_notes" if warn else "pass")
    reasons = ", ".join(f"{k} {v}" for k, v in m["failure_reasons"].items()) or "none"
    return {"status": "ready", "summary": "[Demo] Scripted QA report.", "qa_report": {
        "verdict": verdict,
        "headline": f"[Demo] {p['episodes_delivered']} valid episodes, success rate {m['success_rate']:.0%}.",
        "findings": [
            f"{m['episodes_total'] - m['episodes_valid']} episodes failed structural checks and were excluded.",
            f"Failure reasons among unsuccessful episodes: {reasons}.",
            f"Reproducibility: {m['repro_matched']}/{m['repro_checked']} re-run episodes matched.",
        ] + ([f"Checks with notes: {', '.join(warn)}."] if warn else []),
        "recommended_fixes": [], "rerun_needed": False, "rerun_focus": None,
    }}


def _payment(t: Dict[str, Any]) -> Dict[str, Any]:
    p = json.loads(t["input"])
    d, items = p["delivery"], p["rate_card"]["items"]

    def line(sku: str, qty: float) -> Dict[str, Any]:
        price = items[sku]["unit_price_usd"]
        return {"sku": sku, "description": items[sku]["description"], "quantity": qty, "unit": items[sku]["unit"],
                "unit_price_usd": price, "amount_usd": round(qty * price, 2)}

    lines = [line("scene_setup", 1), line("episode", d["episodes_delivered"]),
             line("gpu_hour", max(0.1, math.ceil(d["gpu_hours"] * 10) / 10)), line("qa_report", 1)]
    if d["mode"] == "failure":
        lines.append(line("failure_repro", 1))
    subtotal = round(sum(li["amount_usd"] for li in lines), 2)
    return {"status": "ready", "summary": "[Demo] Scripted quote.", "quote": {
        "line_items": lines, "subtotal_usd": subtotal, "total_usd": subtotal, "valid_days": 30,
        "notes": ["[Demo] Placeholder pilot rates; no real charge."],
    }}


def _episode_qa(t: Dict[str, Any]) -> Dict[str, Any]:
    facts = json.loads(t["input"].rsplit("Review this episode:\n", 1)[-1])
    done, total = facts["task_stages_completed"], facts["task_stages_total"]
    idle = facts["idle_ratio"]
    if done < total:
        v, q, why = "fail", 1 if done == 0 else 2, f"Stopped after {done} of {total} task stages."
    elif idle > 0.6:
        v, q, why = "fail", 2, f"Task completed but the robot was idle {idle:.0%} of the time."
    else:
        q = 5 if facts["duration_s"] < 40 and idle < 0.3 else 4 if facts["duration_s"] < 90 else 3
        v, why = "pass", f"All {total} stages completed in {facts['duration_s']:.0f} s."
    return {"status": "ready", "episode_review": {"verdict": v, "quality": q, "reason": f"[Demo] {why}"}}


HANDLERS = {"Customer Agent": _customer, "Orchestrator Agent": _orchestrator, "Environment Agent": _environment,
            "Game Environment Agent": _game_environment, "QA Agent": _qa, "Episode QA Agent": _episode_qa, "Payment Agent": _payment}


class DemoBrainbase:
    def __init__(self):
        self.threads: Dict[str, Dict[str, Any]] = {}

    def create_thread(self, agent: Dict[str, Any], input: str) -> Dict[str, Any]:
        tid = str(uuid.uuid4())
        role = agent.get("title", "").split("·")[-1].strip()
        self.threads[tid] = {"role": role, "input": input, "turn_at": time.time(), "replies": [], "user_msgs": 1}
        return {"thread_id": tid, "agent_id": f"demo-{role}", "status": "running"}

    def _advance(self, tid: str) -> Dict[str, Any]:
        t = self.threads[tid]
        if t["user_msgs"] > len(t["replies"]) and time.time() - t["turn_at"] >= TURN_S:
            t["replies"].append(json.dumps(HANDLERS[t["role"]](t), ensure_ascii=False))
        return t

    def get_thread(self, thread_id: str) -> Dict[str, Any]:
        t = self._advance(thread_id)
        running = t["user_msgs"] > len(t["replies"])
        return {"id": thread_id, "status": "running" if running else "success", "machine_id": None}

    def post_message(self, thread_id: str, content: str) -> Dict[str, Any]:
        t = self.threads[thread_id]
        t["user_msgs"] += 1
        t["turn_at"] = time.time()
        t["input"] += "\n" + content
        return {"messages": [], "run_started": True}

    def assistant_reply_count(self, thread_id: str) -> int:
        return len(self._advance(thread_id)["replies"])

    def last_assistant_text(self, thread_id: str) -> str:
        return self._advance(thread_id)["replies"][-1]

    def events(self, thread_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        return []

    def delete_machine(self, machine_id: str) -> Dict[str, Any]:
        return {}
