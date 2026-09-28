"""AI player: an LLM (or scripted) planner operates the robot through skill primitives.

    sim/.venv/bin/python -m agriphilo.players.llm_player --task CoffeeSetupMug \
        --planner scripted|brainbase --seed 0 --out runs/episodes/demo --qa

Writes <out>/<episode_id>.hdf5 (collect_demos format, replayable by agriphilo.game_qa),
<episode_id>.plan.json (plan, per-primitive reasoning and outcomes) and <episode_id>.mp4.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .planner import Plan, brainbase_plan, scripted_plan
from .sim_env import Recorder, env_config, make_env, render_video
from .skills import SCENES, Skills

MAX_REPLANS = 2


def play(task: str, planner: str, seed: int, out: Path, robot: str = "PandaOmron", layout: int = 1,
         style: int = 1, camera: str = "robot0_agentview_center", player_id: str = "",
         game_id: str = "demo", verbose: bool = True) -> dict:
    if task not in SCENES:
        raise SystemExit(f"no scene adapter for {task}; available: {sorted(SCENES)}")
    say = print if verbose else (lambda *a, **k: None)
    config = env_config(task, robot, layout, style)
    env = make_env(config, seed=seed, render=False)
    rec = Recorder(env)
    rec.start()
    scene = SCENES[task](env)
    skills = Skills(env, rec, scene)
    player_id = player_id or f"ai-{planner}"
    episode_id = f"{game_id}-{task}-s{seed}-{planner}"

    usage, plans, history = [], [], []
    t0 = time.time()
    for attempt in range(MAX_REPLANS + 1):
        desc = scene.describe()
        if planner == "brainbase":
            say(f"[plan] asking the LLM (attempt {attempt + 1}) ...")
            plan, u = brainbase_plan(desc, history)
            usage.append(u)
            say(f"[plan] got {len(plan.steps)} steps in {u['seconds']}s")
        else:
            plan = scripted_plan(desc)
        plans.append(plan.model_dump())
        failed = None
        for step in plan.steps:
            res = skills.run(step)
            say(f"  {step.primitive:12s} {step.target or '':18s} -> {'ok' if res.ok else 'FAIL: ' + res.message}"
                f"  ({res.steps} steps)  | {step.reason}")
            if not res.ok:
                failed = {"step": step.model_dump(), "error": res.message}
                break
        if failed is None:
            break
        history.append({"plan": plan.model_dump(), "failed": failed})
        # Recover to a known state before replanning: open and back off.
        skills.release()
        skills.retreat(0.15)
    skills._hold(10)
    success = scene.success()
    wall = time.time() - t0
    env.close()

    out.mkdir(parents=True, exist_ok=True)
    attrs = {"player_id": player_id, "player_kind": "ai", "planner": planner,
             "llm_turns": int(sum(u["turns"] for u in usage)), "llm_calls": len(usage),
             "llm_seconds": float(sum(u["seconds"] for u in usage)), "seed": seed,
             "success_at_record": int(success)}
    h5 = rec.save(out / f"{episode_id}.hdf5", config, attrs)
    video = None
    if camera:
        try:
            video = render_video(h5, out / f"{episode_id}.mp4", camera=camera)
        except Exception as e:  # video is a nicety; never lose the episode over it
            say(f"[video] skipped: {e}")
    sidecar = {
        "episode_id": episode_id, "task": task, "robot": robot, "layout_id": layout, "style_id": style,
        "seed": seed, "planner": planner, "player_id": player_id, "player_kind": "ai",
        "instruction": rec.ep_meta.get("lang", ""), "success": success, "steps": len(rec.actions),
        "sim_seconds": len(rec.actions) / 20.0, "wall_seconds": round(wall, 1),
        "plans": plans, "execution": skills.log, "llm_usage": usage,
        "files": {"hdf5": h5.name, "video": video.name if video else None},
    }
    (out / f"{episode_id}.plan.json").write_text(json.dumps(sidecar, indent=1))
    say(f"[done] success={success} steps={len(rec.actions)} -> {h5}")
    return sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="CoffeeSetupMug")
    p.add_argument("--planner", choices=["scripted", "brainbase"], default="scripted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", help="run several seeds (overrides --seed)")
    p.add_argument("--robot", default="PandaOmron")
    p.add_argument("--layout", type=int, default=1)
    p.add_argument("--style", type=int, default=1)
    p.add_argument("--camera", default="robot0_agentview_center", help="'' to skip video")
    p.add_argument("--game-id", default="demo")
    p.add_argument("--player-id", default="")
    p.add_argument("--out", default=None, help="default runs/episodes/<game_id>")
    p.add_argument("--qa", action="store_true", help="run agriphilo.game_qa (structure + replay) after")
    a = p.parse_args()
    out = Path(a.out or f"runs/episodes/{a.game_id}")
    results = []
    for seed in (a.seeds if a.seeds else [a.seed]):
        r = play(a.task, a.planner, seed, out, a.robot, a.layout, a.style, a.camera, a.player_id, a.game_id)
        if a.qa:
            from .. import game_qa
            try:
                v = game_qa.qa_dataset(out / r["files"]["hdf5"], replay=True)[0]
                r["qa"] = v.to_dict()
                print(f"[qa] passed={v.passed} reasons={v.reasons} metrics={v.metrics}")
            except Exception as e:
                r["qa"] = {"passed": False, "reasons": [f"QA could not run: {e}"]}
                print(f"[qa] error: {e}")
            (out / f"{r['episode_id']}.plan.json").write_text(json.dumps(r, indent=1))
        results.append(r)
    if len(results) > 1:
        ok = sum(r["success"] for r in results)
        qa_ok = sum(r.get("qa", {}).get("passed", False) for r in results)
        print(f"[summary] success {ok}/{len(results)}, QA pass {qa_ok}/{len(results)}")


if __name__ == "__main__":
    main()
