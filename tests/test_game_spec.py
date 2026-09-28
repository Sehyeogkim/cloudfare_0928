"""Step B check: each example game order yields a valid GameSpec through the scripted (demo) agent path.

Run: python3 -m pytest tests/test_game_spec.py -q
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agriphilo import demo
from agriphilo.agents.environment import EnvironmentTask, GameEnvironmentTask
from agriphilo.dataset_spec import DatasetSpec
from agriphilo.demo import DemoBrainbase
from agriphilo.game_spec import GameSpec
from agriphilo.models import EnvironmentPlan

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def fast_demo(monkeypatch):
    monkeypatch.setattr(demo, "TURN_S", 0)
    monkeypatch.setattr("agriphilo.agents.base.POLL_S", 0)


@pytest.mark.parametrize("name,task,episodes", [
    ("coffee_game_order.txt", "CoffeeSetupMug", 200),
    ("pickplace_game_order.txt", "PickPlaceCounterToCabinet", 500),
])
def test_example_order_yields_game_spec(name, task, episodes):
    game = GameEnvironmentTask((EXAMPLES / name).read_text(), DemoBrainbase()).run(timeout_s=10)
    assert isinstance(game, GameSpec)
    assert game.task == task
    assert game.episodes_requested == episodes
    assert game.max_steps >= game.success_rule.time_limit_s * game.control_hz


def test_dataset_spec_sets_episode_count():
    spec = DatasetSpec.model_validate(json.loads((EXAMPLES / "sample_cafe_spec.json").read_text()))
    game = GameEnvironmentTask((EXAMPLES / "coffee_game_order.txt").read_text(), DemoBrainbase(), spec).run(timeout_s=10)
    assert game.episodes_requested == spec.episode_count


def test_existing_environment_plan_path_still_works():
    spec = DatasetSpec.model_validate(json.loads((EXAMPLES / "sample_cafe_spec.json").read_text()))
    assert isinstance(EnvironmentTask(spec, DemoBrainbase()).run(timeout_s=10), EnvironmentPlan)


def _base(**kw):
    d = {"task": "CoffeeSetupMug", "layout_ids": [1], "style_ids": [1], "robot": "PandaOmron",
         "cameras": [{"name": "robot0_eye_in_hand", "width": 128, "height": 128, "fps": 20}],
         "control_hz": 20, "max_steps": 1200, "episodes_requested": 10,
         "success_rule": {"time_limit_s": 60}, "credit_per_pass": 1, "base_seed": 1}
    d.update(kw)
    return d


@pytest.mark.parametrize("bad", [
    {"task": "MakeTomatoSoup"},
    {"robot": "UR5e"},
    {"layout_ids": [0]},
    {"style_ids": [61]},
    {"cameras": [{"name": "top_cam", "width": 128, "height": 128, "fps": 20}]},
    {"max_steps": 100},
])
def test_rejects_invalid(bad):
    GameSpec.model_validate(_base())
    with pytest.raises(ValidationError):
        GameSpec.model_validate(_base(**bad))
