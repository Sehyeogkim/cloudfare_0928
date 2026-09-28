import json
import os

import h5py
import numpy as np
import pytest

from agriphilo import game_qa as G

T, S, A = 40, 12, 7


def _write(path, demos, env_info=None):
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["env"] = "CoffeeSetupMug"
        d.attrs["env_info"] = json.dumps(env_info or {"env_name": "CoffeeSetupMug", "robots": "PandaOmron"})
        for i, (states, actions) in enumerate(demos, 1):
            g = d.create_group(f"demo_{i}")
            g.attrs["model_file"] = "<mujoco/>"
            g.attrs["ep_meta"] = json.dumps({"lang": "put the mug under the coffee machine"})
            g.create_dataset("states", data=states)
            g.create_dataset("actions", data=actions)
    return path


def _episode(seed, steps=T):
    r = np.random.default_rng(seed)
    return r.normal(size=(steps, S)), r.uniform(-0.5, 0.5, size=(steps, A))


def _by_id(vs):
    return {v.episode_id: v for v in vs}


def test_valid_and_tampered(tmp_path):
    good = _episode(0)
    nan_s, nan_a = _episode(1); nan_a[5, 2] = np.nan
    mism = (_episode(2)[0][:-1], _episode(2)[1])
    short = _episode(3, steps=5)
    idle = (_episode(4)[0], np.zeros((T, A)))
    big = _episode(5); big[1][3, 0] = 3.0
    p = _write(tmp_path / "d.hdf5", [good, (nan_s, nan_a), mism, short, (good[0].copy(), good[1].copy()), idle, big])
    v = _by_id(G.qa_dataset(p))
    assert v["demo_1"].passed and v["demo_1"].reasons == []
    assert any("non-finite" in r for r in v["demo_2"].reasons)
    assert any("length mismatch" in r for r in v["demo_3"].reasons)
    assert any("too short" in r for r in v["demo_4"].reasons)
    assert any("duplicate" in r for r in v["demo_5"].reasons)
    assert any("too idle" in r for r in v["demo_6"].reasons)
    assert any("out of range" in r for r in v["demo_7"].reasons)
    assert sum(x.passed for x in v.values()) == 1


def test_duplicate_across_files(tmp_path):
    ep = _episode(7)
    seen = set()
    a = G.qa_dataset(_write(tmp_path / "a.hdf5", [ep]), seen_hashes=seen)
    b = G.qa_dataset(_write(tmp_path / "b.hdf5", [ep]), seen_hashes=seen)
    assert a[0].passed and not b[0].passed


def test_missing_dataset(tmp_path):
    p = tmp_path / "m.hdf5"
    with h5py.File(p, "w") as f:
        f.create_group("data/demo_1").create_dataset("actions", data=np.zeros((T, A)))
    v = G.qa_dataset(p)[0]
    assert not v.passed and "missing" in v.reasons[0]


def test_env_kwargs_both_formats(tmp_path):
    p = _write(tmp_path / "e.hdf5", [_episode(0)])
    with h5py.File(p, "r") as f:
        kw = G.env_kwargs_from_file(f)
    assert kw["env_name"] == "CoffeeSetupMug" and kw["use_camera_obs"] is False
    with h5py.File(p, "a") as f:
        f["data"].attrs["env_args"] = json.dumps({"env_name": "X", "env_kwargs": {"robots": "Panda"}})
    with h5py.File(p, "r") as f:
        assert G.env_kwargs_from_file(f)["env_name"] == "X"


def test_ledger(tmp_path):
    p = _write(tmp_path / "d.hdf5", [_episode(0), _episode(0, steps=5)])
    vs = G.qa_dataset(p)
    ledger = tmp_path / "ledger.json"
    added = G.record_credits(ledger, vs, "p1", "human", 0.25, dataset_id="order-1")
    assert [e["credit"] for e in added] == [0.25, 0.0]
    assert [e["verdict"] for e in added] == ["pass", "fail"]
    assert G.record_credits(ledger, vs, "p1", "human", 0.25, dataset_id="order-1") == []  # idempotent
    G.record_credits(ledger, vs[:1], "bot", "ai", 0.1, dataset_id="order-2")
    assert G.player_balance(ledger, "p1") == 0.25
    assert G.player_balance(ledger, "bot") == 0.1
    assert len(json.loads(ledger.read_text())) == 3
    with pytest.raises(ValueError):
        G.record_credits(ledger, vs, "p1", "robot", 1.0)


# ---- replay: needs robocasa + kitchen assets ----------------------------------------------

def _sim_or_skip():
    try:
        G._import_sim()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"robocasa not importable: {e}")


def test_replay_real_episode(tmp_path):
    _sim_or_skip()
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    config = {"env_name": "CoffeeSetupMug", "robots": "PandaOmron",
              "controller_configs": load_composite_controller_config(controller=None, robot="PandaOmron"),
              "layout_ids": 1, "style_ids": 1}
    try:
        env = robosuite.make(**config, has_renderer=False, has_offscreen_renderer=False,
                             use_camera_obs=False, ignore_done=True, control_freq=20, seed=0)
    except Exception as e:
        pytest.skip(f"could not build RoboCasa env (assets?): {e}")
    # Mimic DataCollectionWrapper: reset, snapshot xml, reload, then record state before each action.
    env.reset()
    xml = env.sim.model.get_xml()
    ep_meta = env.get_ep_meta()
    env.set_ep_meta(ep_meta)
    env.reset_from_xml_string(xml)
    rng = np.random.default_rng(0)
    states, actions = [], []
    for _ in range(30):
        states.append(np.array(env.sim.get_state().flatten()))
        a = rng.uniform(-0.3, 0.3, size=env.action_dim)
        actions.append(a)
        env.step(a)
    env.close()

    p = tmp_path / "real.hdf5"
    with h5py.File(p, "w") as f:
        d = f.create_group("data")
        d.attrs["env"] = "CoffeeSetupMug"
        d.attrs["env_info"] = json.dumps(config)
        for i, acts in enumerate([np.array(actions), np.array(actions) * -1.0], 1):
            g = d.create_group(f"demo_{i}")
            g.attrs["model_file"] = xml
            g.attrs["ep_meta"] = json.dumps(ep_meta)
            g.create_dataset("states", data=np.array(states))
            g.create_dataset("actions", data=acts)

    cfg = G.QAConfig(min_steps=10, require_success=False)
    v = _by_id(G.qa_dataset(p, cfg=cfg, replay=True))
    print("replay metrics:", v["demo_1"].metrics, v["demo_2"].reasons)
    assert v["demo_1"].replay_checked and v["demo_1"].passed, v["demo_1"].reasons
    assert v["demo_2"].replay_checked and not v["demo_2"].passed  # tampered actions diverge
    assert any("diverged" in r for r in v["demo_2"].reasons)


def test_ensure_robomimic_meta(tmp_path):
    p = _write(tmp_path / "r.hdf5", [_episode(0), _episode(1, steps=25)])
    G.ensure_robomimic_meta(p)
    with h5py.File(p, "r") as f:
        meta = json.loads(f["data"].attrs["env_args"])
        assert meta["env_name"] == "CoffeeSetupMug" and meta["env_kwargs"]["translucent_robot"] is False
        assert f["data"].attrs["total"] == T + 25
        assert f["data/demo_2"].attrs["num_samples"] == 25
