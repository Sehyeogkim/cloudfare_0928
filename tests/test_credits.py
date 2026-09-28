import json

import h5py
import numpy as np
import pytest

from agriphilo.credits import InsufficientCredits, Ledger


def test_topup_idempotent(tmp_path):
    L = Ledger(tmp_path / "l.json")
    L.topup("c", 100, "evt_1")
    L.topup("c", 100, "evt_1")  # webhook retry
    assert L.balance("wallet:c") == 100
    assert len(L.entries) == 1
    with pytest.raises(ValueError):
        L.topup("c", 0, "evt_2")


def test_hold_capture_release(tmp_path):
    L = Ledger(tmp_path / "l.json")
    L.topup("c", 50, "evt")
    L.hold("c", "g1", 40)
    assert L.balance("wallet:c") == 10 and L.balance("hold:g1") == 40
    L.capture("g1", "ep1", 4, "alice", "human", 1.5)
    L.capture("g1", "ep1", 4, "alice", "human", 1.5)  # QA re-run: no double charge
    L.capture("g1", "ep2", 4, "bot", "ai", 1.5)
    assert L.balance("hold:g1") == 32
    assert L.balance("payable:alice") == 1.5 and L.balance("ai_budget:bot") == 1.5
    assert L.balance("platform:fee") == 5
    L.token_cost("g1", "ep2", "bot", 0.4, {"input_tokens": 1000})
    L.token_cost("g1", "ep2", "bot", 0.4, {"input_tokens": 1000})
    assert L.balance("ai_budget:bot") == pytest.approx(1.1)
    L.release("c", "g1")
    L.release("c", "g1")
    assert L.balance("hold:g1") == 0 and L.balance("wallet:c") == 42
    # every entry balances; total money in the system equals what came from stripe
    assert all(abs(sum(e["deltas"].values())) < 1e-9 for e in L.entries)
    assert sum(L.balances().values()) == pytest.approx(0)


def test_insufficient(tmp_path):
    L = Ledger(tmp_path / "l.json")
    L.topup("c", 10, "evt")
    with pytest.raises(InsufficientCredits):
        L.hold("c", "g1", 11)
    L.hold("c", "g1", 8)
    L.capture("g1", "e1", 4, "p", "human", 1)
    L.capture("g1", "e2", 4, "p", "human", 1)
    with pytest.raises(InsufficientCredits):
        L.capture("g1", "e3", 4, "p", "human", 1)
    with pytest.raises(InsufficientCredits):
        L.refund("c", 5, "r1")
    L.refund("c", 2, "r1")
    assert L.balance("wallet:c") == 0


def test_bad_inputs(tmp_path):
    L = Ledger(tmp_path / "l.json")
    L.topup("c", 10, "evt"); L.hold("c", "g", 10)
    with pytest.raises(ValueError):
        L.capture("g", "e", 4, "p", "robot", 1)
    with pytest.raises(ValueError):
        L.capture("g", "e", 4, "p", "human", 5)


def test_persistence(tmp_path):
    p = tmp_path / "l.json"
    L = Ledger(p); L.topup("c", 7, "evt")
    assert Ledger(p).balance("wallet:c") == 7


def _episode_file(path, player_kind, seed):
    r = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["env"] = "CoffeeSetupMug"
        d.attrs["env_info"] = json.dumps({"env_name": "CoffeeSetupMug", "robots": "PandaOmron"})
        g = d.create_group("demo_1")
        g.attrs["model_file"] = "<mujoco/>"
        g.attrs["ep_meta"] = json.dumps({"lang": "mug"})
        g.attrs["player_id"] = f"{player_kind}-1"
        g.attrs["player_kind"] = player_kind
        g.create_dataset("states", data=r.normal(size=(40, 12)))
        g.create_dataset("actions", data=r.uniform(-0.5, 0.5, size=(40, 7)))


def test_game_flow_structural(tmp_path, monkeypatch):
    """Order -> spec -> approve (hold) -> episodes QA'd -> capture / token_cost -> close (release)."""
    from agriphilo import games as GM
    from agriphilo.game_spec import GameSpec
    monkeypatch.setattr(GM, "RUNS", tmp_path)
    monkeypatch.setattr(GM, "SETTLE_S", 0)
    monkeypatch.setenv("AGRIPHILO_SKIP_REPLAY", "1")
    store = GM.GameStore(bb=None, ledger=Ledger(tmp_path / "ledger.json"))
    g = GM.Game(store, {"id": "abcdef012345", "created_at": "t", "order_text": "coffee", "customer_id": GM.CUSTOMER,
                        "status": "awaiting_approval", "summary": "", "error": None, "files": {}, "episodes": {},
                        "seen_hashes": [], "price_per_episode": 4.0, "hold_amount": 8.0,
                        "spec": GameSpec.model_validate({
                            "task": "CoffeeSetupMug", "layout_ids": [1], "style_ids": [1], "robot": "PandaOmron",
                            "cameras": [{"name": "robot0_eye_in_hand", "width": 128, "height": 128, "fps": 20}],
                            "control_hz": 20, "max_steps": 900, "episodes_requested": 2,
                            "success_rule": {"time_limit_s": 45}, "credit_per_pass": 1.5, "base_seed": 1}).model_dump()})
    store.games[g.id] = g
    with pytest.raises(RuntimeError):
        g.approve()  # wallet empty
    store.ledger.topup(GM.CUSTOMER, 20, "evt")
    g.approve()
    _episode_file(g.episode_dir / "h1.hdf5", "human", 0)
    _episode_file(g.episode_dir / "a1.hdf5", "ai", 1)
    (g.episode_dir / "a1.json").write_text(json.dumps({"plan": ["reach mug", "grasp"], "reasoning": "mug left",
                                                        "usage": {"input_tokens": 1000, "output_tokens": 100, "cost_usd": 0.05}}))
    _episode_file(g.episode_dir / "z_dup.hdf5", "human", 0)  # same data as h1, QA'd after it -> duplicate
    g.scan()
    import time
    deadline = time.time() + 20
    while any(f["status"] in ("queued", "qa_running") for f in g.d["files"].values()):
        assert time.time() < deadline, g.d["files"]
        time.sleep(0.05)
    assert all(f["status"] == "done" for f in g.d["files"].values()), g.d["files"]
    eps = {e["episode_id"]: e for e in g.to_dict()["episodes"]}
    assert eps["h1/demo_1"]["passed"] and eps["a1/demo_1"]["passed"]
    assert not eps["z_dup/demo_1"]["passed"]
    assert eps["a1/demo_1"]["plan"] == ["reach mug", "grasp"]
    L = store.ledger
    assert L.balance("payable:human-1") == 1.5 and L.balance("ai_budget:ai-1") == pytest.approx(1.45)
    assert L.balance("hold:" + g.id) == 0
    g.close()
    assert L.balance("wallet:" + GM.CUSTOMER) == 12
