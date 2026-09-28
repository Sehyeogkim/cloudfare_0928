"""Prepaid credit ledger (plan_game_platform §6). TEST MODE: no Stripe call is made.

Double-entry and append-only: every entry moves credits between accounts and its deltas sum to zero.
A balance is the sum of an account's deltas; nothing is edited in place. 1 credit = 1 USD (placeholder).

Accounts
  stripe:test            external money source/sink for TEST MODE top-ups and refunds
  wallet:<customer>      customer's available credits
  hold:<game>            credits reserved for one game order
  payable:<player>       owed to a human player (paid out later, e.g. Stripe Connect)
  ai_budget:<player>     AI player's share, spent on LLM tokens
  llm:vendor             token spend leaves the platform here
  platform:fee           platform revenue

Every entry carries an idempotency_key (Stripe event id, "<game>:<episode>" ...). Re-posting the same key
returns the existing entry and changes nothing, so webhook retries and QA re-runs cannot double-count.
"""
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

KINDS = ("topup", "hold", "capture", "token_cost", "release", "refund", "fee", "reward", "purchase")


class InsufficientCredits(RuntimeError):
    pass


def _r(x: float) -> float:
    return round(float(x), 6)


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.entries: List[dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    # ---- reads -------------------------------------------------------------------------------
    def balance(self, account: str) -> float:
        with self._lock:
            return _r(sum(e["deltas"].get(account, 0.0) for e in self.entries))

    def find(self, key: str) -> Optional[dict]:
        with self._lock:
            return next((e for e in self.entries if e["idempotency_key"] == key), None)

    def for_ref(self, ref: str) -> List[dict]:
        with self._lock:
            return [e for e in self.entries if e.get("ref") == ref]

    def balances(self, prefix: str = "") -> Dict[str, float]:
        with self._lock:
            out: Dict[str, float] = {}
            for e in self.entries:
                for a, d in e["deltas"].items():
                    if a.startswith(prefix):
                        out[a] = _r(out.get(a, 0.0) + d)
            return out

    # ---- writes ------------------------------------------------------------------------------
    def _post(self, kind: str, key: str, deltas: Dict[str, float], ref: str = "", meta: Optional[dict] = None) -> dict:
        assert kind in KINDS
        deltas = {a: _r(d) for a, d in deltas.items() if _r(d) != 0}
        if abs(sum(deltas.values())) > 1e-6:
            raise ValueError(f"unbalanced entry: {deltas}")
        e = {"id": uuid.uuid4().hex[:12], "kind": kind, "idempotency_key": key, "ref": ref, "deltas": deltas,
             "meta": meta or {}, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.entries.append(e)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False))
        tmp.replace(self.path)
        return e

    def topup(self, customer: str, amount: float, key: str) -> dict:
        """TEST MODE Stripe Checkout: `key` stands in for the checkout.session.completed event id."""
        if amount <= 0:
            raise ValueError("top-up amount must be positive")
        with self._lock:
            return self.find(key) or self._post("topup", key, {"stripe:test": -amount, f"wallet:{customer}": amount},
                                                meta={"test_mode": True})

    def hold(self, customer: str, game_id: str, amount: float) -> dict:
        key = f"hold:{game_id}"
        with self._lock:
            if (e := self.find(key)):
                return e
            avail = self.balance(f"wallet:{customer}")
            if avail + 1e-9 < amount:
                raise InsufficientCredits(f"wallet has {avail:.2f} credits, order needs {amount:.2f}")
            return self._post("hold", key, {f"wallet:{customer}": -amount, f"hold:{game_id}": amount}, ref=game_id)

    def capture(self, game_id: str, episode_id: str, price: float, player_id: str, player_kind: str,
                player_share: float) -> dict:
        """Charge one QA-passed episode: hold -> player (payable or AI budget) + platform fee."""
        if player_kind not in ("human", "ai"):
            raise ValueError("player_kind must be 'human' or 'ai'")
        if not 0 <= player_share <= price:
            raise ValueError("player_share must be within the episode price")
        key = f"capture:{game_id}:{episode_id}"
        with self._lock:
            if (e := self.find(key)):
                return e
            left = self.balance(f"hold:{game_id}")
            if left + 1e-9 < price:
                raise InsufficientCredits(f"hold for {game_id} has {left:.2f} credits left, episode costs {price:.2f}")
            to = f"payable:{player_id}" if player_kind == "human" else f"ai_budget:{player_id}"
            return self._post("capture", key, {f"hold:{game_id}": -price, to: player_share,
                                               "platform:fee": price - player_share},
                              ref=game_id, meta={"episode_id": episode_id, "player_id": player_id,
                                                 "player_kind": player_kind})

    def token_cost(self, game_id: str, episode_id: str, player_id: str, cost: float, usage: dict) -> dict:
        """Record LLM spend of an AI episode (pass or fail). It may drive ai_budget negative: that is the loss signal."""
        key = f"token_cost:{game_id}:{episode_id}"
        with self._lock:
            return self.find(key) or self._post("token_cost", key, {f"ai_budget:{player_id}": -cost, "llm:vendor": cost},
                                                ref=game_id, meta={"episode_id": episode_id, "usage": usage})

    def release(self, customer: str, game_id: str) -> dict:
        key = f"release:{game_id}"
        with self._lock:
            if (e := self.find(key)):
                return e
            left = self.balance(f"hold:{game_id}")
            return self._post("release", key, {f"hold:{game_id}": -left, f"wallet:{customer}": left}, ref=game_id)

    def _spend(self, kind: str, customer: str, amount: float, key: str, ref: str, meta: Optional[dict]) -> dict:
        with self._lock:
            if (e := self.find(key)):
                return e
            avail = self.balance(f"wallet:{customer}")
            if avail + 1e-9 < amount:
                raise InsufficientCredits(f"wallet has {avail:.2f} credits, this needs {amount:.2f}")
            return self._post(kind, key, {f"wallet:{customer}": -amount, "platform:revenue": amount}, ref=ref, meta=meta)

    def fee(self, customer: str, order_id: str, amount: float) -> dict:
        """Environment fee, charged when the requester approves the spec."""
        return self._spend("fee", customer, amount, f"fee:{order_id}", order_id, {"item": "environment"})

    def purchase(self, customer: str, order_id: str, key: str, amount: float, episode_ids: List[str]) -> dict:
        """Requester buys QA-passed episodes from their marketplace listing."""
        return self._spend("purchase", customer, amount, key, order_id, {"episodes": episode_ids})

    def reward(self, order_id: str, episode_id: str, player_id: str, amount: float) -> dict:
        """Pay a player for a QA-passed episode (from the platform's reward budget)."""
        key = f"reward:{order_id}:{episode_id}"
        with self._lock:
            return self.find(key) or self._post("reward", key, {"platform:rewards": -amount, f"payable:{player_id}": amount},
                                                ref=order_id, meta={"episode_id": episode_id, "player_id": player_id})

    def refund(self, customer: str, amount: float, key: str) -> dict:
        with self._lock:
            if (e := self.find(key)):
                return e
            if self.balance(f"wallet:{customer}") + 1e-9 < amount:
                raise InsufficientCredits("refund exceeds available wallet credits")
            return self._post("refund", key, {f"wallet:{customer}": -amount, "stripe:test": amount},
                              meta={"test_mode": True})
