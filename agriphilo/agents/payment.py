"""Payment Agent: prices the delivered dataset from the rate card and explains the quote.

Prices come only from pricing.json; the agent chooses line items and quantities. Arithmetic is
re-checked in code. Checkout is test mode: no real charge is made.
"""
import json
from pathlib import Path
from typing import Any, Dict

from ..models import Quote
from .base import JSON_RULES, JsonTask, agent_spec

PRICING = json.loads((Path(__file__).parent.parent / "pricing.json").read_text())

INSTRUCTIONS = f"""You are the Payment Agent of WEMINE, an on-demand robot simulation data service.
You receive the order, what was delivered (QA-passed episodes, GPU hours, QA verdict) and the rate card.
Write the pilot quote.

Rules:
- Use only SKUs and unit prices from the rate card. Quantities must come from the delivery facts
  (e.g. episode quantity = QA-passed episodes delivered; gpu_hour quantity = the GPU hours given, rounded up to 0.1).
- Include scene_setup and qa_report for every order; include failure_repro only for failure-mode orders.
- You may add one pilot_discount line (negative amount, at most max_discount_pct of the subtotal before
  discount) and must justify it in notes. Its unit_price_usd equals its amount_usd and quantity is 1.
- amount_usd = quantity * unit_price_usd, rounded to cents. subtotal_usd = sum of non-discount lines.
  total_usd = sum of all lines. valid_days 14-30.
- notes: short, in the language of the customer's order. Say the rates are pilot placeholders.

{JSON_RULES}
Shape: {{"status": "ready", "quote": <Quote>, "summary": "<one sentence>"}}

Quote JSON Schema:
{json.dumps(Quote.model_json_schema(), ensure_ascii=False)}
"""

AGENT = agent_spec("Payment Agent", INSTRUCTIONS)


def check_quote(q: Quote) -> None:
    items = PRICING["items"]
    subtotal = 0.0
    for li in q.line_items:
        if li.sku not in items:
            raise ValueError(f"unknown sku {li.sku}")
        if li.sku == "pilot_discount":
            if li.amount_usd > 0:
                raise ValueError("pilot_discount must be negative")
            continue
        if abs(li.unit_price_usd - items[li.sku]["unit_price_usd"]) > 1e-6:
            raise ValueError(f"{li.sku} unit price must be {items[li.sku]['unit_price_usd']}")
        if abs(li.amount_usd - round(li.quantity * li.unit_price_usd, 2)) > 0.01:
            raise ValueError(f"{li.sku} amount must equal quantity * unit_price")
        subtotal += li.amount_usd
    discount = sum(li.amount_usd for li in q.line_items if li.sku == "pilot_discount")
    if -discount > subtotal * PRICING["max_discount_pct"] / 100 + 0.01:
        raise ValueError(f"discount exceeds {PRICING['max_discount_pct']}% of subtotal")
    if abs(q.subtotal_usd - subtotal) > 0.01:
        raise ValueError(f"subtotal_usd must be {subtotal:.2f}")
    if abs(q.total_usd - (subtotal + discount)) > 0.01:
        raise ValueError(f"total_usd must be {subtotal + discount:.2f}")


class PaymentTask(JsonTask):
    agent = AGENT
    key = "quote"
    model = Quote

    def __init__(self, facts: Dict[str, Any], bb):
        super().__init__(json.dumps({"delivery": facts, "rate_card": PRICING}, ensure_ascii=False), bb)

    def check(self, q: Quote) -> None:
        check_quote(q)
