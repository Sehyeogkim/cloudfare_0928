"""WEMINE web app: the orders console plus a JSON API over the order pipeline.

    python3 -m agriphilo.web            # http://localhost:8000, real Brainbase agents
    python3 -m agriphilo.web --demo     # scripted agents, no API key or credits

The Brainbase key stays in this server process (.env); the browser never sees it.
"""
import argparse
import json
import os
import uuid
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agents.payment import PRICING
from .credits import InsufficientCredits
from . import stripe_checkout
from .games import GameStore
from .pipeline import DELIVERY_FILES, OrderStore
from .sim.worker import default_worker

STATIC = Path(__file__).parent / "static"
EXAMPLES = Path(__file__).parent.parent / "examples"
PAGES = {
    "/": ("login.html", "text/html"),
    "/requester": ("index.html", "text/html"),
    "/player": ("player.html", "text/html"),
    "/static/app.js": ("app.js", "application/javascript"),
    "/static/vendor/three.module.js": ("vendor/three.module.js", "application/javascript"),
}
PLAYER_ROUTE = re.compile(r"/api/players/([a-z0-9._-]{1,120})")
GAME_ROUTE = re.compile(r"/api/games/([0-9a-f]{12})(?:/(approve|close))?")
ORDER_ROUTE = re.compile(r"/api/orders/([0-9a-f-]{36})(?:/(answer|approve|continue|pay|checkout|confirm|retry|files/[\w.]+))?")


class Handler(BaseHTTPRequestHandler):
    store: OrderStore
    games: GameStore
    demo = False

    def log_message(self, fmt, *args):
        if self.command != "GET" or not self.path.startswith("/api/"):
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _send(self, status: int, data: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, body) -> None:
        self._send(status, json.dumps(body, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _file(self, path: Path, ctype: str, download: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _order(self):
        """(order, action) for /api/orders/<id>[/action]; order is False for an unknown id."""
        m = ORDER_ROUTE.fullmatch(self.path.split("?")[0])
        if not m:
            return None, None
        try:
            return self.store.get(m.group(1)), m.group(2)
        except KeyError:
            return False, None

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in PAGES:
            name, ctype = PAGES[path]
            return self._send(200, (STATIC / name).read_bytes(), f"{ctype}; charset=utf-8")
        if path == "/api/config":
            return self._json(200, {"demo": self.demo, "worker": self.store.worker.name, "pricing_note": PRICING["note"],
                                    "stripe": stripe_checkout.enabled()})
        if path == "/api/examples":
            return self._json(200, {p.stem: p.read_text() for p in sorted(EXAMPLES.glob("cafe*_order.txt"))})
        if path == "/api/orders":
            return self._json(200, {"items": [o.summary() for o in self.store.all()]})
        if path == "/api/marketplace":
            return self._json(200, {"items": self.store.marketplace()})
        m = PLAYER_ROUTE.fullmatch(path)
        if m:
            return self._json(200, self.store.player(m.group(1)))
        if path == "/api/wallet":
            return self._json(200, self.games.wallet())
        if path == "/api/games":
            return self._json(200, {"items": [g.summary() for g in self.games.all()],
                                    "game_server_url": os.environ.get("AGRIPHILO_GAME_SERVER_URL", "")})
        m = GAME_ROUTE.fullmatch(path)
        if m and not m.group(2):
            try:
                return self._json(200, self.games.get(m.group(1)).to_dict())
            except KeyError:
                return self._json(404, {"error": "unknown game"})
        order, action = self._order()
        if order is False:
            return self._json(404, {"error": "unknown order"})
        if order and not action:
            return self._json(200, order.to_dict())
        if order and action.startswith("files/"):
            name = action.split("/", 1)[1]
            fp = order.file_path(name)
            if not fp or not fp.exists():
                return self._json(403 if name in DELIVERY_FILES else 404, {"error": "file not available"})
            ctype = "image/png" if name.endswith(".png") else DELIVERY_FILES.get(name, "application/octet-stream")
            return self._file(fp, ctype, download=not name.endswith(".png"))
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": "invalid JSON body"})
        if self.path == "/api/wallet/checkout":
            try:
                credits = float(body.get("credits", 0))
                if not 1 <= credits <= 100000:
                    raise ValueError("credits must be between 1 and 100000")
                origin = f"http://{self.headers.get('Host', 'localhost:8000')}"
                s = stripe_checkout.create_topup_session(
                    self.games.wallet()["customer_id"], credits,
                    success_url=f"{origin}/requester?topup=success&session_id={{CHECKOUT_SESSION_ID}}",
                    cancel_url=f"{origin}/requester?topup=cancel")
            except (TypeError, ValueError) as err:
                return self._json(400, {"error": str(err)})
            except RuntimeError as err:
                return self._json(502, {"error": str(err)})
            return self._json(200, {"url": s["url"]})
        if self.path == "/api/wallet/confirm":
            try:
                s = stripe_checkout.retrieve_session(str(body.get("session_id", "")))
                meta = s.get("metadata") or {}
                customer = self.games.wallet()["customer_id"]
                if meta.get("kind") != "credit_topup" or meta.get("customer") != customer:
                    raise ValueError("not a credit top-up for this wallet")
                if s.get("payment_status") != "paid":
                    raise ValueError(f"Stripe reports payment_status={s.get('payment_status')}")
                credits = float(meta["credits"])
                if s.get("amount_total") != int(round(credits * 100)):
                    raise ValueError("paid amount does not match the credits")
                # keyed by the Checkout Session id, so a reload of the return page never credits twice
                e = self.games.ledger.topup(customer, credits, f"stripe:{s['id']}")
            except ValueError as err:
                return self._json(409, {"error": str(err)})
            except RuntimeError as err:
                return self._json(502, {"error": str(err)})
            return self._json(200, {"entry": e, "wallet": self.games.wallet()})
        if self.path == "/api/wallet/topup":
            try:
                amount = float(body.get("amount", 0))
                e = self.games.ledger.topup(self.games.wallet()["customer_id"], amount,
                                            body.get("idempotency_key") or f"test_evt_{uuid.uuid4().hex}")
            except (TypeError, ValueError) as err:
                return self._json(400, {"error": str(err)})
            return self._json(200, {"entry": e, "wallet": self.games.wallet()})
        if self.path == "/api/games":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json(400, {"error": "order text is empty"})
            return self._json(201, self.games.create(text).to_dict())
        m = GAME_ROUTE.fullmatch(self.path)
        if m and m.group(2):
            try:
                g = self.games.get(m.group(1))
            except KeyError:
                return self._json(404, {"error": "unknown game"})
            try:
                g.approve() if m.group(2) == "approve" else g.close()
            except (RuntimeError, InsufficientCredits) as err:
                return self._json(409, {"error": str(err)})
            return self._json(200, g.to_dict())
        if self.path == "/api/orders":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json(400, {"error": "order text is empty"})
            try:
                order = self.store.create(text, bool(body.get("auto")))
            except Exception as e:
                return self._json(502, {"error": f"could not start the Customer Agent: {e}"})
            return self._json(201, order.to_dict())
        order, action = self._order()
        if order is False:
            return self._json(404, {"error": "unknown order"})
        if not order or action not in ("answer", "approve", "continue", "pay", "checkout", "confirm", "retry"):
            return self._json(404, {"error": "not found"})
        try:
            if action == "answer":
                order.answer(body.get("text", ""))
            elif action == "approve":
                order.approve()
            elif action == "retry":
                order.retry()
            elif action == "continue":
                order.continue_play()
            elif action == "checkout":
                origin = f"http://{self.headers.get('Host', 'localhost:8000')}"
                return self._json(200, {"url": order.start_checkout(origin)})
            elif action == "confirm":
                order.confirm_checkout(str(body.get("session_id", "")))
            else:
                order.pay()
        except RuntimeError as e:
            return self._json(409, {"error": str(e)})
        except Exception as e:
            return self._json(502, {"error": str(e)})
        self._json(200, order.to_dict())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--demo", action="store_true", help="scripted agents; no API key or credits")
    args = p.parse_args()
    if args.demo:
        from . import pipeline
        from .demo import DemoBrainbase

        pipeline.ROOT = Path("out/demo_orders")  # keep demo orders out of the real order list
        bb = DemoBrainbase()
    else:
        from .brainbase import Brainbase

        bb = Brainbase()
    Handler.games = GameStore(bb)
    Handler.store = OrderStore(bb, default_worker(), ledger=Handler.games.ledger)
    Handler.demo = args.demo
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    mode = "demo agents" if args.demo else "Brainbase agents"
    print(f"WEMINE on http://localhost:{args.port} ({mode}, simulator: {Handler.store.worker.name})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
