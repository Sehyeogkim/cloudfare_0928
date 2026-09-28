![WEMINE connects two roles: a requester defines a robot task, WEMINE builds and publishes the game, a separate player performs it, and validated data returns to the requester](docs/assets/wemine-overview.png)

# WEMINE

**Play a robot task. Capture useful robot data.** WEMINE is a robot-data marketplace. A **requester** describes the robot, task, environment and data they need; agents turn it into a MuJoCo game and publish it to the **marketplace**. Human **players** play the game in the browser; every episode they submit is checked by a **QA Agent**. The requester then **buys** the QA-passed episodes with credits and downloads them.

In this hackathon pilot the game is a café: a mobile Franka Panda robot (PandaOmron) picks up a cup of coffee and puts it on a tray (`CafeServeCoffee`, MuJoCo + RoboCasa).

> **Judges — quick start:** run the two servers below, then open **http://127.0.0.1:8000**, sign in with the demo sign-in and pick a role. `--demo` needs no API keys.

## How it works

```
Requester                         Agents (Brainbase · Claude)                 Players
─────────                         ───────────────────────────                 ───────
New order (robot, task,  ──►  Customer Agent → Spec
environment, data, count)          │ approve: pay environment fee (credits)
                                    ▼
                              Orchestrator Agent → MuJoCo production plan
                              Environment Agent  → café scene (rendered)
                                    │
                                    ▼
                              ┌──────────── Marketplace ────────────┐
                              │ game listing  ◄── play in browser ──┤  Player
                              │               ◄── submit episode ───┤
                              │ Episode QA Agent: pass/fail, 1–5,   │
                              │ one-line reason → reward the player │
                              └───────────────┬─────────────────────┘
Buy N QA-passed episodes  ◄───────────────────┘
(credits) → download HDF5
Top up credits (Stripe test mode)
```

| Step | Who | What you see |
| --- | --- | --- |
| 1. Order | Requester | Plain-language request; the **Customer Agent** turns it into a checked spec (robot, task steps, environment, data schema, episode count). |
| 2. Approve | Requester | **Approve spec & pay the environment fee** (100 credits). |
| 3. Plan | Orchestrator Agent | A MuJoCo/RoboCasa production plan: who does what, scene build steps, task-stage checks, QA criteria. |
| 4. Scene | Environment Agent | Layouts, styles, cameras; the scene is built in MuJoCo and a render is shown. |
| 5. Marketplace | Players + QA Agent | The game is published. Players play and submit; the **Episode QA Agent** judges each episode (pass/fail, quality 1–5, reason). Passed episodes earn the player 2.5 credits. |
| 6. Buy | Requester | Pick how many QA-passed episodes to buy (4 credits each) and download `dataset.hdf5`, `manifest.json`, `qa_report.json`. |
| Billing | Requester | Wallet balance, ledger, and **credit top-up through Stripe Checkout (test mode)**. |

Credits are placeholder hackathon rates (1 credit = 1 USD, `agriphilo/pricing.json`). Stripe runs in **test mode only** (live keys are refused), so no real charge is ever made.

## Run it locally

**Prerequisites:** macOS or Linux, Python 3.11, Git, a desktop browser, ~25 GB free disk (MuJoCo kitchen assets).

### 1. Console (requester + player pages) — small, a few seconds

```bash
git clone https://github.com/Sehyeogkim/cloudfare_0928.git
cd cloudfare_0928
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. One-time game setup (MuJoCo + RoboCasa) — large, ~10 GB download

The simulator packages and assets are not in Git. From the repository root:

```bash
python3.11 -m venv sim/.venv
git clone https://github.com/ARISE-Initiative/robosuite.git sim/third_party/robosuite
git -C sim/third_party/robosuite checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git clone https://github.com/robocasa/robocasa.git sim/third_party/robocasa
git -C sim/third_party/robocasa checkout 456174f62b89b8fca99eaaf33949c29fec9cfc2a
sim/.venv/bin/python -m pip install -e sim/third_party/robosuite -e sim/third_party/robocasa
sim/.venv/bin/python -m pip install -r sim/requirements.txt
(cd sim/third_party/robocasa && ../../.venv/bin/python -m robocasa.scripts.setup_macros)
(cd sim/third_party/robocasa && ../../.venv/bin/python -m robocasa.scripts.download_kitchen_assets)
```

Check it: `sim/.venv/bin/python sim/scripts/render_scene.py --task CafeServeCoffee --layout 9 --style 3 --out sim/renders/check` writes café renders to `sim/renders/check/`.

### 3. Start both servers

**Terminal 1 — MuJoCo game server (port 8010):**

```bash
sim/.venv/bin/python sim/web/server.py
```

**Terminal 2 — WEMINE console (port 8000):**

```bash
source .venv/bin/activate
AGRIPHILO_GAME_SERVER_URL=http://127.0.0.1:8010/ python -m agriphilo.web --demo
```

`--demo` uses scripted agents (marked `[Demo]`), no API keys. Stop either server with `Ctrl+C`.

| URL | Page |
| --- | --- |
| http://127.0.0.1:8000 | Sign-in and role choice |
| http://127.0.0.1:8000/requester | Requester console: orders, marketplace, billing |
| http://127.0.0.1:8000/player | Player: marketplace games, my episodes and earnings |
| http://127.0.0.1:8010/?role=player | The café game directly (without an order) |

### 4. Demo walkthrough (about 3 minutes)

1. **Requester** → Billing → top up credits (without a Stripe key this is an instant test top-up).
2. **New order** → pick *Example: cafe coffee serving* → wait for the spec → **Approve spec & pay 100 credits**.
3. Watch **Plan** and **Scene** fill in; the order goes **Live in marketplace**.
4. **Player** (second tab, `/player`) → **Play →** on the café game → click the game view, drive the arm, grab the cup with `Space`, put it on the tray → **Submit episode**.
5. Back in the requester's order → **Marketplace** tab: the episode appears with the QA Agent's verdict, quality and reason (a failed attempt shows why it failed).
6. **Buy** QA-passed episodes → **Files** tab → download the dataset.

| Control | Action |
| --- | --- |
| `W` `S` / `A` `D` / `R` `F` | Arm forward/back, left/right, up/down |
| `Q` `E` / `Z` `C` | Rotate the wrist |
| `Space` | Open / close the gripper |
| `Shift` + move | Fine control |
| `I` `K` / `J` `L` / `U` `O` | Move / rotate the mobile base |

Controls use physical key positions, so they work with any keyboard layout or input language.

### Real agents and Stripe (optional)

Create a `.env` in the repository root (never commit it):

```bash
BRAINBASE_API_KEY=...          # real Brainbase agents (Claude on a Cloudflare sandbox)
STRIPE_API_KEY=sk_test_...     # Stripe test-mode secret key; enables Stripe Checkout for top-ups
```

Then start the console without `--demo`:

```bash
AGRIPHILO_GAME_SERVER_URL=http://127.0.0.1:8010/ python -m agriphilo.web
```

With a Stripe key, **Billing → top up** opens Stripe's hosted checkout. Pay with the test card `4242 4242 4242 4242`, any future date and any CVC; the console confirms the payment with Stripe before crediting the wallet.

On Linux, start the game server with `MUJOCO_GL=egl sim/.venv/bin/python sim/web/server.py` (macOS uses `cgl` by default).

## How it is built

| Layer | Implementation |
| --- | --- |
| Agents | Brainbase-hosted Claude agents (Customer, Orchestrator, Environment, Episode QA) with Pydantic-validated JSON; scripted stand-ins in `--demo` |
| Game | MuJoCo 3 + RoboCasa/robosuite café task; the browser gets JPEG frames over WebSocket and sends key states back; one sim shared by the player and live viewers |
| Episodes | `collect_demos`-style HDF5: simulator states, actions, model XML, episode metadata, camera preview frames |
| QA | Code measures each episode (stages completed, duration, idle ratio, action range, gripper use); the Episode QA Agent gives the verdict in plain language |
| Credits | Double-entry ledger (`agriphilo/credits.py`): top-up, environment fee, player reward, purchase |
| Payments | Stripe Checkout REST API, test mode only (`agriphilo/stripe_checkout.py`) |

## Repository map

| Path | Purpose |
| --- | --- |
| `agriphilo/web.py`, `agriphilo/static/` | HTTP server and API; `login.html`, requester console `index.html`, player `player.html` |
| `agriphilo/pipeline.py` | Order state machine: spec → plan → scene → marketplace, episode intake, purchases |
| `agriphilo/agents/` | Customer, Orchestrator, Environment and Episode QA agents |
| `agriphilo/dataset_spec.py`, `agriphilo/models.py` | Data contracts between stages |
| `agriphilo/credits.py`, `agriphilo/stripe_checkout.py` | Credit ledger and Stripe test-mode checkout |
| `sim/cafe/cafe_env.py` | Custom RoboCasa task `CafeServeCoffee` (2 stages: pick up the cup, place it on the tray) |
| `sim/web/server.py`, `sim/web/static/` | Playable MuJoCo game server and keyboard UI |
| `sim/scripts/render_scene.py` | Scene renders used by the Environment step |
| `examples/` | Sample café orders |
| `tests/` | `python -m pytest -q tests` |

**Scope of the pilot:** everything runs in simulation; the coffee is a visual (no liquid physics). QA passing means the episode completed the simulated task and looks usable, not that a policy transfers to a real robot. Legacy code from an earlier greenhouse/Isaac Sim prototype is kept under `examples/legacy_tomato/` and `agriphilo/static/legacy/` for reference only.
