"""GameSpec: what the Environment Agent hands the Game Server (docs/plan_game_platform_2026-09-28.md §4).

One GameSpec describes one playable RoboCasa game: which kitchen task, which scenes, which robot,
which cameras, and how many QA-passing episodes the data needer asked for.

The allowed values below are copied from the vendored sim sources instead of imported, because importing
robocasa is slow and fails while its kitchen assets are still downloading. Re-derive them if sim/ changes.
"""
from typing import List, Literal

from pydantic import BaseModel, Field, model_validator

# Single-stage ("atomic") kitchen tasks registered by KitchenEnvMeta.
# Source: `grep -h "^class " sim/robocasa/robocasa/environments/kitchen/atomic/*.py`, minus the classes that
# KitchenEnvMeta skips (PickPlace, ManipulateDoor, OpenDoor, CloseDoor) and the other abstract bases
# (PickPlaceCoffee, ManipulateDrawer, ManipulateLowerDoor, OpenDropDownDoor, CloseDropDownDoor,
# MicrowavePressButton, ManipulateStoveKnob, ManipulateSinkFaucet) whose subclasses carry the task.
# Composite tasks (sim/robocasa/robocasa/environments/kitchen/composite/) are left out: too long for a game round.
ROBOCASA_TASKS = (
    "OpenBlenderLid", "CloseBlenderLid", "TurnOnBlender",
    "CoffeeSetupMug", "CoffeeServeMug", "StartCoffeeMachine",
    "OpenDrawer", "CloseDrawer", "SlideDishwasherRack", "OpenFridgeDrawer", "CloseFridgeDrawer",
    "OpenCabinet", "CloseCabinet", "OpenMicrowave", "CloseMicrowave", "OpenFridge", "CloseFridge",
    "OpenOven", "CloseOven", "OpenDishwasher", "CloseDishwasher",
    "OpenToasterOvenDoor", "CloseToasterOvenDoor", "TurnOnMicrowave", "TurnOffMicrowave",
    "TurnOnElectricKettle", "CloseElectricKettleLid", "OpenElectricKettleLid",
    "PreheatOven", "SlideOvenRack", "NavigateKitchen", "OpenStandMixerHead", "CloseStandMixerHead",
    "TurnOnStove", "TurnOffStove", "LowerHeat",
    "PickPlaceCounterToCabinet", "PickPlaceCabinetToCounter", "PickPlaceCounterToSink", "PickPlaceSinkToCounter",
    "PickPlaceCounterToMicrowave", "PickPlaceMicrowaveToCounter", "PickPlaceCounterToOven",
    "PickPlaceCounterToStove", "PickPlaceStoveToCounter", "PickPlaceToasterToCounter",
    "PickPlaceCounterToToasterOven", "PickPlaceToasterOvenToCounter", "PickPlaceCounterToStandMixer",
    "PickPlaceFridgeShelfToDrawer", "PickPlaceFridgeDrawerToShelf", "PickPlaceCounterToDrawer",
    "PickPlaceDrawerToCounter", "PickPlaceCounterToBlender",
    "CheesyBread", "MakeIcedCoffee", "PackDessert",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout", "AdjustWaterTemperature",
    "AdjustToasterOvenTemperature", "TurnOnToasterOven", "SlideToasterOvenRack", "TurnOnToaster",
    # Custom tasks in sim/cafe/
    "CafeServeCoffee",
)

# Robots with kitchen camera configs in sim/robocasa/robocasa/utils/camera_utils.py and registered in
# sim/robosuite/robosuite/robots/__init__.py. PandaOmron (mobile Panda) is RoboCasa's default robot.
ROBOCASA_ROBOTS = ("PandaOmron", "GR1FixedLowerBody")

# Camera names defined in sim/robocasa/robocasa/utils/camera_utils.py (DEFAULT config).
ROBOCASA_CAMERAS = (
    "robot0_agentview_center", "robot0_agentview_left", "robot0_agentview_right",
    "robot0_frontview", "robot0_eye_in_hand",
)

# LayoutType / StyleType in sim/robocasa/robocasa/models/scenes/scene_registry.py: LAYOUT001..060, STYLE001..060.
MAX_LAYOUT_ID = 60
MAX_STYLE_ID = 60


class GameCamera(BaseModel):
    name: str = Field(description=f"One of {', '.join(ROBOCASA_CAMERAS)}")
    width: int = Field(ge=64, le=1024)
    height: int = Field(ge=64, le=1024)
    fps: int = Field(ge=5, le=30)

    @model_validator(mode="after")
    def _known(self):
        if self.name not in ROBOCASA_CAMERAS:
            raise ValueError(f"unknown camera {self.name!r}; allowed: {', '.join(ROBOCASA_CAMERAS)}")
        return self


class GameSuccessRule(BaseModel):
    type: Literal["robocasa_check_success"] = "robocasa_check_success"
    time_limit_s: float = Field(gt=0, le=600, description="The episode fails if _check_success() is not true by then")


class GameSpec(BaseModel):
    task: str = Field(description="RoboCasa atomic task class name")
    layout_ids: List[int] = Field(min_length=1, description=f"Kitchen layouts 1..{MAX_LAYOUT_ID}")
    style_ids: List[int] = Field(min_length=1, description=f"Kitchen styles 1..{MAX_STYLE_ID}")
    robot: str = Field(description=f"One of {', '.join(ROBOCASA_ROBOTS)}")
    cameras: List[GameCamera] = Field(min_length=1)
    control_hz: int = Field(ge=10, le=30)
    max_steps: int = Field(ge=50, le=20000)
    episodes_requested: int = Field(ge=1, le=10000, description="QA-passing episodes the data needer asked for")
    success_rule: GameSuccessRule
    credit_per_pass: float = Field(gt=0, le=100, description="Credits paid to the player per QA-passing episode")
    base_seed: int = Field(ge=0, lt=2**31)
    rationale: str = ""

    @model_validator(mode="after")
    def _valid(self):
        if self.task not in ROBOCASA_TASKS:
            raise ValueError(f"unknown RoboCasa task {self.task!r}")
        if self.robot not in ROBOCASA_ROBOTS:
            raise ValueError(f"unsupported robot {self.robot!r}; allowed: {', '.join(ROBOCASA_ROBOTS)}")
        for name, ids, hi in (("layout_ids", self.layout_ids, MAX_LAYOUT_ID), ("style_ids", self.style_ids, MAX_STYLE_ID)):
            bad = [i for i in ids if not 1 <= i <= hi]
            if bad:
                raise ValueError(f"{name} out of range 1..{hi}: {bad}")
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} has duplicates")
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError("cameras lists a camera twice")
        needed = self.success_rule.time_limit_s * self.control_hz
        if self.max_steps < needed:
            raise ValueError(f"max_steps {self.max_steps} < time_limit_s * control_hz = {needed:.0f}")
        return self
