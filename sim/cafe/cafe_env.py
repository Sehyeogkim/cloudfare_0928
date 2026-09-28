"""Cafe coffee-serving task built on RoboCasa.

Layout on the coffee-machine counter (robot faces the counter):
  [tray]  [cup of coffee]  [COFFEE MACHINE]
The cup already holds coffee and stands on the open counter in front of the robot.

Steps:
  1. pick up the cup of coffee
  2. place the cup on the tray
"""
import numpy as np

from robocasa.environments.kitchen.kitchen import *

TRAY_GROUPS = "tray"
CUP_GROUPS = "mug"
COFFEE_RGBA = [0.35, 0.2, 0.1, 0.9]


class CafeServeCoffee(Kitchen):
    STAGES = [
        "Pick up the cup of coffee",
        "Place the cup on the tray",
    ]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine", dict(id=FixtureType.COFFEE_MACHINE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.coffee_machine)
        )
        self.init_robot_base_ref = self.coffee_machine

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the cup of coffee and place it on the tray."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "cup", COFFEE_RGBA)

    def _get_obj_cfgs(self):
        return [
            # cup of coffee on the open counter, in front of the robot between the tray and the machine
            dict(
                name="cup",
                obj_groups=CUP_GROUPS,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.coffee_machine),
                    size=(0.12, 0.12),
                    pos=("ref", -1.0),
                    offset=(-0.17, 0.02),
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            ),
            # tray on the counter next to the machine, within arm's reach
            dict(
                name="tray",
                obj_groups=TRAY_GROUPS,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.coffee_machine),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    offset=(-0.4, -0.05),
                    rotation=(0, 0),
                ),
            ),
        ]

    def _reset_internal(self):
        super()._reset_internal()
        # let the mug and tray settle so episodes start from rest (keeps replays deterministic)
        for _ in range(300):
            self.sim.step()
        self.stage = 0

    def update_stages(self):
        """Advance the task progress (latched, in order). Returns the number of completed stages."""
        if not hasattr(self, "stage"):
            self.stage = 0
        if self.stage == 0 and OU.check_obj_grasped(self, "cup"):
            self.stage = 1
        if self.stage == 1 and OU.check_obj_in_receptacle(self, "cup", "tray") and not OU.check_obj_grasped(self, "cup"):
            self.stage = 2
        return self.stage

    def _post_action(self, action):
        # advance stages every step so replays (which only call step) reach the same verdict
        self.update_stages()
        return super()._post_action(action)

    def _check_success(self):
        return getattr(self, "stage", 0) >= len(self.STAGES)
