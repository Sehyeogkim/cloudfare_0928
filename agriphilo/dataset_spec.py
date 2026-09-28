"""DatasetSpec: the contract the Customer Agent hands to the rest of the pipeline.

Pilot scope: one cafe-counter scene built in MuJoCo (RoboCasa), a mobile Franka Panda (PandaOmron),
and one multi-step task: serving a coffee (CafeServeCoffee in sim/cafe/cafe_env.py).
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Sensor = Literal[
    "rgb",
    "depth",
    "joint_state",
    "eef_pose",
    "gripper_state",
    "action",
    "object_pose",
    "task_stage",
]
Camera = Literal[
    "robot0_agentview_center", "robot0_agentview_left", "robot0_agentview_right",
    "robot0_frontview", "robot0_eye_in_hand",
]
Variety = Literal["single", "few", "many"]

TASK_STEPS = [
    "Pick up the cup of coffee",
    "Place the cup on the tray",
]


class Robot(BaseModel):
    sim_model: Literal["PandaOmron"] = "PandaOmron"
    customer_robot: Optional[str] = Field(
        None, description="The customer's real robot/gripper, if stated. The sim model is a proxy."
    )


class Environment(BaseModel):
    scene_template: Literal["robocasa_cafe_counter"] = "robocasa_cafe_counter"
    description: str = Field(description="The setting the customer asked for, in plain words")
    layout_variety: Variety = Field(description="How many different counter layouts to cover")
    style_variety: Variety = Field(description="How many different visual styles (materials, colors) to cover")


class CameraSpec(BaseModel):
    name: Camera
    width: int = Field(ge=64, le=1024)
    height: int = Field(ge=64, le=1024)
    fps: int = Field(ge=5, le=30)


class DataSchema(BaseModel):
    cameras: List[CameraSpec] = Field(min_length=1)
    sensors: List[Sensor] = Field(min_length=1)
    control_hz: int = Field(ge=10, le=30)
    format: Literal["hdf5", "lerobot"] = "hdf5"

    @model_validator(mode="after")
    def _valid(self):
        for required in ("rgb", "joint_state", "action"):
            if required not in self.sensors:
                raise ValueError(f"sensors must include {required} (episode contract)")
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError("cameras lists a camera twice")
        return self


class SuccessRule(BaseModel):
    type: Literal["all_stages_completed"] = "all_stages_completed"
    time_limit_s: float = Field(gt=0, le=600)


class DatasetSpec(BaseModel):
    mode: Literal["generation", "failure"] = "generation"
    robot: Robot
    task: Literal["CafeServeCoffee"] = "CafeServeCoffee"
    task_steps: List[str] = Field(default_factory=lambda: list(TASK_STEPS),
                                  description="The task in order; the pilot task has exactly these steps")
    environment: Environment
    data: DataSchema
    episode_count: int = Field(ge=1, le=10000)
    success_rule: SuccessRule
    reference_failure: Optional[str] = Field(None, description="Failure-mode orders: what the customer saw")
    assumptions: List[str] = Field(
        default_factory=list, description="Values the agent chose because the customer did not state them"
    )
    out_of_scope: List[str] = Field(
        default_factory=list, description="Requested items this pilot does not cover"
    )

    @property
    def sensors(self) -> List[str]:
        return list(self.data.sensors)

    @model_validator(mode="after")
    def _valid(self):
        if self.task_steps != TASK_STEPS:
            raise ValueError(f"task_steps must be exactly {TASK_STEPS}")
        if self.mode == "failure" and not self.reference_failure:
            raise ValueError("failure mode requires reference_failure")
        return self
