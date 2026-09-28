# Simulator worker contract

The pipeline hands every simulation job to a **worker** and expects an HDF5 dataset back. Today the default is
the mock worker (`agriphilo/sim/mock.py`), which writes synthetic data in exactly this format. The Isaac Sim worker
on RunPod plugs in by implementing the HTTP API below; point the app at it with:

```bash
export AGRIPHILO_SIM_WORKER_URL=https://<pod-id>-8080.proxy.runpod.net
export AGRIPHILO_SIM_WORKER_TOKEN=...   # optional, sent as "Authorization: Bearer <token>"
python3 -m agriphilo.web
```

The client is `HttpSimWorker` in `agriphilo/sim/worker.py`.

## Input: `SimJob`

JSON, defined in `agriphilo/models.py`. The fields a worker needs:

| Field | Meaning |
| --- | --- |
| `job_id` | Id to echo back |
| `plan.scene_template` | Always `greenhouse_tomato_v1` for the pilot |
| `plan.camera` | `mount` (`wrist` / `fixed_front` / `fixed_side`), `resolution` (`320x240` / `640x480`), `fps` |
| `plan.lighting` | `key_light` (`sun_window` / `overhead_led` / `mixed`), `direction` (`front` / `back` / `side` / `top`) |
| `plan.control_hz`, `plan.max_steps` | Control rate and step cap |
| `episodes[]` | One entry per episode: `episode_id`, `seed`, `leaf_occlusion` (0–1 fraction of the target hidden), `fruit_offset_m` [x, y, z] from the nominal fruit pose, `lighting_lux`, `camera_offset_m` [x, y, z] from the nominal camera mount |
| `sensors` | Which streams to record (see below) |
| `success_rule` | `position_tolerance_m`, `time_limit_s`, `forbid_collisions` |

Each episode must be a deterministic function of its `seed` and parameters: QA re-runs a few episodes with the
same seed and compares `joint_state`, `action`, `ee_pose` and the success flag.

## HTTP API

| Method | Path | Body / response |
| --- | --- | --- |
| `POST` | `/jobs` | body: `SimJob` → `202 {"job_id": "<remote id>"}` |
| `GET` | `/jobs/<id>` | `{"status": "queued" \| "running" \| "done" \| "failed", "progress": {"done": n, "total": n}, "error": null, "result": {...}}` |
| `GET` | `/jobs/<id>/dataset` | the HDF5 file (streamed) |

`result` (only when `status` is `done`):

```json
{
  "worker": "isaac_sim@runpod",
  "simulator_version": "isaac-sim-6.1.0",
  "compute_seconds": 5400.0,
  "episodes": [
    {"episode_id": "ep_0000", "seed": 302720121, "success": true, "failure_reason": null, "steps": 600, "final_distance_m": 0.012}
  ]
}
```

`compute_seconds` is GPU wall time; the Payment Agent bills it as GPU hours.

## Output: HDF5 layout

```
/                                   attrs: order_id, job_id, worker, simulator_version, scene_template, plan (JSON)
/episodes/<episode_id>/             attrs: seed, params (EpisodeParams JSON), success (bool),
                                           failure_reason (str, "" on success), scene_version, simulator_version
    sim_step        (T,)            int64    simulator step index of each recorded frame, strictly increasing
    sim_time        (T,)            float64  seconds, strictly increasing
    joint_state     (T, 9)          float32  7 arm joints + 2 finger joints
    action          (T, 8)          float32  7 joint velocity commands (rad/s) + gripper command (0..1)
    ee_pose         (T, 7)          float32  end-effector position xyz (m) + quaternion wxyz, world frame
    target_pose     (T, 7)          float32  pregrasp target pose, same frame
    rgb             (T, H, W, 3)    uint8    if "rgb" in sensors
    depth           (T, H, W)       float16/32, meters     if "depth"
    instance_segmentation (T, H, W) uint8/uint16            if "instance_segmentation"
    semantic_segmentation (T, H, W) uint8/uint16            if "semantic_segmentation"
    object_pose     (T, 7)          float32  target fruit pose  if "object_pose"
    contacts        (T,)            uint8    1 when a forbidden contact happens  if "contacts"
```

All streams in an episode share the same `T` (one row per recorded frame). QA checks: every required stream
present, equal lengths, no NaN/Inf, monotonic time and steps, joint velocities within 10 rad/s, and that
`success` agrees with the final `ee_pose`/`target_pose` distance and `contacts`. Episodes that fail any check
are replaced once with new seeds and are never delivered.

Failure reasons used so far: `target_lost_occlusion`, `target_lost_low_light`, `collision_stem`, `timeout`.
Add new ones freely; they are shown to the customer as-is.
