# LLM Planner ROS2

LLM-based motion planning and odom-based execution from recognized text.  
Main nodes: `llm_planner_nav_node` (text → motion plan) and `llm_plan_executor_node` (plan execution) → publishes `/llm_planner/motion_plan`, `/pipeline/executor/status`, and `/cmd_vel`.

Odom execution supports `move_forward`, `move_backward`, `rotate_*`, `wait`, and `navigate_to` (virtual named poses from `locations.json`).

Typical sim/hardware setup: [`robotont_simple_simulator`](https://github.com/robotont/robotont_simple_simulator) driver mode, or `robotont_driver` on real robot.

## Installation

### Prerequisites

- **ROS2**: Jazzy (or compatible)
- **Python**: ≥ 3.10
- **Ollama**: daemon reachable; model pulled (default in code: `gemma3n:e4b-it-q4_K_M`)
- **Sim** (optional): [`robotont_simple_simulator`](https://github.com/robotont/robotont_simple_simulator) in workspace

### Dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select llm_planner_ros2
source install/setup.bash
```

## Quick Start

Recommended (full pipeline):

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=text \
  environment:=simulation
```

Standalone run (advanced/debugging):

```bash
ros2 run llm_planner_ros2 llm_planner_nav_node
```

```bash
# Odom mode (only mode): ensure /odom and /cmd_vel (e.g. robotont_driver or simple_driver)
ros2 run llm_planner_ros2 llm_plan_executor_node
```

Optional metrics:

```bash
ros2 run pipeline_eval_ros2 pipeline_eval_ros2
```

## Usage examples

```bash
ros2 run llm_planner_ros2 llm_plan_executor_node --ros-args \
  -p odom_linear_speed:=0.10 \
  -p odom_angular_speed:=0.40
```

## Parameters

### `llm_planner_nav_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_sim_time` | bool | `false` | `true` in sim (`/clock`) |
| `ollama_model` | string | `gemma3n:e4b-it-q4_K_M` | Ollama tag (`robot_motion_planner.OLLAMA_MODEL`) |
| `recognized_text_topic` | string | `/asl_recognition_node/recognized_text` | Input |
| `target_frame` | string | `map` | Frame for pose targets in messages (bringup sets `odom`) |

### `llm_plan_executor_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_sim_time` | bool | `false` | Match environment |
| `motion_plan_topic` | string | `/llm_planner/motion_plan` | JSON plan (`std_msgs/String`) |
| `cmd_vel_topic` | string | `/cmd_vel` | Odom-based motion output |
| `odom_topic` | string | `/odom` | Odom input for feedback control |
| `odom_linear_speed` | float | `0.12` | m/s |
| `odom_angular_speed` | float | `0.35` | rad/s |
| `odom_position_tolerance_m` | float | `0.03` | Stop threshold for straight moves |
| `odom_angle_tolerance_rad` | float | `0.08` | Stop threshold for rotations |
| `odom_linear_timeout_sec` | float | `60.0` | |
| `odom_rotate_timeout_sec` | float | `60.0` | |
| `odom_control_period_sec` | float | `0.05` | Control loop period |
| `odom_stuck_timeout_sec` | float | `4.0` | Abort straight-line move if odom progress stalls |
| `odom_stuck_epsilon_m` | float | `0.002` | Minimum progress (m) per window to count as moving |
| `odom_distance_scale` | float | `1.0` | Scale factor for planned forward/back distances |

## Topics

### `llm_planner_nav_node`

**Publishes**

| Topic | Type | Note |
|-------|------|------|
| `/llm_planner/normalized_text` | `sign_language_msgs/RecognizedText` | Normalized command text |
| `/llm_planner/motion_plan` | `std_msgs/String` | JSON list of validated steps |
| `/pipeline/planner/timing` | `std_msgs/String` | Planner timing events (`LLM TIMING ...`) |
| `/pipeline/planner/report` | `std_msgs/String` | Planner summarized report events (`PLANNER REPORT ...`) |
| `/pipeline/planner/status` | `std_msgs/String` | Planner-local status stream (`llm_planning`, `plan_ready`, `plan_skipped:*`) |
| `/pipeline/status` | `std_msgs/String` | Unified pipeline status updates from planner |

**Subscribes**

| Topic | Type | Note |
|-------|------|------|
| `recognized_text_topic` | `sign_language_msgs/RecognizedText` | Default full name: `/asl_recognition_node/recognized_text` |

### `llm_plan_executor_node`

**Publishes**

| Topic | Type | Note |
|-------|------|------|
| `cmd_vel_topic` | `geometry_msgs/Twist` | Default full name: `/cmd_vel` |
| `/pipeline/executor/status` | `std_msgs/String` | Execution/result events (`PLAN START`, `STEP ...`, `EXECUTION DONE`, ...) |
| `/pipeline/executor/timing` | `std_msgs/String` | Executor timing events (`EXEC TIMING ...`) |
| `/pipeline/executor/report` | `std_msgs/String` | Executor summarized report events (`EXECUTOR REPORT ...`) |
| `/pipeline/status` | `std_msgs/String` | Unified pipeline status updates from executor |

**Subscribes**

| Topic | Type | Note |
|-------|------|------|
| `motion_plan_topic` | `std_msgs/String` | Default full name: `/llm_planner/motion_plan` |
| `odom_topic` | `nav_msgs/Odometry` | Default full name: `/odom` |

## `locations.json`

Bundled next to the Python package. Optional per-location `pose` for `navigate_to`:

```json
{
  "id": "kitchen",
  "name": "Kitchen",
  "aliases": ["kitchen", "cooking area"],
  "description": "The kitchen where food is prepared and stored.",
  "pose": { "x": 1.0, "y": 0.0, "theta_deg": 0.0 }
}
```

Missing `pose`: executor logs and skips that `navigate_to`. Pose coordinates and yaw are interpreted in the `/odom` reference used by the executor.