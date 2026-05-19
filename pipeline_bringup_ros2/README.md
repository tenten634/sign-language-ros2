# Pipeline Bringup ROS2

Launch orchestration for the ASL → LLM planner → executor → eval pipeline.  
Main launch file: `pipeline_bringup_ros2.launch.py` (unified).

This package is designed around two explicit configuration axes:

- `operation_mode`: `camera` | `video` | `text`
- `environment`: `simulation` | `robotont`

Pick one value from each axis and run a single unified launch command.  
`asl_recognition_node` exit shuts down the rest of the launch.

## Required package docs (must read)

For full reproducibility, read these package READMEs before running:

- [`../asl_recognition_ros2/README.md`](../asl_recognition_ros2/README.md) - input mode behavior, model files, camera/video/text requirements
- [`../llm_planner_ros2/README.md`](../llm_planner_ros2/README.md) - planner/executor behavior and odom navigation semantics
- [`../pipeline_eval_ros2/README.md`](../pipeline_eval_ros2/README.md) - CSV metrics and summarization
- [`../pipeline_led_ros2/README.md`](../pipeline_led_ros2/README.md) - Robotont LED feedback orchestration
- [`../sign_language_msgs/README.md`](../sign_language_msgs/README.md) - shared interface definitions

## Installation

### Prerequisites

- **ROS2**: Jazzy (or compatible)
- **Packages**: `sign_language_msgs`, `asl_recognition_ros2`, `llm_planner_ros2`, `pipeline_eval_ros2`, `pipeline_led_ros2`
- **`environment:=simulation`**: `robotont_simple_simulator` in workspace (driver mode: `simple_driver.launch.py`)
- **`environment:=robotont`**: run **`robotont_driver`** (or equivalent stack) so `/odom` and `/cmd_vel` exist
- **`operation_mode:=camera`**: camera pipeline is auto-started via `realsense2_camera` when available, and a popup is shown for expected-text input during camera evaluation — [RealSense ROS2](https://github.com/realsenseai/realsense-ros)

### Dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Build

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Quick Start

Use one unified command with the 2-axis selection:

- `operation_mode`: `camera` | `video` | `text`
- `environment`: `simulation` | `robotont`

Detailed behavior differences between `operation_mode` values are documented in
[`../asl_recognition_ros2/README.md`](../asl_recognition_ros2/README.md) (`Quick Start` and `Parameters` sections).

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=<camera|video|text> \
  environment:=<simulation|robotont>
```

Reproducible example combinations:

```bash
# Simulation + video
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=video \
  environment:=simulation

# Robotont + text (file commands)
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=text \
  environment:=robotont
```

### Mode x environment matrix

| `operation_mode` | `simulation` | `robotont` |
|------------------|--------------|------------|
| `camera` | Supported (camera auto-start) | Supported (camera auto-start) |
| `video` | Supported | Supported |
| `text` | Supported | Supported |

This matrix is the recommended baseline for validation runs.

## Minimal reproducible workflow

1. Install dependencies and build the workspace.
2. Source the workspace: `source install/setup.bash`.
3. Start one run with explicit axis values.
4. Verify pipeline events on `/pipeline/status` and execution completion on `/pipeline/executor/status`.

Example checks:

```bash
ros2 topic echo /pipeline/status
ros2 topic echo /pipeline/executor/status
```

## Parameters

### `environment`

| `environment` | Bringup includes | Pipeline `use_sim_time` |
|---------------|------------------|-------------------------|
| `simulation` (default) | `robotont_simple_simulator` → **`simple_driver.launch.py`** | `true` |
| `robotont` | *(no motion stack — start `robotont_driver` yourself)* | `false` |

Executor: odom-only (`/cmd_vel` + `/odom`).

### `pipeline_bringup_ros2.launch.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `environment` | `simulation` | `simulation` or `robotont`. |
| `operation_mode` | `camera` | `camera`, `video`, or `text`. |
| `run_planner` | `true` | Launch `llm_planner_nav_node`. |
| `run_executor` | `true` | Launch `llm_plan_executor_node`. |
| `run_eval` | `true` | Launch `pipeline_eval_ros2`. |
| `run_led` | `true` | Launch `pipeline_led_node` (dedicated LED orchestrator). |
| `text_list_file` | `""` | `text` mode: command list path (default under `$HOME/ros2_ws/src/asl_recognition_ros2/` when empty). |
| `video_input_dir` | `/home/robotont/ros2_ws/src/asl_recognition_ros2/input_videos` | `video`: input root directory. |
| `wait_timeout_sec` | `120.0` | `video` / `text`: max wait for `EXECUTION DONE` on `/pipeline/executor/status` per sample. |
| `output_csv` | `$HOME/ros2_ws/pipeline_eval_metrics.csv` | Eval CSV output. |
| `led_mode_topic` | `/led_mode` | Absolute `LedModuleMode` topic (matches `robotont_driver`; relative names would publish under the LED node’s namespace). |

## Usage examples

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=video \
  environment:=robotont \
  video_input_dir:=/absolute/path/to/input_videos

ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=text \
  text_list_file:=/absolute/path/to/commands.txt

ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=text run_planner:=false run_executor:=true
```