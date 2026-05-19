# Pipeline Evaluation ROS2

ROS2 package for recording and summarizing evaluation metrics for the
ASL-to-LLM-to-execution pipeline.

- Input: `sign_language_msgs/RecognizedText`, planner/executor events, and optional unified `/pipeline/status`
- Output: **one CSV file** with per-plan pipeline metrics

## Installation

### Prerequisites

- **ROS2**: Jazzy (or compatible)
- **Python**: ≥ 3.10
- **Upstream nodes running**:
  - `asl_recognition_ros2` (for `RecognizedText` and recognition-eval stream)
  - `llm_planner_ros2` planner and executor (for execution events)

### Dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select pipeline_eval_ros2
source install/setup.bash
```

## Quick Start

Recommended (full pipeline):

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=video \
  environment:=simulation \
  run_eval:=true
```

Standalone run (advanced/debugging):

```bash
ros2 run pipeline_eval_ros2 pipeline_eval_ros2
```

Offline summary:

```bash
ros2 run pipeline_eval_ros2 pipeline_eval_summary \
  --metrics-csv ~/ros2_ws/pipeline_eval_metrics.csv
```

## Parameters

### `pipeline_eval_ros2`

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `use_sim_time` | bool | `false` | Match `/clock` in simulation setups. |
| `output_csv` | string | `~/ros2_ws/pipeline_eval_metrics.csv` | CSV file for per-plan timing / LLM / recognition metrics. |

## Topics

### `pipeline_eval_ros2`

**Subscribes**

| Topic Name | Type | Description |
|-----------|------|-------------|
| `/asl_recognition_node/recognized_text` | `sign_language_msgs/RecognizedText` | Recognized text from `asl_recognition_ros2` (raw mode). |
| `/pipeline/executor/status` | `std_msgs/String` | Execution status/events from executor (raw mode). |
| `/pipeline/planner/timing` | `std_msgs/String` | Planner timing payload (`LLM TIMING ...`, raw mode). |
| `/pipeline/executor/timing` | `std_msgs/String` | Executor timing payload (`EXEC TIMING ...`, raw mode). |
| `/pipeline/asl/timing` | `std_msgs/String` | ASL timing payload (`ASL TIMING ...`, raw mode). |
| `/pipeline/asl/report` | `std_msgs/String` | ASL summarized report payload (includes `ground_truth`, `prediction`, `cer`). |
| `/pipeline/planner/report` | `std_msgs/String` | Planner summarized report payload (`PLANNER REPORT ...`). |
| `/pipeline/executor/report` | `std_msgs/String` | Executor summarized report payload (`EXECUTOR REPORT ...`). |
| `/pipeline/status` | `std_msgs/String` | Unified pipeline phase label (same topic as LED node). |

## Output CSV

Default path: `~/ros2_ws/pipeline_eval_metrics.csv`

Columns:
- `plan_seq`, `num_steps`, `text`
- `t_text`, `t_plan`, `t_done`
- `dt_text_to_plan`, `dt_plan_to_done`, `dt_text_to_done`
- `llm_total`, `llm_norm`, `llm_plan`, `llm_valid`
- `gt_text`, `pred_text`, `cer`
- `plan_success`, `execution_success`
- `pipeline_status` (last message on `/pipeline/status` when the row is written at `EXECUTION DONE`)
- `target_distance_m`, `actual_distance_m`, `distance_error_m`, `timeout_flag`, `stop_reason`
- `asl_mode`, `asl_stage`, `asl_t_in`, `asl_t_done`, `asl_total`
- `asl_report_stage`, `asl_report_result`, `asl_report_total`
- `planner_report_result`, `planner_report_total`
- `executor_report_result`, `executor_report_total`
- `cpu_percent_done`, `mem_percent_done`

Existing CSV files are auto-migrated on node start: missing columns are appended and older rows get empty/default values.

Plan timing is taken from executor `PLAN START` / `EXECUTION DONE` lines on `/pipeline/executor/status`, not from `/llm_planner/motion_plan` (that topic is only consumed by the executor). Raw and report streams are always subscribed in the current design.
