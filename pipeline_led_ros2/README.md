# Pipeline LED ROS2

LED orchestration node for Robotont.  
Main node: `pipeline_led_node` → subscribes `/pipeline/status` and publishes `/led_mode` (`robotont_msgs/LedModuleMode`).

## Installation

### Prerequisites

- **ROS2**: Jazzy (or compatible)
- **Messages**: `robotont_msgs` available in workspace/underlay
- **Hardware run**: `robotont_driver` with LED plugin enabled (subscriber on `/led_mode`)

### Dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select pipeline_led_ros2
source install/setup.bash
```

## Quick Start

Recommended (full pipeline):

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=camera \
  environment:=robotont \
  run_led:=true
```

Standalone run (advanced/debugging):

```bash
ros2 run pipeline_led_ros2 pipeline_led_node
```

Usually started from bringup:

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py run_led:=true
```

## Parameters

### `pipeline_led_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `led_mode_topic` | string | `/led_mode` | Absolute LED topic to match driver namespace |
| `heartbeat_hz` | double | `3.0` | Refresh rate for periodic LED republish (clamped to ~0.5–20) |
| `state_hold_sec` | double | `0.35` | Ignore short “downgrades” to lower-priority statuses to reduce flicker |

Fixed in code:
- status input topic: `/pipeline/status`

## Topics

### `pipeline_led_node`

**Publishes**

| Topic | Type | Note |
|-------|------|------|
| `/led_mode` (or `led_mode_topic`) | `robotont_msgs/LedModuleMode` | LED commands (`mode`, `params`) |

**Subscribes**

| Topic | Type | Note |
|-------|------|------|
| `/pipeline/status` | `std_msgs/String` | Unified pipeline state labels |

## State mapping (implementation)

Commands use `robotont_msgs/LedModuleMode` (`SPIN`, `PULSE`, or `NONE`). Colors are RGB tuples in the message; spin speed uses a slow default (~15) or fast (~150) where noted.

| Pipeline status | LED behavior |
|-----------------|--------------|
| `waiting_for_palm`, `text_waiting`, `video_waiting` | Blue `SPIN` (slow) |
| `open_palm_hold`, `recording_end_palm_hold` | Blue `SPIN` (fast) |
| `countdown`, `recording_ready_end` | Blue `PULSE` |
| `recording` | `NONE` (off) |
| `predicting`, `text_processing`, `video_processing`, `llm_planning`, `text_waiting_plan`, `video_waiting_plan` | Red `SPIN` |
| `plan_executing` | Green `SPIN` |
| `showing_result` | `NONE` |
| `plan_ready`, `pipeline_done`, other / unknown | `NONE` (else branch) |

Rapid status “downgrades” shorter than `state_hold_sec` may be ignored to avoid flicker.
