# ASL Recognition ROS2 Node

Sign-language recognition from camera, video batch, or a text command list.  
Main node: `asl_recognition_node` → publishes `~/recognized_text` (`sign_language_msgs/RecognizedText`).

## Installation

### Prerequisites

- **ROS2**: Jazzy (or compatible)
- **Python**: ≥ 3.10
- **Camera mode**: `realsense2_camera` — [RealSense ROS2](https://github.com/realsenseai/realsense-ros)
- **Models**: before build, populate `models/` (see [Required model files](#required-model-files))

### Dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select asl_recognition_ros2
source install/setup.bash
```

### Required model files

Under `src/asl_recognition_ros2/models/`:

**MediaPipe** — [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe/solutions/guide):

```
models/mediapipe/
├── gesture_recognizer.task
├── hand_landmarker.task
├── pose_landmarker_lite.task   # or full / heavy
└── face_landmarker.task
```

**ASL** — based on [Kaggle 1st-place solution](https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution) (used with permission):

```
models/asl/
├── model.tflite
├── inference_args.json
└── character_to_prediction_index.json   # from [Kaggle dataset](https://www.kaggle.com/competitions/asl-fingerspelling/data)
```

## Quick Start

Recommended (full pipeline):

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  operation_mode:=camera \
  environment:=simulation
```

Standalone run (advanced/debugging):

### Camera (default)

```bash
ros2 launch realsense2_camera rs_launch.py   # if needed
ros2 run asl_recognition_ros2 asl_recognition_node
```

Workflow: open palm (hold) → countdown → record → result on `~/recognized_text` (repeat).

### Video

Recursive scan of `video_input_dir`. With planner/executor up, after each **non-empty** publish the node blocks until a line starting with `EXECUTION DONE` on `/pipeline/executor/status` or `wait_timeout_sec`. Empty results skip the wait. (Same wait behavior as `text` mode.)

Suggested layout (subfolder name = ground truth; underscores → spaces):

```
input_videos/go_to_kitchen/1.mp4
input_videos/turn_right_90_degrees/1.mp4
```

```bash
ros2 run asl_recognition_ros2 asl_recognition_node --ros-args -p operation_mode:=video
```

### Text (command list from file)

```bash
ros2 run asl_recognition_ros2 asl_recognition_node --ros-args -p operation_mode:=text
```

Custom list:

```bash
ros2 run asl_recognition_ros2 asl_recognition_node --ros-args \
  -p operation_mode:=text \
  -p text_list_file:=/absolute/path/to/commands.txt
```

## Parameters

### `asl_recognition_node`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation_mode` | string | `camera` | `camera`, `video`, or `text` (command list from file). |
| `image_topic` | string | `/camera/camera/color/image_raw` | Camera mode only |
| `text_list_file` | string | `<share>/test_commands.txt` | `text` mode; one command per line (comments/empty skipped); empty → `<share>/test_commands.txt` |
| `video_input_dir` | string | `/home/robotont/ros2_ws/src/asl_recognition_ros2/input_videos` | `video` root (recursive); empty → `<share>/input_videos` |
| `wait_timeout_sec` | double | `120.0` | `video` / `text`: max wait for `EXECUTION DONE` after each step |
| `use_sim_time` | bool | `false` | Match `/clock` in sim |

**Camera-only**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `palm_detection_duration` | double | `2.0` | Open-palm gate (s); range 0.5–10 |
| `countdown_duration` | double | `3.0` | Pre-record countdown (s); range 1–10 |
| `result_display_duration_sec` | double | `5.0` | How long to show the camera result UI (s); range 0–20 |
| `palm_rearm_delay_sec` | double | `1.5` | Ignore open-palm triggers for this long after returning to waiting (s) |
| `pose_model` | string | `lite` | MediaPipe pose: `lite`, `full`, `heavy` |

## Usage examples

```bash
ros2 run asl_recognition_ros2 asl_recognition_node --ros-args \
  -p palm_detection_duration:=1.5 \
  -p countdown_duration:=2.0 \
  -p pose_model:=full
```

## Topics

### `asl_recognition_node`

**Publishes**

| Topic | Type | Note |
|-------|------|------|
| `~/recognized_text` | `sign_language_msgs/RecognizedText` | Default full name: `/asl_recognition_node/recognized_text` |
| `/pipeline/asl/status` | `std_msgs/String` | ASL finite-state labels (e.g. `waiting_for_palm`, `text_waiting`, `video_processing`, `text_waiting_plan`; same strings as `/pipeline/status` while ASL drives the pipeline) |
| `/pipeline/asl/timing` | `std_msgs/String` | Timing events (`ASL TIMING ...`) for `text` emit and camera/video inference |
| `/pipeline/asl/report` | `std_msgs/String` | ASL summary/eval events (`ASL REPORT ...`, includes `ground_truth`, `prediction`, `cer`, `execution_success`, `stop_reason`) |
| `/pipeline/status` | `std_msgs/String` | Unified pipeline status mirrored by ASL for LED/eval consumers |

**Subscribes**

| Topic | Type | Note |
|-------|------|------|
| `image_topic` | `sensor_msgs/Image` | `camera` only |
| `/pipeline/executor/status` | `std_msgs/String` | `video` / `text`; unblocks on `EXECUTION DONE` lines |
| `/pipeline/planner/status` | `std_msgs/String` | `video` / `text`; unblocks on `plan_skipped:*` |

## Services

| Service | Type |
|---------|------|
| `~/process_video` | `sign_language_msgs/srv/ProcessVideo` |

```bash
ros2 service call /asl_recognition_node/process_video \
  sign_language_msgs/srv/ProcessVideo "{video_path: ''}"
ros2 service call /asl_recognition_node/process_video \
  sign_language_msgs/srv/ProcessVideo "{video_path: '/path/to/video.mp4'}"
```