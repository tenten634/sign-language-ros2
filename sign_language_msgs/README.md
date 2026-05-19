# Sign Language Messages ROS2

Shared ROS 2 interface package for the sign-language stack.  
It defines **`RecognizedText`** (planner input and eval) and **`ProcessVideo`** (ASL video helper service). The planner consumes `std_msgs/String` JSON on `/llm_planner/motion_plan`; that type is not part of this package.

## Installation

### Prerequisites

- **ROS2**: Jazzy (or compatible)
- **Build type**: `ament_cmake` + ROS 2 interface generation (`rosidl`)

### Dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select sign_language_msgs
source install/setup.bash
```

## Messages

| Type | Purpose |
|------|---------|
| `RecognizedText.msg` | Recognized command text payload |

## Services

| Type | Purpose |
|------|---------|
| `ProcessVideo.srv` | Request ASL processing for a video path |
