# Sign language ROS 2 stack

Colcon packages for ASL recognition, LLM-based command planning, and launch bringup.

## Start here

For reproducible runs, use the unified bringup documented in
[`pipeline_bringup_ros2/README.md`](pipeline_bringup_ros2/README.md).

Pick one value from each axis:

- **Environment**: `simulation` or `robotont`
- **Operation mode**: `camera`, `video`, or `text`

Launch once with both values:

```bash
ros2 launch pipeline_bringup_ros2 pipeline_bringup_ros2.launch.py \
  environment:=simulation \
  operation_mode:=video
```

## Recommended reading order

1. [`pipeline_bringup_ros2/README.md`](pipeline_bringup_ros2/README.md) (top-level launch and matrix)
2. [`asl_recognition_ros2/README.md`](asl_recognition_ros2/README.md) (camera/video/text input semantics)
3. [`llm_planner_ros2/README.md`](llm_planner_ros2/README.md) (planner and odom executor behavior)
4. [`pipeline_eval_ros2/README.md`](pipeline_eval_ros2/README.md) (metrics CSV and summarizer)

## Packages

| Package | Description |
|--------|-------------|
| [`asl_recognition_ros2`](asl_recognition_ros2/README.md) | ASL fingerspelling recognition node |
| [`llm_planner_ros2`](llm_planner_ros2/README.md) | LLM planner + odom executor ROS 2 package |
| [`sign_language_msgs`](sign_language_msgs/README.md) | `RecognizedText` message and `ProcessVideo` service (shared ASL I/O) |
| [`pipeline_eval_ros2`](pipeline_eval_ros2/README.md) | Pipeline evaluation recorder and CSV summary tools |
| [`pipeline_led_ros2`](pipeline_led_ros2/README.md) | Dedicated Robotont LED feedback orchestrator for ASL + planner/executor states |
| [`pipeline_bringup_ros2`](pipeline_bringup_ros2/README.md) | Launch orchestration for odom-based execution (`/cmd_vel` + `/odom`) |

## Results

The [`results/`](results/README.md) directory holds cleaned per-trial CSV exports used in the thesis appendix (video, text-injection, and robot-camera evaluations). They are derived from [`pipeline_eval_ros2`](pipeline_eval_ros2/README.md) logger output with appendix-aligned columns and row order; see [`results/README.md`](results/README.md) for file names, column definitions, and camera-trial exclusions.

## License

All packages in this repository are licensed under MIT.
