#!/usr/bin/env python3
"""Unified bringup for ASL recognition, planner, executor, and evaluation."""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# Fixed simple sim packages when environment=simulation.
_SIMULATOR_PACKAGE = "robotont_simple_simulator"
_SIM_DRIVER_LAUNCH = "simple_driver.launch.py"
_REALSENSE_PACKAGE = "realsense2_camera"
_REALSENSE_LAUNCH = "rs_launch.py"


def _launch_setup(context, *_args, **_kwargs):
    environment = LaunchConfiguration("environment").perform(context).strip().lower()
    if environment not in ("simulation", "robotont"):
        environment = "simulation"

    operation_mode = LaunchConfiguration("operation_mode").perform(context).strip().lower()
    if operation_mode not in ("camera", "video", "text"):
        operation_mode = "camera"

    use_sim_time = environment == "simulation"

    actions = []

    if environment == "simulation":
        sim_share = get_package_share_directory(_SIMULATOR_PACKAGE)
        sim_launch_path = os.path.join(sim_share, "launch", _SIM_DRIVER_LAUNCH)
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sim_launch_path),
            )
        )

    if operation_mode == "camera":
        try:
            camera_share = get_package_share_directory(_REALSENSE_PACKAGE)
            camera_launch_path = os.path.join(camera_share, "launch", _REALSENSE_LAUNCH)
            actions.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(camera_launch_path),
                )
            )
        except PackageNotFoundError:
            pass

    executor_params = [
        {"use_sim_time": use_sim_time},
    ]

    asl_node = Node(
        package="asl_recognition_ros2",
        executable="asl_recognition_node",
        name="asl_recognition_node",
        output="screen",
        parameters=[
            {"operation_mode": operation_mode},
            {"text_list_file": LaunchConfiguration("text_list_file")},
            {"video_input_dir": LaunchConfiguration("video_input_dir")},
            {"wait_timeout_sec": LaunchConfiguration("wait_timeout_sec")},
            {"use_sim_time": use_sim_time},
        ],
    )

    planner_params = [
        {"use_sim_time": use_sim_time},
        {"target_frame": "odom"},
    ]

    planner_node = Node(
        package="llm_planner_ros2",
        executable="llm_planner_nav_node",
        name="llm_planner_nav_node",
        output="screen",
        parameters=planner_params,
    )

    executor_node = Node(
        package="llm_planner_ros2",
        executable="llm_plan_executor_node",
        name="llm_plan_executor_node",
        output="screen",
        parameters=executor_params
        + [
        ],
    )

    eval_node = Node(
        package="pipeline_eval_ros2",
        executable="pipeline_eval_ros2",
        name="pipeline_eval_ros2",
        output="screen",
        parameters=[
            {"output_csv": LaunchConfiguration("output_csv")},
            {"use_sim_time": use_sim_time},
        ],
    )

    led_node = Node(
        package="pipeline_led_ros2",
        executable="pipeline_led_node",
        name="pipeline_led_node",
        output="screen",
        parameters=[
            {"led_mode_topic": LaunchConfiguration("led_mode_topic")},
        ],
    )

    shutdown_on_asl_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=asl_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="asl_recognition_node exited; shutting down pipeline bringup")
                )
            ],
        )
    )

    run_planner = LaunchConfiguration("run_planner").perform(context).strip().lower() == "true"
    run_executor = LaunchConfiguration("run_executor").perform(context).strip().lower() == "true"
    run_eval = LaunchConfiguration("run_eval").perform(context).strip().lower() == "true"
    run_led = LaunchConfiguration("run_led").perform(context).strip().lower() == "true"

    nodes_after = [asl_node]
    if run_planner:
        nodes_after.append(planner_node)
    if run_executor:
        nodes_after.append(executor_node)
    if run_eval:
        nodes_after.append(eval_node)
    if run_led:
        nodes_after.append(led_node)
    nodes_after.append(shutdown_on_asl_exit)

    return actions + nodes_after


def generate_launch_description() -> LaunchDescription:
    environment_arg = DeclareLaunchArgument(
        "environment",
        default_value="simulation",
        description="simulation: robotont_simple_simulator driver mode; robotont: bring your own odom/cmd_vel driver.",
        choices=["simulation", "robotont"],
    )
    mode_arg = DeclareLaunchArgument(
        "operation_mode",
        default_value="camera",
        description="ASL operation mode: camera, video, text",
    )
    run_planner_arg = DeclareLaunchArgument(
        "run_planner",
        default_value="true",
        description="Run llm_planner_nav_node",
    )
    run_executor_arg = DeclareLaunchArgument(
        "run_executor",
        default_value="true",
        description="Run llm_plan_executor_node",
    )
    run_eval_arg = DeclareLaunchArgument(
        "run_eval",
        default_value="true",
        description="Run pipeline_eval_ros2",
    )
    run_led_arg = DeclareLaunchArgument(
        "run_led",
        default_value="true",
        description="Run pipeline_led_node for dedicated Robotont LED feedback orchestration.",
    )
    text_list_file_arg = DeclareLaunchArgument(
        "text_list_file",
        default_value="",
        description="Path to command list for text mode",
    )
    video_input_dir_arg = DeclareLaunchArgument(
        "video_input_dir",
        default_value="/home/robotont/ros2_ws/src/asl_recognition_ros2/input_videos",
        description="Video input directory for video mode",
    )
    wait_timeout_sec_arg = DeclareLaunchArgument(
        "wait_timeout_sec",
        default_value="120.0",
        description="Max seconds to wait for EXECUTION DONE after each sample in text and video modes",
    )
    output_csv_arg = DeclareLaunchArgument(
        "output_csv",
        default_value=PathJoinSubstitution(
            [EnvironmentVariable("HOME"), "ros2_ws", "pipeline_eval_metrics.csv"]
        ),
        description="CSV path for per-plan pipeline metrics",
    )
    led_mode_topic_arg = DeclareLaunchArgument(
        "led_mode_topic",
        default_value="/led_mode",
        description="Absolute topic for robotont_msgs/LedModuleMode (driver default is /led_mode).",
    )

    return LaunchDescription(
        [
            environment_arg,
            mode_arg,
            run_planner_arg,
            run_executor_arg,
            run_eval_arg,
            run_led_arg,
            text_list_file_arg,
            video_input_dir_arg,
            wait_timeout_sec_arg,
            output_csv_arg,
            led_mode_topic_arg,
            OpaqueFunction(function=_launch_setup),
        ]
    )
