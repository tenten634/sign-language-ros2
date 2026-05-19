#!/usr/bin/env python3
"""
ROS2 node for ASL fingerspelling recognition.

Subscribes to RealSense camera image topic, detects open palm gesture,
records frames, and runs inference to recognize sign language.
"""

from __future__ import annotations

import json
import time
import threading
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import (
    ParameterDescriptor,
    ParameterType,
    FloatingPointRange,
)
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from sign_language_msgs.msg import RecognizedText
from sign_language_msgs.srv import ProcessVideo
from cv_bridge import CvBridge

import tensorflow as tf

# Late-joining llm_planner_nav_node (later process in the same launch) must not miss the
# first publish; default VOLATILE drops data if no subscriber exists yet.
_RECOGNIZED_TEXT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

PIPELINE_STATUS_TOPIC = "/pipeline/status"
ASL_STATUS_TOPIC = "/pipeline/asl/status"
EXECUTION_STATUS_TOPIC = "/pipeline/executor/status"
PLANNER_STATUS_TOPIC = "/pipeline/planner/status"
ASL_TIMING_TOPIC = "/pipeline/asl/timing"
ASL_REPORT_TOPIC = "/pipeline/asl/report"

try:
    import tkinter as tk
    from tkinter import simpledialog
except Exception:  # noqa: BLE001
    tk = None
    simpledialog = None


class State:
    """State management for the prediction loop"""
    WAITING_FOR_PALM = "waiting_for_palm"
    OPEN_PALM_HOLD = "open_palm_hold"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    RECORDING_READY_END = "recording_ready_end"
    RECORDING_END_PALM_HOLD = "recording_end_palm_hold"
    PREDICTING = "predicting"
    SHOWING_RESULT = "showing_result"
    
    def __init__(self):
        self.current_state = self.WAITING_FOR_PALM
        self.prediction_result = ""
        self.prediction_lock = threading.Lock()
        self.is_predicting = False
        self.palm_detection_start = None
        self.countdown_start = None
        self.recording_frames = []
        self.recorded_frames = None
        # For extended recording: continue recording after 384 frames until palm is detected
        self.min_frames_reached = False  # True when 384 frames are reached
        self.palm_detection_end_start = None  # Start time of palm detection at the end
        self.palm_detection_end_frames_start_idx = None  # Frame index where palm detection started at the end


class LandmarkBuffer:
    """Buffer for landmarks"""
    def __init__(self, max_frames=384, selected_columns=None):
        self.max_frames = max_frames
        self.buffer = deque(maxlen=max_frames)
        self.selected_columns = selected_columns or []
        
    def add_frame(self, landmarks_dict):
        """Add frame landmarks"""
        frame_data = [landmarks_dict.get(col, 0.0) for col in self.selected_columns]
        self.buffer.append(frame_data)
        
    def get_sequence(self):
        """Get current sequence"""
        if len(self.buffer) == 0:
            return None
        return np.array(self.buffer, dtype=np.float32)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def is_full(self):
        """Check if buffer is full"""
        return len(self.buffer) >= self.max_frames


def split_hand_landmarks(
    hand_result: Optional[mp_vision.HandLandmarkerResult],
    mirror: bool = False,
) -> Tuple[Optional[List], Optional[List]]:
    """Split hand landmarks into left and right"""
    left_hand = None
    right_hand = None
    if hand_result and hand_result.hand_landmarks:
        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            handedness_list = hand_result.handedness[idx] if hand_result.handedness else []
            if handedness_list:
                handedness = handedness_list[0]
                category_name = handedness.category_name.lower()
                
                if mirror:
                    if category_name == 'left':
                        category_name = 'right'
                    elif category_name == 'right':
                        category_name = 'left'
                
                if category_name == 'left':
                    left_hand = landmarks
                elif category_name == 'right':
                    right_hand = landmarks
            elif not handedness_list and left_hand is None:
                left_hand = landmarks
    return left_hand, right_hand


def extract_landmarks_to_dict(
    pose_landmarks: Optional[List],
    face_landmarks: Optional[List],
    left_hand_landmarks: Optional[List],
    right_hand_landmarks: Optional[List],
    selected_columns: List[str],
) -> Dict[str, float]:
    """Convert MediaPipe landmarks to dictionary format"""
    landmarks_dict = {}
    landmark_map = {
        'face': face_landmarks,
        'left_hand': left_hand_landmarks,
        'right_hand': right_hand_landmarks,
        'pose': pose_landmarks,
    }
    
    for col in selected_columns:
        value = 0.0
        try:
            parts = col.split('_')
            if len(parts) >= 3:
                coord = parts[0]
                landmark_type = '_'.join(parts[1:-1])
                idx_str = parts[-1]
                
                try:
                    idx = int(idx_str)
                    landmarks = landmark_map.get(landmark_type)
                    
                    if landmarks and idx < len(landmarks):
                        if coord == 'x':
                            value = landmarks[idx].x
                        elif coord == 'y':
                            value = landmarks[idx].y
                        elif coord == 'z':
                            value = landmarks[idx].z
                except (ValueError, IndexError, AttributeError):
                    pass
        except (ValueError, IndexError, AttributeError, KeyError):
            pass
        
        landmarks_dict[col] = value
    
    return landmarks_dict


def decode_prediction(output_scores, num_to_char, pad_token_id, bos_token_id, eos_token_id):
    """Convert model output to string"""
    if isinstance(output_scores, np.ndarray):
        if len(output_scores.shape) == 1:
            if output_scores.shape[0] > 100:
                pred_ids = output_scores.astype(np.int32)
            else:
                pred_ids = np.array([np.argmax(output_scores)])
        elif len(output_scores.shape) == 2:
            if output_scores.shape[1] >= 50:
                pred_ids = np.argmax(output_scores, axis=-1)
            else:
                pred_ids = output_scores[0] if output_scores.shape[0] == 1 else output_scores.flatten()
        elif len(output_scores.shape) == 3:
            pred_ids = np.argmax(output_scores[0], axis=-1)
        else:
            pred_ids = np.argmax(output_scores, axis=-1)
    else:
        pred_ids = output_scores
        if hasattr(pred_ids, 'shape') and len(pred_ids.shape) > 1:
            pred_ids = pred_ids.flatten()
    
    pred_ids = np.array(pred_ids).flatten()
    
    pred_str = ''
    for idx, token_id in enumerate(pred_ids):
        token_id = int(token_id)
        if token_id in [pad_token_id, bos_token_id]:
            continue
        if token_id == eos_token_id:
            break
        char = num_to_char.get(token_id, None)
        if char is not None:
            pred_str += char
    
    result = pred_str.strip()
    return result


def open_video_capture_with_orientation(video_path: Path):
    """
    Open a video with OpenCV and enable CAP_PROP_ORIENTATION_AUTO when available
    to correct orientation (e.g., for portrait videos).
    Returns: (cap, metadata_dict) — metadata is for logging
    (META / AUTO values before and after enabling).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return cap, {}

    metadata = {}
    if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        metadata["orientation_meta"] = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        metadata["orientation_auto_before"] = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        metadata["orientation_auto_after"] = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)

    return cap, metadata


class ASLRecognitionNode(Node):
    """ROS2 node for ASL fingerspelling recognition"""
    
    def __init__(self):
        super().__init__('asl_recognition_node')
        
        # Models / data roots (used only in camera/video modes)
        package_root = Path(__file__).resolve().parent.parent
        models_base = package_root / 'models'
        try:
            share_dir = Path(get_package_share_directory('asl_recognition_ros2'))
            if (share_dir / 'models').exists():
                models_base = share_dir / 'models'
                package_root = share_dir
        except Exception:
            pass
        self.models_dir = models_base / 'mediapipe'
        self.asl_models_dir = models_base / 'asl'
        self.package_root = package_root

        # Frame count is fixed by model requirements (384 frames)
        self.num_frames = 384

        # use_sim_time: launch often passes this; rclpy (e.g. Jazzy) may already declare it on Node.
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter(
                'use_sim_time',
                False,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_BOOL,
                    description='Use /clock when true (simulation); false for real robot / wall clock.',
                ),
            )

        # Configurable parameters (with descriptors for ros2 param describe)
        self.declare_parameter(
            'operation_mode',
            'camera',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Operation mode: "camera" (camera input with state machine), "video" (video-only batch processing), or "text" (commands from file).',
                additional_constraints='One of: camera, video, text',
            ),
        )
        self.declare_parameter(
            'image_topic',
            '/camera/camera/color/image_raw',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Image topic to subscribe to (e.g. RealSense color image). Can be remapped via launch.',
            ),
        )
        self.declare_parameter(
            'palm_detection_duration',
            2.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Seconds to detect open palm before starting countdown.',
                floating_point_range=[FloatingPointRange(from_value=0.5, to_value=10.0, step=0.5)],
            ),
        )
        self.declare_parameter(
            'countdown_duration',
            3.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Countdown duration in seconds before recording starts.',
                floating_point_range=[FloatingPointRange(from_value=1.0, to_value=10.0, step=0.5)],
            ),
        )
        self.declare_parameter(
            'result_display_duration_sec',
            5.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Seconds to show camera recognition result popup.',
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=20.0, step=0.5)],
            ),
        )
        self.declare_parameter(
            'palm_rearm_delay_sec',
            1.5,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Seconds to ignore open-palm trigger after returning to waiting state from result.',
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.5)],
            ),
        )
        self.declare_parameter(
            'pose_model',
            'lite',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='MediaPipe pose model: lite, full, or heavy.',
                additional_constraints='One of: lite, full, heavy',
            ),
        )
        self.declare_parameter(
            'text_list_file',
            str(self.package_root / 'test_commands.txt'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path to text list file used when operation_mode is "text".',
            ),
        )
        self.declare_parameter(
            'wait_timeout_sec',
            120.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    'In text and video modes: max seconds to wait for an EXECUTION DONE event '
                    'on execution_status_topic after each published command (or timeout).'
                ),
            ),
        )
        # Determine operation mode
        operation_mode_param = self.get_parameter('operation_mode').value.strip().lower()

        if operation_mode_param in ('camera', 'video', 'text'):
            self.operation_mode = operation_mode_param
        else:
            self.get_logger().warning(f'operation_mode="{operation_mode_param}" is invalid, using default "camera"')
            self.operation_mode = 'camera'
        self.palm_detection_duration = self.get_parameter('palm_detection_duration').value
        self.countdown_duration = self.get_parameter('countdown_duration').value
        self.result_display_duration_sec = max(0.0, float(self.get_parameter('result_display_duration_sec').value))
        self.palm_rearm_delay_sec = max(0.0, float(self.get_parameter('palm_rearm_delay_sec').value))
        self.pose_model = self.get_parameter('pose_model').value.strip().lower()
        if self.pose_model not in ('lite', 'full', 'heavy'):
            self.get_logger().warning(f'pose_model="{self.pose_model}" unknown, using "lite"')
            self.pose_model = 'lite'
        # CV Bridge
        self.bridge = CvBridge()
        
        # State
        self.state = State()
        self.frame_timestamp_ms = 0
        self._palm_rearm_until: float = 0.0
        self._result_display_until: float = 0.0
        
        # Performance optimization: skip frames for gesture recognition
        self.gesture_check_counter = 0
        self.gesture_check_interval = 5  # Check gesture every 5 frames (~6fps instead of 30fps)
        
        # Publisher for recognized text (all modes)
        self.recognized_text_pub = self.create_publisher(
            RecognizedText,
            '~/recognized_text',
            _RECOGNIZED_TEXT_QOS,
        )
        self.asl_timing_pub = self.create_publisher(
            String,
            ASL_TIMING_TOPIC,
            10,
        )
        self.asl_report_pub = self.create_publisher(String, ASL_REPORT_TOPIC, 10)
        self.state_pub = self.create_publisher(
            String,
            ASL_STATUS_TOPIC,
            10,
        )
        self.pipeline_status_pub = self.create_publisher(
            String,
            PIPELINE_STATUS_TOPIC,
            10,
        )
        self._last_published_state: Optional[str] = None
        
        # Video processing
        # Default video input directory (can be overridden via parameter)
        self.declare_parameter(
            'video_input_dir',
            '/home/robotont/ros2_ws/src/asl_recognition_ros2/input_videos',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Directory containing input videos when operation_mode is "video".',
            ),
        )
        video_input_dir_param = self.get_parameter('video_input_dir').value.strip()
        self.video_input_dir = Path(video_input_dir_param) if video_input_dir_param else (self.package_root / 'input_videos')
        
        # Video processing state
        self.processed_videos = set()
        self.video_processing_lock = threading.Lock()
        self._plan_done_counter = 0
        self._execution_done_counter = 0
        self._planner_skipped_counter = 0
        self._last_planner_skip_reason = ""
        self.wait_timeout_sec = float(self.get_parameter('wait_timeout_sec').value) or 120.0
        self.exec_sub = None
        self.planner_status_sub = None
        # Shutdown flag - set to True when node wants to exit
        self.should_shutdown = False
        
        # Services (camera / video modes only)
        self.process_video_service = None

        # Initialize based on operation mode
        if self.operation_mode == 'text':
            self.image_sub = None
            self.timer = None
            text_list_file_param = str(self.get_parameter('text_list_file').value).strip()
            self.text_list_file = Path(text_list_file_param) if text_list_file_param else (self.package_root / 'test_commands.txt')
            self.exec_sub = self.create_subscription(
                String,
                EXECUTION_STATUS_TOPIC,
                self._exec_event_callback,
                10,
            )
            self.planner_status_sub = self.create_subscription(
                String,
                PLANNER_STATUS_TOPIC,
                self._exec_event_callback,
                10,
            )
            self.get_logger().info(
                f'Text mode: using file {self.text_list_file}, '
                f'wait_timeout_sec={self.wait_timeout_sec:.1f} '
                f'(waits for EXECUTION DONE on {EXECUTION_STATUS_TOPIC} or '
                f'plan_skipped on {PLANNER_STATUS_TOPIC}, or timeout per command)'
            )
            threading.Thread(target=self._run_text_loop, daemon=True).start()
        else:
            # For camera / video modes we need models and services
            self.get_logger().info('Loading models...')
            self._load_models()
            self.get_logger().info('Models loaded successfully')

            self.process_video_service = self.create_service(
                ProcessVideo,
                '~/process_video',
                self.process_video_callback
            )
            self.get_logger().info('Service process_video is ready')

        if self.operation_mode == 'camera':
            # Camera mode: subscribe to image topic and run state machine
            image_topic = self.get_parameter('image_topic').value
            self.image_sub = self.create_subscription(
                Image,
                image_topic,
                self.image_callback,
                10
            )
            self.camera_connected = False  # Set to True when first image is received
            self._camera_expected_text: Optional[str] = None
            self.get_logger().info(f'Camera mode: subscribing to {image_topic}')
            self.timer = self.create_timer(0.033, self.timer_callback)
            self._prepare_next_camera_ground_truth()
        elif self.operation_mode == 'video':
            # Video-only mode: no camera subscription, no state machine; process all videos and exit
            self.image_sub = None
            self.timer = None
            self.exec_sub = self.create_subscription(
                String,
                EXECUTION_STATUS_TOPIC,
                self._exec_event_callback,
                10,
            )
            self.planner_status_sub = self.create_subscription(
                String,
                PLANNER_STATUS_TOPIC,
                self._exec_event_callback,
                10,
            )
            self.get_logger().info(
                'Video mode: processing all videos in folder (no camera); '
                f'will wait for EXECUTION DONE on {EXECUTION_STATUS_TOPIC} '
                f'or timeout ({self.wait_timeout_sec:.1f}s) after each non-empty prediction'
            )
            threading.Thread(
                target=self._process_all_videos_in_folder_and_exit,
                daemon=False,
                kwargs={'log_comparison': True},
            ).start()

        self._publish_state_if_changed(force=True)

    def _publish_state_if_changed(self, state: Optional[str] = None, *, force: bool = False) -> None:
        """Publish unified pipeline state for downstream orchestrators (LED/UI/etc)."""
        state = state if state is not None else self.state.current_state
        if not force and state == self._last_published_state:
            return
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.pipeline_status_pub.publish(msg)
        self._last_published_state = state

    def _publish_asl_timing(self, stage: str, t_in: float, t_done: float, **extra: object) -> None:
        """Publish ASL timing telemetry as key=value fields for eval subscribers."""
        msg = String()
        parts = [
            "ASL TIMING",
            f"mode={self.operation_mode}",
            f"stage={stage}",
            f"t_in={t_in:.3f}",
            f"t_done={t_done:.3f}",
            f"total={max(0.0, t_done - t_in):.3f}",
        ]
        for key, value in extra.items():
            parts.append(f"{key}={value}")
        msg.data = " ".join(parts)
        self.asl_timing_pub.publish(msg)

    def _publish_asl_report(self, stage: str, t_in: float, t_done: float, result: str, **extra: object) -> None:
        """Publish summarized ASL report events for evaluation."""
        msg = String()
        parts = [
            "ASL REPORT",
            f"mode={self.operation_mode}",
            f"stage={stage}",
            f"result={result}",
            f"t_in={t_in:.3f}",
            f"t_done={t_done:.3f}",
            f"total={max(0.0, t_done - t_in):.3f}",
        ]
        for key, value in extra.items():
            parts.append(f"{key}={value}")
        msg.data = " ".join(parts)
        self.asl_report_pub.publish(msg)

    def _load_text_commands(self, file_path: Path) -> List[str]:
        """Load non-empty, non-comment commands from file."""
        if not file_path.exists():
            raise FileNotFoundError(f'Text list file not found: {file_path}')
        commands: List[str] = []
        for line in file_path.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            commands.append(stripped)
        return commands

    def _exec_event_callback(self, msg: String) -> None:
        """Track EXECUTION DONE / planner skip events on execution_status_topic (text and video modes)."""
        text = msg.data or ''
        if text.startswith('EXECUTION DONE'):
            self._execution_done_counter += 1
            self.get_logger().info(
                f'Text/video: EXECUTION DONE received, counter={self._execution_done_counter}'
            )
        elif text.startswith('plan_skipped:'):
            self._planner_skipped_counter += 1
            self._last_planner_skip_reason = text.split(':', 1)[1].strip() or 'unknown'
            self.get_logger().info(
                f'Text/video: planner skipped ({self._last_planner_skip_reason}), '
                f'counter={self._planner_skipped_counter}'
            )

    def _run_text_loop(self) -> None:
        """Publish text commands sequentially and report video-aligned evaluation metrics."""
        self._publish_state_if_changed("text_waiting", force=True)
        try:
            commands = self._load_text_commands(self.text_list_file)
        except Exception as exc:
            self.get_logger().error(f'Text mode failed to load commands: {exc}')
            self.should_shutdown = True
            return

        if not commands:
            self.get_logger().warning(f'Text mode: no commands found in {self.text_list_file}')
            self.should_shutdown = True
            return

        self.get_logger().info(f'Text mode: loaded {len(commands)} command(s)')
        total = 0
        total_cer = 0.0
        downstream_total = 0
        downstream_success = 0

        for idx, command in enumerate(commands, start=1):
            if not rclpy.ok() or self.should_shutdown:
                break

            self.get_logger().info(f'Text [{idx}/{len(commands)}]: "{command}"')
            t_cmd_start = time.time()
            self._publish_state_if_changed("text_processing")
            self._publish_result(command)

            # Keep evaluation format aligned with video mode for publication tables.
            gt = self._normalize_sentence(command)
            pred = self._normalize_sentence(command)
            cer = self._character_error_rate(gt, pred)
            total += 1
            total_cer += cer
            self.get_logger().info(
                f'Recognition eval: cer={cer:.4f}'
            )
            self._publish_asl_timing(
                "text_emit",
                t_cmd_start,
                time.time(),
                sample_idx=idx,
                samples_total=len(commands),
            )
            self._publish_asl_report(
                "text_emit",
                t_cmd_start,
                time.time(),
                result="ok",
                ground_truth=gt.replace(" ", "_"),
                prediction=pred.replace(" ", "_"),
                cer=f"{cer:.6f}",
                sample_idx=idx,
                samples_total=len(commands),
            )

            downstream_total += 1
            self._publish_state_if_changed("text_waiting_plan")
            wait_kind, wait_reason = self._wait_for_plan_done()
            if wait_kind == "execution_done":
                downstream_success += 1
                self.get_logger().info('Text: EXECUTION DONE received, moving to next command')
            elif wait_kind == "plan_skipped":
                self.get_logger().warning(
                    f'Text: plan skipped ({wait_reason}); mark failure and move to next command'
                )
                self._publish_recognition_eval(
                    "text",
                    gt,
                    pred,
                    cer,
                    execution_success=0,
                    stop_reason=f"plan_skipped:{wait_reason}",
                )
                time.sleep(1.0)
            else:
                self.get_logger().warning('Text: timeout waiting for EXECUTION DONE, moving to next command')

        if total > 0:
            avg_cer = total_cer / total
            self.get_logger().info(
                f'Text recognition summary: samples={total}, avg_cer={avg_cer:.4f}'
            )
            downstream_acc = (downstream_success / downstream_total) * 100.0 if downstream_total else 0.0
            self.get_logger().info(
                f'Text downstream summary: plan_done={downstream_success}/{downstream_total} ({downstream_acc:.1f}%)'
            )

        self.get_logger().info('Text mode finished. Exiting...')
        self._publish_state_if_changed("pipeline_done", force=True)

        self.should_shutdown = True

    def _wait_for_plan_done(self) -> Tuple[str, str]:
        """Wait for terminal event and return (kind, reason)."""
        start_exec_seq = self._execution_done_counter
        start_skip_seq = self._planner_skipped_counter
        self.get_logger().info('Waiting for EXECUTION DONE...')
        end_time = time.time() + self.wait_timeout_sec
        while (
            rclpy.ok()
            and not self.should_shutdown
            and self._execution_done_counter <= start_exec_seq
            and self._planner_skipped_counter <= start_skip_seq
            and time.time() < end_time
        ):
            time.sleep(0.2)
        if self._execution_done_counter > start_exec_seq:
            return ("execution_done", "")
        if self._planner_skipped_counter > start_skip_seq:
            return ("plan_skipped", self._last_planner_skip_reason or "unknown")
        return ("timeout", "")

    def _load_models(self):
        """Load MediaPipe and TFLite models"""
        # Load gesture recognizer
        gesture_model_path = self.models_dir / "gesture_recognizer.task"
        if not gesture_model_path.exists():
            self.get_logger().error(f'Gesture model not found: {gesture_model_path}')
            raise FileNotFoundError(f'Gesture model not found: {gesture_model_path}')
        
        base_options = mp_tasks.BaseOptions(model_asset_path=str(gesture_model_path))
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(options)
        
        # Load MediaPipe landmark models
        hand_model_path = self.models_dir / "hand_landmarker.task"
        pose_model_filename = f"pose_landmarker_{self.pose_model}.task"
        pose_model_path = self.models_dir / pose_model_filename
        if not pose_model_path.exists():
            self.get_logger().error(f'Pose model not found: {pose_model_path}')
            raise FileNotFoundError(f'Pose model not found: {pose_model_path}')
        self.get_logger().info(f'Using pose model: {pose_model_filename}')
        face_model_path = self.models_dir / "face_landmarker.task"
        
        base_hand_options = mp_tasks.BaseOptions(model_asset_path=str(hand_model_path))
        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=base_hand_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
        
        base_pose_options = mp_tasks.BaseOptions(model_asset_path=str(pose_model_path))
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=base_pose_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
        
        base_face_options = mp_tasks.BaseOptions(model_asset_path=str(face_model_path))
        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=base_face_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
        
        # Load TFLite model
        model_path = self.asl_models_dir / "model.tflite"
        if not model_path.exists():
            self.get_logger().error(f'TFLite model not found: {model_path}')
            raise FileNotFoundError(f'TFLite model not found: {model_path}')
        
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load character mapping
        char_to_num_path = self.asl_models_dir / "character_to_prediction_index.json"
        with open(char_to_num_path, 'r') as f:
            char_to_num = json.load(f)
        
        pad_token = 'P'
        start_token = 'S'
        end_token = 'E'
        n = len(char_to_num)
        char_to_num[pad_token] = n
        char_to_num[start_token] = n + 1
        char_to_num[end_token] = n + 2
        
        self.num_to_char = {j: i for i, j in char_to_num.items()}
        self.pad_token_id = char_to_num[pad_token]
        self.bos_token_id = char_to_num[start_token]
        self.eos_token_id = char_to_num[end_token]
        
        # Load inference args
        inference_args_path = self.asl_models_dir / 'inference_args.json'
        with open(inference_args_path, 'r') as f:
            inference_args = json.load(f)
        self.selected_columns = inference_args['selected_columns']
        
        # Config
        class SimpleConfig:
            pass
        self.cfg = SimpleConfig()
        self.cfg.n_landmarks = 130
        self.cfg.max_len_for_dummy = 15
        self.cfg.max_len = 384
        
    def image_callback(self, msg: Image):
        """Callback for image messages"""
        try:
            if not self.camera_connected:
                self.camera_connected = True
                self.get_logger().info('Camera connected successfully')
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Update timestamp
            self.frame_timestamp_ms = int(msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec / 1e6)
            
            # Process based on current state
            if self.state.current_state in (State.WAITING_FOR_PALM, State.OPEN_PALM_HOLD):
                self._check_open_palm(cv_image)
            elif self.state.current_state == State.COUNTDOWN:
                self._handle_countdown(cv_image)
            elif self.state.current_state in (
                State.RECORDING,
                State.RECORDING_READY_END,
                State.RECORDING_END_PALM_HOLD,
            ):
                self._handle_recording(cv_image)
                # Also check for palm detection during recording (for end detection)
                if self.state.min_frames_reached:
                    self._check_palm_during_recording(cv_image)
            self._publish_state_if_changed()
                
        except (ValueError, AttributeError, RuntimeError) as e:
            self.get_logger().error(f'Error in image callback: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error in image callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _check_open_palm(self, cv_image):
        """Check for open palm gesture"""
        now = time.time()
        if now < self._palm_rearm_until:
            self.state.palm_detection_start = None
            self.state.current_state = State.WAITING_FOR_PALM
            return

        # Performance optimization: skip frames to reduce CPU load
        self.gesture_check_counter += 1
        if self.gesture_check_counter < self.gesture_check_interval:
            return  # Skip gesture recognition this frame
        
        self.gesture_check_counter = 0
        
        # Debug: Log that gesture recognition is being performed
        if not hasattr(self, '_gesture_check_logged'):
            self.get_logger().debug('Performing gesture recognition check...')
            self._gesture_check_logged = True
        
        # MediaPipe can work with the image directly without copying
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
        
        gesture_result = self.gesture_recognizer.recognize_for_video(mp_image, self.frame_timestamp_ms)
        
        is_open_palm = False
        detected_gesture = None
        if gesture_result and gesture_result.gestures:
            for gesture_list in gesture_result.gestures:
                if gesture_list:
                    gesture = gesture_list[0]
                    detected_gesture = gesture.category_name.lower()
                    if detected_gesture == "open_palm":
                        is_open_palm = True
                        break
        
        # Debug: Log detected gesture (throttled to avoid spam)
        if not hasattr(self, '_last_gesture_log_time'):
            self._last_gesture_log_time = 0
        current_time = time.time()
        if current_time - self._last_gesture_log_time >= 3.0:  # Log every 3 seconds
            if detected_gesture:
                self.get_logger().debug(f'Detected gesture: {detected_gesture} (open_palm: {is_open_palm})')
            else:
                self.get_logger().debug('No gesture detected')
            self._last_gesture_log_time = current_time
        
        if is_open_palm:
            if self.state.palm_detection_start is None:
                self.state.palm_detection_start = time.time()
                self.get_logger().info(f'Open palm detected. Hold for {self.palm_detection_duration} seconds...')
            self.state.current_state = State.OPEN_PALM_HOLD
            elapsed = time.time() - self.state.palm_detection_start
            if elapsed >= self.palm_detection_duration:
                if not self._camera_expected_text:
                    # Continue pipeline even without camera-eval ground truth.
                    # Recognition scoring is skipped later when GT is not set.
                    self.get_logger().warning(
                        'Camera eval target is empty, but continuing to countdown without scoring.'
                    )
                self.get_logger().info('Open palm confirmed! Starting countdown...')
                self.state.current_state = State.COUNTDOWN
                self.state.countdown_start = time.time()
                self.state.palm_detection_start = None
        else:
            if self.state.palm_detection_start is not None:
                self.get_logger().info('Open palm lost. Waiting for open palm...')
            self.state.palm_detection_start = None
            self.state.current_state = State.WAITING_FOR_PALM
            self._last_palm_log_step = -1
    
    def _handle_countdown(self, cv_image):
        """Handle countdown state"""
        elapsed = time.time() - self.state.countdown_start
        remaining = max(0, self.countdown_duration - elapsed)
        
        if remaining <= 0:
            self.get_logger().info('Starting recording...')
            self.state.current_state = State.RECORDING
            self.state.recording_frames = []
            # Reset extended recording flags
            self.state.min_frames_reached = False
            self.state.palm_detection_end_start = None
            self.state.palm_detection_end_frames_start_idx = None
            # Reset gesture check counter for palm detection during recording
            self.gesture_check_counter = 0
        else:
            countdown_num = int(remaining) + 1
            self.get_logger().info(f'Countdown: {countdown_num}', throttle_duration_sec=1.0)
    
    def _handle_recording(self, cv_image):
        """Handle recording state - continue recording until palm is detected at the end"""
        self.state.recording_frames.append(cv_image.copy())
        
        # Check if minimum frames (384) are reached
        if len(self.state.recording_frames) >= self.num_frames and not self.state.min_frames_reached:
            self.state.min_frames_reached = True
            self.state.current_state = State.RECORDING_READY_END
            self.get_logger().info(f'Minimum {self.num_frames} frames reached. Continuing recording until palm is detected...')
    
    def _check_palm_during_recording(self, cv_image):
        """Check for palm detection during recording (after 384 frames) to determine end of recording"""
        # Performance optimization: skip frames to reduce CPU load
        self.gesture_check_counter += 1
        if self.gesture_check_counter < self.gesture_check_interval:
            return  # Skip gesture recognition this frame
        
        self.gesture_check_counter = 0
        
        # MediaPipe can work with the image directly without copying
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
        
        gesture_result = self.gesture_recognizer.recognize_for_video(mp_image, self.frame_timestamp_ms)
        
        is_open_palm = False
        if gesture_result and gesture_result.gestures:
            for gesture_list in gesture_result.gestures:
                if gesture_list:
                    gesture = gesture_list[0]
                    if gesture.category_name.lower() == "open_palm":
                        is_open_palm = True
                        break
        
        if is_open_palm:
            if self.state.palm_detection_end_start is None:
                # Start of palm detection at the end
                self.state.palm_detection_end_start = time.time()
                self.state.palm_detection_end_frames_start_idx = len(self.state.recording_frames) - 1
                self.state.current_state = State.RECORDING_END_PALM_HOLD
                self.get_logger().info(f'Palm detected. Hold for {self.palm_detection_duration} seconds...')
            else:
                # Check if palm has been detected for sufficient duration
                elapsed = time.time() - self.state.palm_detection_end_start
                if elapsed >= self.palm_detection_duration:
                    # End recording
                    self.get_logger().info(f'Recording complete. Collected {len(self.state.recording_frames)} frames. Processing...')
                    self._finish_recording()
        else:
            # Palm lost - reset detection
            if self.state.palm_detection_end_start is not None:
                self.get_logger().info('Palm lost during end detection. Continuing recording...', throttle_duration_sec=2.0)
            self.state.palm_detection_end_start = None
            self.state.palm_detection_end_frames_start_idx = None
            self.state.current_state = (
                State.RECORDING_READY_END if self.state.min_frames_reached else State.RECORDING
            )
            # Reset log step when palm is lost
            if hasattr(self, '_last_end_palm_log_step'):
                self._last_end_palm_log_step = -1
    
    def _finish_recording(self):
        """Finish recording, remove palm detection sections, and process frames"""
        # Copy recorded frames
        all_frames = self.state.recording_frames.copy()
        
        # Remove palm detection section at the end
        if self.state.palm_detection_end_frames_start_idx is not None:
            # Remove frames from palm detection start to the end
            frames_to_remove = len(all_frames) - self.state.palm_detection_end_frames_start_idx
            all_frames = all_frames[:self.state.palm_detection_end_frames_start_idx]
            self.get_logger().info(f'Removed {frames_to_remove} frames from end (palm detection section)')
        
        # Sample 384 frames from remaining frames (similar to video processing)
        total_frames = len(all_frames)
        if total_frames > self.num_frames:
            # Sample frames evenly
            frame_interval = total_frames / self.num_frames
            sampled_frames = []
            next_collect_frame = 0.0
            for frame_idx in range(total_frames):
                if frame_idx >= next_collect_frame and len(sampled_frames) < self.num_frames:
                    sampled_frames.append(all_frames[frame_idx])
                    next_collect_frame += frame_interval
            self.get_logger().info(f'Sampled {len(sampled_frames)} frames from {total_frames} frames')
            processed_frames = sampled_frames
        else:
            # Use all frames if less than 384
            processed_frames = all_frames
        
        # Reset state
        self.state.current_state = State.PREDICTING
        self.state.recorded_frames = processed_frames
        self.state.recording_frames = []
        self.state.min_frames_reached = False
        self.state.palm_detection_end_start = None
        self.state.palm_detection_end_frames_start_idx = None
        
        # Start prediction in separate thread
        prediction_thread = threading.Thread(
            target=self._run_prediction,
            args=(self.state.recorded_frames,)
        )
        prediction_thread.daemon = True
        prediction_thread.start()
    
    def _run_prediction(self, frames):
        """Run inference on recorded frames in separate thread"""
        try:
            with self.state.prediction_lock:
                self.state.is_predicting = True
            
            inference_start_time = time.time()
            self.get_logger().info('Running inference...')
            
            # Create new MediaPipe landmarker instances for this thread
            # MediaPipe landmarkers are not thread-safe, so we need separate instances
            hand_model_path = self.models_dir / "hand_landmarker.task"
            pose_model_filename = f"pose_landmarker_{self.pose_model}.task"
            pose_model_path = self.models_dir / pose_model_filename
            face_model_path = self.models_dir / "face_landmarker.task"
            
            base_hand_options = mp_tasks.BaseOptions(model_asset_path=str(hand_model_path))
            hand_options = mp_vision.HandLandmarkerOptions(
                base_options=base_hand_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            inference_hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
            
            base_pose_options = mp_tasks.BaseOptions(model_asset_path=str(pose_model_path))
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=base_pose_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            inference_pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
            
            base_face_options = mp_tasks.BaseOptions(model_asset_path=str(face_model_path))
            face_options = mp_vision.FaceLandmarkerOptions(
                base_options=base_face_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            inference_face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
            
            # Extract landmarks
            landmark_buffer = LandmarkBuffer(max_frames=len(frames), selected_columns=self.selected_columns)
            
            with ExitStack() as stack:
                stack.enter_context(inference_hand_landmarker)
                stack.enter_context(inference_pose_landmarker)
                stack.enter_context(inference_face_landmarker)
                
                for frame_idx, frame in enumerate(frames):
                    # MediaPipe can work with the frame directly without copying
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    timestamp_ms = int((frame_idx + 1) * (1000 / 30))
                    
                    pose_result = inference_pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                    face_result = inference_face_landmarker.detect_for_video(mp_image, timestamp_ms)
                    hand_result = inference_hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    pose_landmarks = (
                        pose_result.pose_landmarks[0] if pose_result and pose_result.pose_landmarks else None
                    )
                    face_landmarks = (
                        face_result.face_landmarks[0] if face_result and face_result.face_landmarks else None
                    )
                    left_hand_landmarks, right_hand_landmarks = split_hand_landmarks(hand_result, mirror=False)
                    
                    landmarks_dict = extract_landmarks_to_dict(
                        pose_landmarks,
                        face_landmarks,
                        left_hand_landmarks,
                        right_hand_landmarks,
                        self.selected_columns,
                    )
                    
                    landmark_buffer.add_frame(landmarks_dict)
            
            # Get sequence
            sequence = landmark_buffer.get_sequence()
            if sequence is None:
                raise RuntimeError("No landmarks extracted")
            
            # Run TFLite inference
            prediction = self._run_tflite_inference(sequence)
            
            inference_end_time = time.time()
            inference_duration = inference_end_time - inference_start_time
            self.get_logger().info(f'Inference completed in {inference_duration:.2f} seconds')
            self._publish_asl_timing("camera_inference", inference_start_time, inference_end_time)
            self._publish_asl_report("camera_inference", inference_start_time, inference_end_time, result="ok")
            
            with self.state.prediction_lock:
                self.state.prediction_result = prediction
                self.state.is_predicting = False
                self.state.current_state = State.SHOWING_RESULT
                self._result_display_until = time.time() + self.result_display_duration_sec
            self._publish_state_if_changed()
            self._show_result_popup(prediction)
            
            # Publish result
            self._publish_result(prediction)
            
        except RuntimeError as e:
            self.get_logger().error(f'Runtime error in prediction: {e}')
            with self.state.prediction_lock:
                self.state.prediction_result = f"Error: {str(e)}"
                self.state.is_predicting = False
                self.state.current_state = State.SHOWING_RESULT
                self._result_display_until = time.time() + self.result_display_duration_sec
            self._publish_state_if_changed()
            self._show_result_popup(self.state.prediction_result)
        except Exception as e:
            self.get_logger().error(f'Unexpected error in prediction: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            with self.state.prediction_lock:
                self.state.prediction_result = f"Error: {str(e)}"
                self.state.is_predicting = False
                self.state.current_state = State.SHOWING_RESULT
                self._result_display_until = time.time() + self.result_display_duration_sec
            self._publish_state_if_changed()
            self._show_result_popup(self.state.prediction_result)
    
    def _run_tflite_inference(self, sequence: np.ndarray) -> str:
        """Run TFLite inference on landmark sequence and return prediction"""
        expected_shape = self.input_details[0]['shape']
        seq_len = sequence.shape[0]
        input_data = sequence.astype(np.float32)
        
        # Resize tensor if needed
        if len(expected_shape) == 2 and expected_shape[0] != seq_len:
            try:
                self.interpreter.resize_tensor_input(self.input_details[0]['index'], [seq_len, expected_shape[1]])
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                expected_shape = self.input_details[0]['shape']
            except (ValueError, RuntimeError) as e:
                self.get_logger().warning(f'Failed to resize tensor: {e}')
        
        # Final shape adjustment
        if len(expected_shape) == 2:
            if seq_len < expected_shape[0]:
                padding = np.zeros((expected_shape[0] - seq_len, input_data.shape[1]), dtype=input_data.dtype)
                input_data = np.concatenate([input_data, padding], axis=0)
            elif seq_len > expected_shape[0]:
                input_data = input_data[:expected_shape[0], :]
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Decode prediction
        prediction = decode_prediction(
            output_scores, self.num_to_char, self.pad_token_id, self.bos_token_id, self.eos_token_id
        )
        
        # Check for dummy score
        is_dummy_score = False
        if seq_len < self.cfg.max_len_for_dummy:
            is_dummy_score = True
        
        if prediction.strip():
            pred_lower = prediction.strip().lower()
            if (pred_lower.startswith('2 ') and len(pred_lower) < 10) or \
               (pred_lower.startswith('2 a-') or pred_lower.startswith('2 a e')):
                is_dummy_score = True
        
        if is_dummy_score:
            prediction = ""
        
        return prediction
    
    def _publish_result(self, text):
        """Publish recognition result"""
        msg = RecognizedText()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.text = text
        msg.confidence = 1.0  # TODO: calculate actual confidence
        
        self.recognized_text_pub.publish(msg)
        self.get_logger().info(f'Published result: "{text}"')
        if self.operation_mode == 'camera' and self._camera_expected_text:
            gt = self._camera_expected_text
            pred = self._normalize_sentence(text)
            cer = self._character_error_rate(gt, pred)
            self.get_logger().info(f'Camera recognition eval: gt="{gt}" pred="{pred}" cer={cer:.4f}')
            self._publish_recognition_eval("camera", gt, pred, cer)

    def _show_result_popup(self, text: str) -> None:
        """Show recognition result in a dedicated popup window for a fixed duration."""
        if self.operation_mode != 'camera' or self.result_display_duration_sec <= 0.0:
            return
        if tk is None:
            self.get_logger().warning('Result popup unavailable (tkinter not installed).')
            return

        def _popup_worker() -> None:
            root = tk.Tk()
            root.title('ASL Recognition Result')
            root.attributes('-topmost', True)
            root.geometry('900x240')
            root.configure(bg='black')

            title = tk.Label(
                root,
                text='Recognized Command',
                fg='white',
                bg='black',
                font=('Arial', 20, 'bold'),
            )
            title.pack(pady=(20, 8))

            value = tk.Label(
                root,
                text=(text or '(empty)'),
                fg='#00d4ff',
                bg='black',
                font=('Arial', 30, 'bold'),
                wraplength=860,
                justify='center',
            )
            value.pack(padx=20, pady=(0, 20), fill='both', expand=True)

            root.after(int(self.result_display_duration_sec * 1000), root.destroy)
            root.mainloop()

        threading.Thread(target=_popup_worker, daemon=True).start()
    
    def timer_callback(self):
        """Timer callback for state machine (only active in camera mode)"""
        if self.operation_mode != 'camera':
            return
        
        # Reset to waiting state immediately after showing result
        if self.state.current_state == State.SHOWING_RESULT:
            if time.time() < self._result_display_until:
                self._publish_state_if_changed()
                return
            self.state.current_state = State.WAITING_FOR_PALM
            self.state.prediction_result = ""
            self.state.palm_detection_start = None
            self._palm_rearm_until = time.time() + self.palm_rearm_delay_sec
            self._camera_expected_text = None
            self._prepare_next_camera_ground_truth()
            self.get_logger().debug('Reset to waiting state')
        self._publish_state_if_changed()
    
    def process_video_callback(self, request: ProcessVideo.Request, response: ProcessVideo.Response):
        """Service callback for video processing"""
        video_path_str = request.video_path.strip()
        
        if not video_path_str:
            # Process all videos in folder
            self.get_logger().info('Service call: process all videos in folder')
            threading.Thread(target=self._process_all_videos_in_folder, daemon=True).start()
            response.success = True
            response.recognized_text = "Processing all videos in folder..."
            return response
        
        # Process single video
        video_path = Path(video_path_str)
        if not video_path.exists():
            response.success = False
            response.error_message = f"Video file not found: {video_path}"
            self.get_logger().error(f'{response.error_message}')
            return response
        
        self.get_logger().info(f'Service call: process video {video_path.name}')
        
        # Process in thread to avoid blocking
        result = self._process_single_video(str(video_path))
        
        if result:
            response.success = True
            response.recognized_text = result
            response.confidence = 1.0
        else:
            response.success = False
            response.error_message = "No recognition result"
        
        return response
    
    def _ground_truth_from_filename(self, video_path: Path) -> str:
        """Extract ground truth from video filename (e.g. hello_world.mp4 -> 'hello world')."""
        return video_path.stem.replace("_", " ").strip().lower()

    def _normalize_sentence(self, text: str) -> str:
        """Normalize sentence for robust comparison."""
        s = " ".join((text or "").strip().lower().split())
        # Some datasets/templates may include trailing punctuation (e.g. "bed.").
        # Remove it so exact-match isn't overly sensitive.
        return s.strip(".,;:!?")

    def _normalize_ground_truth_input(self, text: str) -> str:
        """Normalize popup input where underscores represent spaces."""
        return self._normalize_sentence((text or "").replace("_", " "))

    @staticmethod
    def _character_error_rate(reference: str, hypothesis: str) -> float:
        """Compute CER = Levenshtein distance / max(1, len(reference))."""
        ref = reference or ""
        hyp = hypothesis or ""
        n = len(ref)
        m = len(hyp)
        if n == 0:
            return 0.0 if m == 0 else 1.0
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[n][m] / float(n)

    def _publish_recognition_eval(
        self,
        mode: str,
        ground_truth: str,
        prediction: str,
        cer: float,
        execution_success: Optional[int] = None,
        stop_reason: str = "",
    ) -> None:
        """
        Backward-compatible wrapper: recognition eval is now merged into ASL REPORT.
        """
        self._publish_asl_report(
            f"{mode}_recognition_eval",
            time.time(),
            time.time(),
            result="ok" if execution_success in (None, 1) else "failed",
            mode=mode,
            ground_truth=ground_truth.replace(" ", "_"),
            prediction=prediction.replace(" ", "_"),
            cer=f"{cer:.6f}",
            execution_success=execution_success if execution_success is not None else "",
            stop_reason=stop_reason.replace(" ", "_") if stop_reason else "",
        )

    def _prompt_camera_ground_truth(self) -> Tuple[Optional[str], bool]:
        """Open popup for camera-mode ground truth input.

        Returns:
            (normalized_text, cancelled)
            - cancelled=True when the user explicitly cancels/closes the popup.
        """
        if tk is None or simpledialog is None:
            self.get_logger().warning(
                "Camera evaluation popup unavailable (tkinter not installed)."
            )
            return None, False
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            value = simpledialog.askstring(
                "Camera Evaluation Input",
                "Enter expected command (use underscores, e.g. go_forward):",
                parent=root,
            )
        finally:
            root.destroy()
        if value is None:
            return None, True
        normalized = self._normalize_ground_truth_input(value or "")
        return (normalized or None), False

    def _prepare_next_camera_ground_truth(self) -> None:
        """Request next expected text for camera-mode evaluation."""
        if self.operation_mode != "camera":
            return
        gt, cancelled = self._prompt_camera_ground_truth()
        if cancelled:
            self.get_logger().warning(
                "Camera eval popup was cancelled. Shutting down asl_recognition_node."
            )
            self._publish_state_if_changed("pipeline_failed", force=True)
            self.should_shutdown = True
            return
        if gt:
            self._camera_expected_text = gt
            self.get_logger().info(f'Camera eval target set: "{gt}"')
        else:
            self._camera_expected_text = None
            self.get_logger().warning("Camera eval target not set. Prediction will not be scored.")

    def _ground_truth_from_path(self, video_path: Path) -> str:
        """Extract ground truth from parent folder name if available, fallback to filename."""
        try:
            parent = video_path.parent
            if parent and parent != self.video_input_dir:
                return self._normalize_sentence(parent.name.replace("_", " "))
        except Exception:
            pass
        return self._normalize_sentence(self._ground_truth_from_filename(video_path))
    
    def _format_comparison(self, ground_truth: str, prediction: str, width: int = 60) -> str:
        """Format ground truth and prediction for side-by-side comparison."""
        lines = []
        lines.append("─" * width)
        lines.append(f"{'Ground Truth':<30} │ {'Prediction':<30}")
        lines.append("─" * width)
        gt_lines = [ground_truth[i:i+28] for i in range(0, len(ground_truth), 28)]
        pred_lines = [prediction[i:i+28] for i in range(0, len(prediction), 28)]
        max_lines = max(len(gt_lines), len(pred_lines)) or 1
        for i in range(max_lines):
            gt_part = gt_lines[i] if i < len(gt_lines) else ""
            pred_part = pred_lines[i] if i < len(pred_lines) else ""
            lines.append(f"{gt_part:<30} │ {pred_part:<30}")
        lines.append("─" * width)
        return "\n".join(lines)
    
    def _process_all_videos_in_folder(self, log_comparison: bool = False):
        """Process all videos recursively and report recognition/downstream evaluation."""
        self._publish_state_if_changed("video_waiting", force=True)
        if not self.video_input_dir.exists():
            self.get_logger().warning(f'Video input directory does not exist: {self.video_input_dir}')
            return
        
        self.video_input_dir.mkdir(parents=True, exist_ok=True)
        video_files = sorted(
            list(self.video_input_dir.rglob("*.mp4")) +
            list(self.video_input_dir.rglob("*.MP4")) +
            list(self.video_input_dir.rglob("*.avi")) +
            list(self.video_input_dir.rglob("*.AVI"))
        )
        
        if not video_files:
            self.get_logger().info(f'No video files found in {self.video_input_dir}')
            return

        self.get_logger().info(f'Found {len(video_files)} video(s) to process')
        total = 0
        total_cer = 0.0
        downstream_total = 0
        downstream_success = 0

        for idx, video_path in enumerate(video_files, 1):
            if str(video_path) in self.processed_videos:
                self.get_logger().info(f'Skipping already processed: {video_path.name}')
                continue

            self.get_logger().info(f'[{idx}/{len(video_files)}] Processing: {video_path.name}')
            self._publish_state_if_changed("video_processing")
            result = self._process_single_video(str(video_path))

            # Small delay between videos to ensure MediaPipe landmarkers are fully cleaned up
            if idx < len(video_files):
                import time
                time.sleep(1.0)  # Wait 1 second between videos
            pred = self._normalize_sentence(result or "")
            gt = self._ground_truth_from_path(video_path)
            cer = self._character_error_rate(gt, pred)
            total += 1
            total_cer += cer

            if log_comparison:
                cmp_text = f'Comparison for {video_path.name}:\n{self._format_comparison(gt, pred)}'
                self.get_logger().info(cmp_text)
            else:
                if result:
                    self.get_logger().info(f'Video {video_path.name} result: "{result}"')
                else:
                    self.get_logger().warning(f'Video {video_path.name}: No recognition result')

            self.get_logger().info(
                f'Recognition eval: cer={cer:.4f}'
            )
            self._publish_recognition_eval("video", gt, pred, cer)
            if pred:
                downstream_total += 1
                self._publish_state_if_changed("video_waiting_plan")
                wait_kind, wait_reason = self._wait_for_plan_done()
                if wait_kind == "execution_done":
                    downstream_success += 1
                    self.get_logger().info('Downstream eval: EXECUTION DONE received')
                elif wait_kind == "plan_skipped":
                    self.get_logger().warning(
                        f'Downstream eval: plan skipped ({wait_reason}); mark failure and move to next sample'
                    )
                    self._publish_recognition_eval(
                        "video",
                        gt,
                        pred,
                        cer,
                        execution_success=0,
                        stop_reason=f"plan_skipped:{wait_reason}",
                    )
                    time.sleep(1.0)
                else:
                    self.get_logger().warning('Downstream eval: timeout waiting for EXECUTION DONE')
            else:
                self.get_logger().info(
                    'Downstream eval: skipped EXECUTION DONE wait because recognition result is empty'
                )

        if total == 0:
            return

        avg_cer = total_cer / total
        self.get_logger().info(
            f'Video recognition summary: samples={total}, avg_cer={avg_cer:.4f}'
        )
        downstream_acc = (downstream_success / downstream_total) * 100.0 if downstream_total else 0.0
        self.get_logger().info(
            f'Video downstream summary: execution_done={downstream_success}/{downstream_total} ({downstream_acc:.1f}%)'
        )
        self._publish_state_if_changed("pipeline_done", force=True)

    def _process_all_videos_in_folder_and_exit(self, log_comparison: bool = False):
        """Process all videos in folder and exit program when done"""
        self._process_all_videos_in_folder(log_comparison=log_comparison)
        self.get_logger().info('All videos processed. Exiting...')
        # Schedule shutdown after a short delay
        time.sleep(0.5)
        self.should_shutdown = True
    
    def _process_single_video(self, video_path_str: str) -> Optional[str]:
        """Process a single video file and return recognition result"""
        video_path = Path(video_path_str)
        
        with self.video_processing_lock:
            if str(video_path) in self.processed_videos:
                return None
        
        try:
            # Start timing from video file loading
            inference_start_time = time.time()
            
            cap, video_meta = open_video_capture_with_orientation(video_path)
            if not cap.isOpened():
                self.get_logger().error(f'Cannot open video: {video_path}')
                return None
            
            if video_meta:
                orient_msg = (
                    'Video orientation: '
                    f"meta={video_meta.get('orientation_meta')}, "
                    f"CAP_PROP_ORIENTATION_AUTO before->after="
                    f"{video_meta.get('orientation_auto_before')}->"
                    f"{video_meta.get('orientation_auto_after')}"
                )
                self.get_logger().info(orient_msg)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vinfo = f'Video: {video_path.name} ({total_frames} frames @ {fps:.2f} FPS)'
            self.get_logger().info(vinfo)
            
            # Calculate frame sampling if video is longer than max_len
            if total_frames > self.cfg.max_len:
                frame_interval = total_frames / self.cfg.max_len
                samp = f'Sampling {self.cfg.max_len} frames from {total_frames} frames'
                self.get_logger().info(samp)
            else:
                frame_interval = 1.0
            
            # Load MediaPipe models (same as original script)
            # Create new MediaPipe landmarker instances for this video processing
            # MediaPipe landmarkers are not thread-safe, so we need separate instances
            hand_model_path = self.models_dir / "hand_landmarker.task"
            pose_model_filename = f"pose_landmarker_{self.pose_model}.task"
            pose_model_path = self.models_dir / pose_model_filename
            face_model_path = self.models_dir / "face_landmarker.task"
            
            # Hand landmarker
            base_hand_options = mp_tasks.BaseOptions(model_asset_path=str(hand_model_path))
            hand_options = mp_vision.HandLandmarkerOptions(
                base_options=base_hand_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            video_hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
            
            # Pose landmarker
            base_pose_options = mp_tasks.BaseOptions(model_asset_path=str(pose_model_path))
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=base_pose_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            video_pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
            
            # Face landmarker
            base_face_options = mp_tasks.BaseOptions(model_asset_path=str(face_model_path))
            face_options = mp_vision.FaceLandmarkerOptions(
                base_options=base_face_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            video_face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
            
            # Extract landmarks from video
            landmark_buffer = LandmarkBuffer(max_frames=self.cfg.max_len, selected_columns=self.selected_columns)
            frame_buffer = deque(maxlen=self.cfg.max_len)
            
            frame_idx = 0
            collected_frame_count = 0
            next_collect_frame = 0.0
            
            # Use ExitStack to manage landmarker lifecycle (same as original script)
            with ExitStack() as stack:
                stack.enter_context(video_hand_landmarker)
                stack.enter_context(video_pose_landmarker)
                stack.enter_context(video_face_landmarker)
                
                self.get_logger().info(f'Extracting landmarks from {video_path.name}...')
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame is None or frame.size == 0:
                        frame_idx += 1
                        continue
                    
                    should_collect = (frame_idx >= next_collect_frame) and (len(frame_buffer) < self.cfg.max_len)
                    
                    if should_collect:
                        try:
                            next_collect_frame += frame_interval
                            
                            # Convert BGR to RGB for MediaPipe (in-place conversion not possible, so copy is needed)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            timestamp_ms = int((collected_frame_count + 1) * (1000 / fps))
                            
                            pose_result = video_pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                            face_result = video_face_landmarker.detect_for_video(mp_image, timestamp_ms)
                            hand_result = video_hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                            
                            pose_landmarks = (
                                pose_result.pose_landmarks[0] if pose_result and pose_result.pose_landmarks else None
                            )
                            face_landmarks = (
                                face_result.face_landmarks[0] if face_result and face_result.face_landmarks else None
                            )
                            left_hand_landmarks, right_hand_landmarks = split_hand_landmarks(hand_result, mirror=False)
                            
                            landmarks_dict = extract_landmarks_to_dict(
                                pose_landmarks,
                                face_landmarks,
                                left_hand_landmarks,
                                right_hand_landmarks,
                                self.selected_columns,
                            )
                            
                            frame_buffer.append(frame.copy())
                            landmark_buffer.add_frame(landmarks_dict)
                            collected_frame_count += 1
                            
                            if collected_frame_count % 50 == 0:
                                self.get_logger().info(f'Processed {collected_frame_count}/{self.cfg.max_len} frames...')
                        except (ValueError, AttributeError, RuntimeError) as e:
                            self.get_logger().error(f'Error processing frame {frame_idx}: {str(e)}')
                            # Continue with next frame
                    
                    frame_idx += 1
                
                # Release video capture before ExitStack closes landmarkers
                cap.release()
            
            lm_done = f'Landmark extraction completed. Collected {collected_frame_count} frames.'
            self.get_logger().info(lm_done)
            run_inf = f'Running inference on {video_path.name}...'
            self.get_logger().info(run_inf)
            
            # Run inference
            sequence = landmark_buffer.get_sequence()
            if sequence is None:
                self.get_logger().error(f'No landmarks extracted from {video_path.name}')
                return None
            
            # Run TFLite inference
            prediction = self._run_tflite_inference(sequence)
            
            inference_end_time = time.time()
            inference_duration = inference_end_time - inference_start_time
            inf_done = f'Inference completed in {inference_duration:.2f} seconds for {video_path.name}'
            self.get_logger().info(inf_done)
            self._publish_asl_timing(
                "video_inference",
                inference_start_time,
                inference_end_time,
                sample=video_path.name.replace(" ", "_"),
            )
            self._publish_asl_report(
                "video_inference",
                inference_start_time,
                inference_end_time,
                result="ok",
                sample=video_path.name.replace(" ", "_"),
            )
            
            # Mark as processed
            with self.video_processing_lock:
                self.processed_videos.add(str(video_path))
            
            # Publish result
            if prediction:
                self._publish_result(prediction)
                pub = f'Published result: "{prediction}"'
                self.get_logger().info(pub)
                self.get_logger().info(f'Video {video_path.name} result: "{prediction}"')
            
            return prediction if prediction else None
            
        except (FileNotFoundError, IOError) as e:
            self.get_logger().error(f'File error processing video {video_path.name}: {e}')
            return None
        except (RuntimeError, ValueError) as e:
            self.get_logger().error(f'Processing error for video {video_path.name}: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'Unexpected error processing video {video_path.name}: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None


def main(args=None):
    rclpy.init(args=args)
    node = ASLRecognitionNode()
    
    try:
        # Spin until node requests shutdown or interrupted
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
