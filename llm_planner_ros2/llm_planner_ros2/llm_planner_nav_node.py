#!/usr/bin/env python3
"""ROS2 node: recognized text → navigation target pose."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import String
from sign_language_msgs.msg import RecognizedText

from .robot_motion_planner import run_pipeline, OLLAMA_MODEL

# Match asl_recognition_node publisher so the first text-mode publish is not lost if the
# planner process starts slightly later than the ASL process.
_RECOGNIZED_TEXT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

PIPELINE_STATUS_TOPIC = "/pipeline/status"
PLANNER_STATUS_TOPIC = "/pipeline/planner/status"
PLANNER_TIMING_TOPIC = "/pipeline/planner/timing"
PLANNER_REPORT_TOPIC = "/pipeline/planner/report"


class LlmPlannerNavNode(Node):
    """Convert recognized text into a validated motion plan via llm_planner."""

    def __init__(self) -> None:
        super().__init__("llm_planner_nav_node")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter(
                "use_sim_time",
                False,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_BOOL,
                    description="Use /clock when true (simulation); false on the real robot.",
                ),
            )
        self.declare_parameter(
            "ollama_model",
            OLLAMA_MODEL,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Ollama model name used by llm_planner (e.g. gemma3n:e4b).",
            ),
        )
        self.declare_parameter(
            "recognized_text_topic",
            "/asl_recognition_node/recognized_text",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Input topic with recognized text (sign_language_msgs/RecognizedText).",
            ),
        )
        self.declare_parameter(
            "target_frame",
            "map",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='TF frame id for target poses (e.g. "map").',
            ),
        )

        self.ollama_model: str = self.get_parameter("ollama_model").value.strip()
        recognized_text_topic_param = self.get_parameter("recognized_text_topic").value.strip()
        self.target_frame: str = self.get_parameter("target_frame").value.strip() or "map"

        recognized_text_topic = recognized_text_topic_param or "/asl_recognition_node/recognized_text"

        self.recognized_text_sub = self.create_subscription(
            RecognizedText,
            recognized_text_topic,
            self.recognized_text_callback,
            _RECOGNIZED_TEXT_QOS,
        )
        self.get_logger().info(f"[LLM Planner Nav] Subscribing to: {recognized_text_topic}")

        self.normalized_text_pub = self.create_publisher(
            RecognizedText,
            "/llm_planner/normalized_text",
            10,
        )

        # Full motion plan as JSON string; consumed by an executor node
        self.motion_plan_pub = self.create_publisher(
            String,
            "/llm_planner/motion_plan",
            10,
        )

        self.planner_timing_pub = self.create_publisher(
            String,
            PLANNER_TIMING_TOPIC,
            10,
        )
        self.planner_report_pub = self.create_publisher(
            String,
            PLANNER_REPORT_TOPIC,
            10,
        )
        self.pipeline_status_pub = self.create_publisher(String, PIPELINE_STATUS_TOPIC, 10)
        self.planner_status_pub = self.create_publisher(String, PLANNER_STATUS_TOPIC, 10)

        self.get_logger().info(f"[LLM Planner] Initialized with Ollama model: {self.ollama_model}")

    def _publish_pipeline_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.pipeline_status_pub.publish(msg)
        self.planner_status_pub.publish(msg)

    def recognized_text_callback(self, msg: RecognizedText) -> None:
        """Handle recognized text and run planning pipeline."""
        text = (msg.text or "").strip()
        if not text:
            return

        self.get_logger().info(f'[LLM Planner] Received text: "{text}"')
        self._publish_pipeline_status("llm_planning")
        t_in = self.get_clock().now().nanoseconds / 1e9

        try:
            result = run_pipeline(text, model=self.ollama_model)
        except Exception as e:
            self.get_logger().error(f"[LLM Planner] Exception in run_pipeline: {e}")
            self._publish_pipeline_status("pipeline_failed")
            self._publish_planner_status_only(f"plan_skipped:planner_exception")
            self._publish_planner_report_skipped("planner_exception", t_in=t_in, t_done=self.get_clock().now().nanoseconds / 1e9)
            return

        # Debug information to understand why a plan was rejected
        if result.error:
            self.get_logger().warn(f"[LLM Planner] Pipeline error field: {result.error}")
        if result.clarification:
            self.get_logger().warn(
                f"[LLM Planner] Clarification requested: {result.clarification.get('question', '')}"
            )
        self.get_logger().info(
            f"[LLM Planner] Plan length after validation: {len(result.plan)}; "
            f"validation_messages={len(result.validation_messages)}"
        )

        # Publish LLM timing info for analysis (even if plan is empty, to see failures)
        total = result.total_time_s
        t_done = t_in + total
        timing_msg = String()
        timing_msg.data = (
            f"LLM TIMING t_in={t_in:.3f} t_done={t_done:.3f} "
            f"total={result.total_time_s:.3f} "
            f"norm={result.normalization_time_s:.3f} "
            f"plan={result.planning_time_s:.3f} "
            f"valid={result.validation_time_s:.3f}"
        )
        self.planner_timing_pub.publish(timing_msg)
        self._publish_planner_report(
            t_in=t_in,
            t_done=t_done,
            result="ok" if not result.error and not result.clarification and bool(result.plan) else "skipped",
            plan_steps=len(result.plan),
        )

        # Skip invalid inputs: no outputs when pipeline fails or needs clarification
        if result.error or result.clarification or not result.plan:
            self.get_logger().info("[LLM Planner] Input could not produce a valid plan. Skipping output.")
            self._publish_pipeline_status("pipeline_failed")
            if result.clarification:
                skip_reason = "clarification_required"
            elif result.error:
                skip_reason = "pipeline_error"
            else:
                skip_reason = "no_valid_plan"
            self._publish_planner_status_only(f"plan_skipped:{skip_reason}")
            self._publish_planner_report_skipped(skip_reason, t_in=t_in, t_done=t_done)
            return

        self._publish_normalized_text(result.normalized_text, msg)
        self._publish_motion_plan(result.plan)
        self._publish_pipeline_status("plan_ready")

    def _publish_normalized_text(self, normalized_text: str, src_msg: RecognizedText) -> None:
        out = RecognizedText()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = src_msg.header.frame_id or "llm_planner_nav"
        out.text = normalized_text
        out.confidence = src_msg.confidence if src_msg.confidence > 0.0 else 1.0
        self.normalized_text_pub.publish(out)
        self.get_logger().info(f'[LLM Planner] Published normalized text: "{normalized_text}"')

    def _publish_motion_plan(self, plan: List[Dict[str, Any]]) -> None:
        """Publish full motion plan as JSON string."""
        msg = String()
        msg.data = json.dumps(plan, ensure_ascii=False)
        self.motion_plan_pub.publish(msg)
        self.get_logger().info(f"[LLM Planner] Published motion plan with {len(plan)} steps.")

    def _publish_planner_report(self, *, t_in: float, t_done: float, result: str, plan_steps: int) -> None:
        msg = String()
        msg.data = (
            f"PLANNER REPORT t_in={t_in:.3f} t_done={t_done:.3f} total={max(0.0, t_done - t_in):.3f} "
            f"result={result} plan_steps={int(plan_steps)}"
        )
        self.planner_report_pub.publish(msg)

    def _publish_planner_status_only(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.planner_status_pub.publish(msg)

    def _publish_planner_report_skipped(self, reason: str, *, t_in: float, t_done: float) -> None:
        msg = String()
        msg.data = (
            f"PLANNER REPORT t_in={t_in:.3f} t_done={t_done:.3f} total={max(0.0, t_done - t_in):.3f} "
            f"result=plan_skipped reason={reason} plan_steps=0"
        )
        self.planner_report_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LlmPlannerNavNode()
    try:
        rclpy.spin(node)
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

