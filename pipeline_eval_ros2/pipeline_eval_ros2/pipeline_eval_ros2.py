#!/usr/bin/env python3
"""ROS2 evaluation recorder for ASL->LLM->execution pipeline."""

from __future__ import annotations

import csv
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import psutil
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sign_language_msgs.msg import RecognizedText
from std_msgs.msg import String


_KV_RE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")

# Match asl_recognition_node publisher (TRANSIENT_LOCAL) for recognized_text.
_RECOGNIZED_TEXT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

RECOGNIZED_TEXT_TOPIC = "/asl_recognition_node/recognized_text"
EXECUTION_STATUS_TOPIC = "/pipeline/executor/status"
PLANNER_TIMING_TOPIC = "/pipeline/planner/timing"
EXECUTOR_TIMING_TOPIC = "/pipeline/executor/timing"
ASL_TIMING_TOPIC = "/pipeline/asl/timing"
ASL_REPORT_TOPIC = "/pipeline/asl/report"
PLANNER_REPORT_TOPIC = "/pipeline/planner/report"
EXECUTOR_REPORT_TOPIC = "/pipeline/executor/report"
PIPELINE_STATUS_TOPIC = "/pipeline/status"


@dataclass
class PlanRecord:
    """Per-plan temporary state before writing CSV row."""

    text: str = ""
    t_text: Optional[float] = None
    t_plan: Optional[float] = None
    t_done: Optional[float] = None
    num_steps: int = 0
    llm_total: float = 0.0
    llm_norm: float = 0.0
    llm_plan: float = 0.0
    llm_valid: float = 0.0
    gt_text: str = ""
    pred_text: str = ""
    cer: Optional[float] = None
    plan_success: Optional[int] = None
    execution_success: Optional[int] = None
    pipeline_status: str = ""
    target_distance_m: Optional[float] = None
    actual_distance_m: Optional[float] = None
    distance_error_m: Optional[float] = None
    timeout_flag: int = 0
    stop_reason: str = ""
    asl_mode: str = ""
    asl_stage: str = ""
    asl_t_in: Optional[float] = None
    asl_t_done: Optional[float] = None
    asl_total: Optional[float] = None
    asl_report_stage: str = ""
    asl_report_result: str = ""
    asl_report_total: Optional[float] = None
    planner_report_result: str = ""
    planner_report_total: Optional[float] = None
    executor_report_result: str = ""
    executor_report_total: Optional[float] = None


def _parse_kv(message: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in _KV_RE.finditer(message or "")}


class PipelineEvalRos2Node(Node):
    """Subscribe planner/executor streams and append evaluation CSV rows."""

    def __init__(self) -> None:
        super().__init__("pipeline_eval_ros2")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.declare_parameter("output_csv", "~/ros2_ws/pipeline_eval_metrics.csv")

        self.recognized_text_topic = RECOGNIZED_TEXT_TOPIC
        self.execution_status_topic = EXECUTION_STATUS_TOPIC
        self.planner_timing_topic = PLANNER_TIMING_TOPIC
        self.executor_timing_topic = EXECUTOR_TIMING_TOPIC
        self.asl_timing_topic = ASL_TIMING_TOPIC
        self.asl_report_topic = ASL_REPORT_TOPIC
        self.planner_report_topic = PLANNER_REPORT_TOPIC
        self.executor_report_topic = EXECUTOR_REPORT_TOPIC
        self.pipeline_status_topic = PIPELINE_STATUS_TOPIC
        self.output_csv = Path(str(self.get_parameter("output_csv").value)).expanduser()

        self._lock = threading.Lock()
        self._current: PlanRecord = PlanRecord()
        self._plan_seq = 0
        self._last_pipeline_status: str = ""

        self._prepare_csv_files()

        self.create_subscription(
            RecognizedText,
            self.recognized_text_topic,
            self._recognized_text_callback,
            _RECOGNIZED_TEXT_QOS,
        )
        self.create_subscription(
            String,
            self.execution_status_topic,
            self._execution_events_callback,
            10,
        )
        self.create_subscription(
            String,
            self.planner_timing_topic,
            self._planner_timing_callback,
            10,
        )
        self.create_subscription(
            String,
            self.executor_timing_topic,
            self._executor_timing_callback,
            10,
        )
        self.create_subscription(
            String,
            self.asl_timing_topic,
            self._asl_timing_callback,
            10,
        )
        self.create_subscription(
            String,
            self.asl_report_topic,
            self._asl_report_callback,
            10,
        )
        self.create_subscription(
            String,
            self.planner_report_topic,
            self._planner_report_callback,
            10,
        )
        self.create_subscription(
            String,
            self.executor_report_topic,
            self._executor_report_callback,
            10,
        )
        self.create_subscription(
            String,
            self.pipeline_status_topic,
            self._pipeline_status_callback,
            10,
        )

        self.get_logger().info(
            f"Pipeline eval started. output_csv={self.output_csv} "
            f"pipeline_status_topic={self.pipeline_status_topic} "
            f"asl_timing_topic={self.asl_timing_topic}"
        )

    def _migrate_csv_add_eval_columns(self) -> None:
        """Add newly introduced evaluation columns to existing CSV if missing."""
        if not self.output_csv.exists():
            return
        with self.output_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        required = [
            "pipeline_status",
            "target_distance_m",
            "actual_distance_m",
            "distance_error_m",
            "timeout_flag",
            "stop_reason",
            "asl_mode",
            "asl_stage",
            "asl_t_in",
            "asl_t_done",
            "asl_total",
            "asl_report_stage",
            "asl_report_result",
            "asl_report_total",
            "planner_report_result",
            "planner_report_total",
            "executor_report_result",
            "executor_report_total",
        ]
        missing = [name for name in required if name not in fieldnames]
        if not missing:
            return
        new_fieldnames = fieldnames + missing
        for row in rows:
            for name in missing:
                if name == "timeout_flag":
                    row[name] = 0
                else:
                    row[name] = ""
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.get_logger().info(f"Migrated metrics CSV: added columns {missing}")

    def _prepare_csv_files(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_csv_add_eval_columns()

        if not self.output_csv.exists():
            with self.output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "plan_seq",
                        "num_steps",
                        "text",
                        "t_text",
                        "t_plan",
                        "t_done",
                        "dt_text_to_plan",
                        "dt_plan_to_done",
                        "dt_text_to_done",
                        "llm_total",
                        "llm_norm",
                        "llm_plan",
                        "llm_valid",
                        "gt_text",
                        "pred_text",
                        "cer",
                        "plan_success",
                        "execution_success",
                        "pipeline_status",
                        "target_distance_m",
                        "actual_distance_m",
                        "distance_error_m",
                        "timeout_flag",
                        "stop_reason",
                        "asl_mode",
                        "asl_stage",
                        "asl_t_in",
                        "asl_t_done",
                        "asl_total",
                        "asl_report_stage",
                        "asl_report_result",
                        "asl_report_total",
                        "planner_report_result",
                        "planner_report_total",
                        "executor_report_result",
                        "executor_report_total",
                        "cpu_percent_done",
                        "mem_percent_done",
                    ]
                )

    def _recognized_text_callback(self, msg: RecognizedText) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        text = (msg.text or "").strip()
        if not text:
            return
        with self._lock:
            self._current.text = text
            self._current.t_text = now

    def _pipeline_status_callback(self, msg: String) -> None:
        status = (msg.data or "").strip()
        if not status:
            return
        with self._lock:
            self._last_pipeline_status = status

    def _asl_timing_callback(self, msg: String) -> None:
        text = msg.data or ""
        if not text.startswith("ASL TIMING"):
            return
        kv = _parse_kv(text)
        with self._lock:
            self._current.asl_mode = str(kv.get("mode", "") or "")
            self._current.asl_stage = str(kv.get("stage", "") or "")
            t_in_raw = kv.get("t_in")
            t_done_raw = kv.get("t_done")
            total_raw = kv.get("total")
            try:
                self._current.asl_t_in = float(t_in_raw) if t_in_raw is not None else None
            except (TypeError, ValueError):
                self._current.asl_t_in = None
            try:
                self._current.asl_t_done = float(t_done_raw) if t_done_raw is not None else None
            except (TypeError, ValueError):
                self._current.asl_t_done = None
            try:
                self._current.asl_total = float(total_raw) if total_raw is not None else None
            except (TypeError, ValueError):
                self._current.asl_total = None

    def _execution_events_callback(self, msg: String) -> None:
        text = msg.data or ""
        kv = _parse_kv(text)
        with self._lock:
            if text.startswith("PLAN START"):
                self._current.t_plan = float(kv.get("t", self.get_clock().now().nanoseconds / 1e9))
                self._current.num_steps = int(kv.get("steps", 0))
                self._current.plan_success = 1
                return

            if text.startswith("EXECUTION DONE") or text.startswith("PLAN DONE"):
                self._current.t_done = float(kv.get("t", self.get_clock().now().nanoseconds / 1e9))
                result = str(kv.get("result", "") or "").lower()
                if result == "failed":
                    self._current.execution_success = 0
                    if not self._current.stop_reason:
                        self._current.stop_reason = "executor_failed"
                elif result == "success":
                    self._current.execution_success = 1
                if self._current.execution_success is None:
                    self._current.execution_success = 1 if self._current.timeout_flag == 0 else 0
                if self._current.plan_success is None:
                    self._current.plan_success = 1 if self._current.t_plan is not None else 0
                self._current.pipeline_status = self._last_pipeline_status
                self._append_plan_row_locked()
                self._current = PlanRecord()
                return

            if text.startswith("STEP ODOM LINEAR START"):
                target_raw = kv.get("target_m")
                try:
                    self._current.target_distance_m = abs(float(target_raw)) if target_raw is not None else None
                except (TypeError, ValueError):
                    self._current.target_distance_m = None
                return

            if text.startswith("STEP ODOM LINEAR OK"):
                traveled_raw = kv.get("traveled_m")
                try:
                    self._current.actual_distance_m = abs(float(traveled_raw)) if traveled_raw is not None else None
                except (TypeError, ValueError):
                    self._current.actual_distance_m = None
                if self._current.target_distance_m is not None and self._current.actual_distance_m is not None:
                    self._current.distance_error_m = abs(self._current.target_distance_m - self._current.actual_distance_m)
                self._current.timeout_flag = 0
                self._current.stop_reason = "goal_reached"
                self._current.execution_success = 1
                return

            if text.startswith("STEP ODOM LINEAR TIMEOUT"):
                traveled_raw = kv.get("traveled_m")
                target_raw = kv.get("target_m")
                try:
                    self._current.actual_distance_m = abs(float(traveled_raw)) if traveled_raw is not None else None
                except (TypeError, ValueError):
                    self._current.actual_distance_m = None
                try:
                    self._current.target_distance_m = abs(float(target_raw)) if target_raw is not None else None
                except (TypeError, ValueError):
                    pass
                if self._current.target_distance_m is not None and self._current.actual_distance_m is not None:
                    self._current.distance_error_m = abs(self._current.target_distance_m - self._current.actual_distance_m)
                self._current.timeout_flag = 1
                self._current.stop_reason = "timeout"
                self._current.execution_success = 0
                return

            if text.startswith("STEP ODOM LINEAR STUCK"):
                traveled_raw = kv.get("traveled_m")
                target_raw = kv.get("target_m")
                try:
                    self._current.actual_distance_m = abs(float(traveled_raw)) if traveled_raw is not None else None
                except (TypeError, ValueError):
                    self._current.actual_distance_m = None
                try:
                    self._current.target_distance_m = abs(float(target_raw)) if target_raw is not None else None
                except (TypeError, ValueError):
                    pass
                if self._current.target_distance_m is not None and self._current.actual_distance_m is not None:
                    self._current.distance_error_m = abs(self._current.target_distance_m - self._current.actual_distance_m)
                self._current.timeout_flag = 1
                self._current.stop_reason = "odom_stuck"
                self._current.execution_success = 0

    def _planner_timing_callback(self, msg: String) -> None:
        text = msg.data or ""
        if not text.startswith("LLM TIMING"):
            return
        kv = _parse_kv(text)
        with self._lock:
            self._current.t_text = float(kv.get("t_in", self._current.t_text or 0.0))
            self._current.llm_total = float(kv.get("total", 0.0))
            self._current.llm_norm = float(kv.get("norm", 0.0))
            self._current.llm_plan = float(kv.get("plan", 0.0))
            self._current.llm_valid = float(kv.get("valid", 0.0))

    def _executor_timing_callback(self, msg: String) -> None:
        _ = msg

    def _asl_report_callback(self, msg: String) -> None:
        text = msg.data or ""
        if not text.startswith("ASL REPORT"):
            return
        kv = _parse_kv(text)
        with self._lock:
            self._current.asl_report_stage = str(kv.get("stage", "") or "")
            self._current.asl_report_result = str(kv.get("result", "") or "")
            total_raw = kv.get("total")
            try:
                self._current.asl_report_total = float(total_raw) if total_raw is not None else None
            except (TypeError, ValueError):
                self._current.asl_report_total = None
            gt_raw = kv.get("ground_truth")
            pred_raw = kv.get("prediction")
            if gt_raw is not None:
                self._current.gt_text = str(gt_raw).replace("_", " ")
            if pred_raw is not None:
                self._current.pred_text = str(pred_raw).replace("_", " ")
            cer_raw = kv.get("cer")
            try:
                self._current.cer = float(cer_raw) if cer_raw is not None else self._current.cer
            except (TypeError, ValueError):
                pass
            exec_success_raw = kv.get("execution_success")
            if exec_success_raw not in (None, ""):
                try:
                    self._current.execution_success = int(exec_success_raw)
                except (TypeError, ValueError):
                    pass
            stop_reason_raw = kv.get("stop_reason")
            if stop_reason_raw:
                self._current.stop_reason = str(stop_reason_raw).replace("_", " ")

    def _planner_report_callback(self, msg: String) -> None:
        text = msg.data or ""
        if not text.startswith("PLANNER REPORT"):
            return
        kv = _parse_kv(text)
        with self._lock:
            self._current.planner_report_result = str(kv.get("result", "") or "")
            total_raw = kv.get("total")
            try:
                self._current.planner_report_total = float(total_raw) if total_raw is not None else None
            except (TypeError, ValueError):
                self._current.planner_report_total = None
            if self._current.planner_report_result == "plan_skipped":
                t_done_raw = kv.get("t_done")
                try:
                    self._current.t_done = (
                        float(t_done_raw)
                        if t_done_raw is not None
                        else self.get_clock().now().nanoseconds / 1e9
                    )
                except (TypeError, ValueError):
                    self._current.t_done = self.get_clock().now().nanoseconds / 1e9
                if self._current.plan_success is None:
                    self._current.plan_success = 0
                if self._current.execution_success is None:
                    self._current.execution_success = 0
                if not self._current.stop_reason:
                    reason = str(kv.get("reason", "") or "unknown")
                    self._current.stop_reason = f"plan_skipped:{reason}"
                self._current.pipeline_status = self._last_pipeline_status
                self._append_plan_row_locked()
                self._current = PlanRecord()

    def _executor_report_callback(self, msg: String) -> None:
        text = msg.data or ""
        if not text.startswith("EXECUTOR REPORT"):
            return
        kv = _parse_kv(text)
        with self._lock:
            self._current.executor_report_result = str(kv.get("result", "") or "")
            total_raw = kv.get("total")
            try:
                self._current.executor_report_total = float(total_raw) if total_raw is not None else None
            except (TypeError, ValueError):
                self._current.executor_report_total = None

    def _append_plan_row_locked(self) -> None:
        self._plan_seq += 1
        rec = self._current

        dt_text_to_plan = (rec.t_plan - rec.t_text) if (rec.t_text is not None and rec.t_plan is not None) else None
        dt_plan_to_done = (rec.t_done - rec.t_plan) if (rec.t_plan is not None and rec.t_done is not None) else None
        dt_text_to_done = (rec.t_done - rec.t_text) if (rec.t_text is not None and rec.t_done is not None) else None
        cpu_percent_done = psutil.cpu_percent(interval=None)
        mem_percent_done = psutil.virtual_memory().percent

        with self.output_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self._plan_seq,
                    rec.num_steps,
                    rec.text,
                    rec.t_text,
                    rec.t_plan,
                    rec.t_done,
                    dt_text_to_plan,
                    dt_plan_to_done,
                    dt_text_to_done,
                    rec.llm_total,
                    rec.llm_norm,
                    rec.llm_plan,
                    rec.llm_valid,
                    rec.gt_text,
                    rec.pred_text,
                    rec.cer,
                    rec.plan_success,
                    rec.execution_success,
                    rec.pipeline_status,
                    rec.target_distance_m,
                    rec.actual_distance_m,
                    rec.distance_error_m,
                    rec.timeout_flag,
                    rec.stop_reason,
                    rec.asl_mode,
                    rec.asl_stage,
                    rec.asl_t_in,
                    rec.asl_t_done,
                    rec.asl_total,
                    rec.asl_report_stage,
                    rec.asl_report_result,
                    rec.asl_report_total,
                    rec.planner_report_result,
                    rec.planner_report_total,
                    rec.executor_report_result,
                    rec.executor_report_total,
                    cpu_percent_done,
                    mem_percent_done,
                ]
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineEvalRos2Node()
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
