#!/usr/bin/env python3
"""Execute motion plans from /llm_planner/motion_plan using odom + cmd_vel.

This executor is odom-only and supports:
- move_forward
- move_backward
- rotate_left
- rotate_right
- wait
- navigate_to (virtual named locations from locations.json)
"""

from __future__ import annotations

import json
import math
import threading
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from .robot_motion_planner import LOCATIONS

PIPELINE_STATUS_TOPIC = "/pipeline/status"
EXECUTION_STATUS_TOPIC = "/pipeline/executor/status"
EXECUTOR_TIMING_TOPIC = "/pipeline/executor/timing"
EXECUTOR_REPORT_TOPIC = "/pipeline/executor/report"

def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PlanExecutorNode(Node):
    def __init__(self) -> None:
        super().__init__("llm_plan_executor")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.declare_parameter("motion_plan_topic", "/llm_planner/motion_plan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_linear_speed", 0.12)
        self.declare_parameter("odom_angular_speed", 0.35)
        self.declare_parameter("odom_position_tolerance_m", 0.03)
        self.declare_parameter("odom_angle_tolerance_rad", 0.08)
        self.declare_parameter("odom_linear_timeout_sec", 60.0)
        self.declare_parameter("odom_rotate_timeout_sec", 60.0)
        self.declare_parameter("odom_control_period_sec", 0.05)
        self.declare_parameter("odom_stuck_timeout_sec", 4.0)
        self.declare_parameter("odom_stuck_epsilon_m", 0.002)
        self.declare_parameter("odom_distance_scale", 1.0)

        motion_plan_topic = self.get_parameter("motion_plan_topic").get_parameter_value().string_value
        self._cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self._odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self._odom_lin_speed = float(self.get_parameter("odom_linear_speed").value)
        self._odom_ang_speed = float(self.get_parameter("odom_angular_speed").value)
        self._odom_pos_tol = float(self.get_parameter("odom_position_tolerance_m").value)
        self._odom_ang_tol = float(self.get_parameter("odom_angle_tolerance_rad").value)
        self._odom_lin_timeout = float(self.get_parameter("odom_linear_timeout_sec").value)
        self._odom_rot_timeout = float(self.get_parameter("odom_rotate_timeout_sec").value)
        self._odom_dt = float(self.get_parameter("odom_control_period_sec").value)
        self._odom_stuck_timeout = max(0.2, float(self.get_parameter("odom_stuck_timeout_sec").value))
        self._odom_stuck_epsilon = max(1e-4, float(self.get_parameter("odom_stuck_epsilon_m").value))
        self._odom_distance_scale = max(0.1, float(self.get_parameter("odom_distance_scale").value))
        self._plan_failed: bool = False

        self.current_odom: Optional[Odometry] = None
        self.odom_sub = self.create_subscription(
            Odometry,
            self._odom_topic,
            self._odom_callback,
            10,
        )

        self.plan_sub = self.create_subscription(
            String,
            motion_plan_topic,
            self._plan_callback,
            10,
        )

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)

        self.exec_events_pub = self.create_publisher(
            String,
            EXECUTION_STATUS_TOPIC,
            10,
        )
        self.executor_timing_pub = self.create_publisher(
            String,
            EXECUTOR_TIMING_TOPIC,
            10,
        )
        self.executor_report_pub = self.create_publisher(
            String,
            EXECUTOR_REPORT_TOPIC,
            10,
        )
        self.pipeline_status_pub = self.create_publisher(String, PIPELINE_STATUS_TOPIC, 10)
        self._plan_seq: int = 0
        self._active_plan_steps: int = 0
        self._exec_thread: Optional[threading.Thread] = None

        self.get_logger().info(
            f"[LLM Plan Executor] mode=odom "
            f"plan_topic={motion_plan_topic} cmd_vel={self._cmd_vel_topic} odom={self._odom_topic}"
        )

    def _publish_pipeline_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.pipeline_status_pub.publish(msg)

    def _odom_callback(self, msg: Odometry) -> None:
        self.current_odom = msg

    def _plan_callback(self, msg: String) -> None:
        try:
            plan: Any = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse motion plan JSON: {e}")
            return

        if not isinstance(plan, list):
            self.get_logger().error("Motion plan JSON is not a list; skipping.")
            return

        if self._exec_thread is not None and self._exec_thread.is_alive():
            self.get_logger().warn("Executor is busy with previous plan; skipping newly received plan.")
            self._publish_exec_event("PLAN SKIP reason=executor_busy")
            return

        self._plan_seq += 1
        plan_id = self._plan_seq
        self._active_plan_steps = len(plan)
        self._exec_thread = threading.Thread(
            target=self._execute_plan,
            args=(plan_id, plan),
            daemon=True,
        )
        self._exec_thread.start()

    def _execute_plan(self, plan_id: int, plan: list[dict[str, Any]]) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info(f"[LLM Plan Executor] Received motion plan with {len(plan)} steps (plan_id={plan_id}).")
        self._publish_pipeline_status("plan_executing")
        self._plan_failed = False
        self._publish_exec_event(f"PLAN START id={plan_id} steps={len(plan)} t={now_sec:.3f}")
        exec_t_start = now_sec

        for idx, step in enumerate(plan):
            if not isinstance(step, dict):
                self.get_logger().warn(f"Step {idx} is not an object; skipping.")
                continue

            action = step.get("action")
            if not isinstance(action, str):
                self.get_logger().warn(f"Step {idx} missing 'action'; skipping.")
                continue

            action = action.strip()
            self.get_logger().info(f"[LLM Plan Executor] Executing step {idx}: {action}")
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._publish_exec_event(f"STEP START plan={plan_id} idx={idx} action={action} t={now_sec:.3f}")

            if action == "navigate_to":
                self._handle_navigate_to(step)
            elif action in ("wait",):
                self._handle_wait(step)
            elif action in ("move_forward", "move_backward"):
                self._handle_linear_move(step, action)
            elif action in ("rotate_left", "rotate_right"):
                self._handle_rotate(step, action)
            else:
                self.get_logger().info(f"Unknown or unsupported action '{action}', skipping.")

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._plan_failed:
            self._publish_exec_event(f"EXECUTION DONE id={plan_id} result=failed t={now_sec:.3f}")
            self._publish_executor_timing(plan_id=plan_id, t_in=exec_t_start, t_done=now_sec, result="failed")
            self._publish_executor_report(plan_id=plan_id, t_in=exec_t_start, t_done=now_sec, result="failed")
        else:
            self._publish_exec_event(f"EXECUTION DONE id={plan_id} result=success t={now_sec:.3f}")
            self._publish_executor_timing(plan_id=plan_id, t_in=exec_t_start, t_done=now_sec, result="success")
            self._publish_executor_report(plan_id=plan_id, t_in=exec_t_start, t_done=now_sec, result="success")
        self._publish_pipeline_status("pipeline_done")

    # ---- Step handlers -------------------------------------------------

    def _handle_wait(self, step: Dict[str, Any]) -> None:
        duration = step.get("duration_s")
        if not isinstance(duration, (int, float)) or duration <= 0:
            self.get_logger().warn(f"Invalid wait duration: {duration!r}; skipping.")
            return
        self.get_logger().info(f"Waiting for {duration:.2f} seconds...")
        self._sleep_blocking(duration)

    def _handle_navigate_to(self, step: Dict[str, Any]) -> None:
        location = step.get("location")
        if not isinstance(location, str):
            self.get_logger().warn(f"navigate_to step missing valid 'location': {location!r}; skipping.")
            return

        loc_spec = self._find_location(location)
        if loc_spec is None:
            self.get_logger().warn(f"Unknown location '{location}', skipping navigate_to.")
            self._publish_exec_event(f"STEP ODOM SKIP action=navigate_to reason=unknown_location loc={location}")
            return

        pose = loc_spec.get("pose")
        if not isinstance(pose, dict):
            self.get_logger().warn(f"Location '{location}' has no pose, skipping navigate_to.")
            self._publish_exec_event(f"STEP ODOM SKIP action=navigate_to reason=missing_pose loc={location}")
            return

        try:
            x_rel = float(pose.get("x"))
            y_rel = float(pose.get("y"))
        except (TypeError, ValueError):
            self.get_logger().warn(f"Location '{location}' has invalid pose x/y, skipping navigate_to.")
            self._publish_exec_event(f"STEP ODOM SKIP action=navigate_to reason=invalid_pose loc={location}")
            return

        # Interpret location pose as body-frame relative displacement from current pose
        # (x: forward, y: left), then convert to world-frame target.
        self._wait_odom_ready()
        yaw_now = self._yaw_from_quaternion(self.current_odom.pose.pose.orientation)
        x_now = self.current_odom.pose.pose.position.x
        y_now = self.current_odom.pose.pose.position.y
        c = math.cos(yaw_now)
        s = math.sin(yaw_now)
        x_goal = x_now + (c * x_rel - s * y_rel)
        y_goal = y_now + (s * x_rel + c * y_rel)
        self.get_logger().info(
            f"[odom] navigate_to '{location}' relative=({x_rel:.2f}, {y_rel:.2f}) "
            f"from=({x_now:.2f}, {y_now:.2f}, yaw={yaw_now:.2f}) "
            f"target=({x_goal:.2f}, {y_goal:.2f})"
        )

        yaw_goal = self._parse_pose_yaw_rad(pose)
        self._run_navigate_to_odom(location, x_goal, y_goal, yaw_goal)

    def _handle_linear_move(self, step: Dict[str, Any], action: str) -> None:
        distance = step.get("distance_m")
        if not isinstance(distance, (int, float)) or distance == 0:
            self.get_logger().warn(f"Invalid distance for {action}: {distance!r}; skipping.")
            return
        # Calibrate commanded distance to real-world traveled distance.
        scaled_distance = float(distance) * self._odom_distance_scale
        max_distance_m = 1.0 * self._odom_distance_scale
        if abs(scaled_distance) > max_distance_m:
            self.get_logger().warn(
                f"Scaled distance {scaled_distance:.3f} m exceeds limit {max_distance_m:.3f} m; clamping."
            )
            scaled_distance = max_distance_m if scaled_distance > 0 else -max_distance_m

        if self.current_odom is None:
            self.get_logger().warn("No odom yet; cannot execute linear move.")
            return

        if action == "move_backward":
            distance = -abs(scaled_distance)
        else:
            distance = abs(scaled_distance)

        self._publish_exec_event(f"STEP ODOM LINEAR START target_m={abs(distance):.3f}")
        self._run_linear_odom(distance)

    def _handle_rotate(self, step: Dict[str, Any], action: str) -> None:
        angle = step.get("angle_deg")
        if not isinstance(angle, (int, float)) or angle == 0:
            self.get_logger().warn(f"Invalid angle for {action}: {angle!r}; skipping.")
            return

        if self.current_odom is None:
            self.get_logger().warn("No odom yet; cannot execute rotation.")
            return

        if action == "rotate_right":
            angle = -float(angle)
        else:
            angle = float(angle)

        self._run_rotate_odom(math.radians(angle))

    # ---- Odom backend ---------------------------------------------------

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        if self._cmd_pub is None:
            return
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self._cmd_pub.publish(msg)

    def _stop_cmd_repeat(self, n: int = 5) -> None:
        for _ in range(n):
            self._publish_cmd(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)

    def _run_linear_odom(self, distance_m: float) -> None:
        """Blocking: drive along current heading until |delta position| >= |distance_m|."""
        if self._cmd_pub is None:
            return
        self._wait_odom_ready()
        x0 = self.current_odom.pose.pose.position.x
        y0 = self.current_odom.pose.pose.position.y
        target = abs(float(distance_m))
        speed = self._odom_lin_speed
        direction = 1.0 if distance_m >= 0.0 else -1.0
        t0 = self.get_clock().now().nanoseconds / 1e9
        last_progress_time = t0
        last_traveled = 0.0
        self.get_logger().info(
            f"[odom] linear target={distance_m:.3f} m speed={speed * direction:.3f} m/s"
        )

        while rclpy.ok():
            if (self.get_clock().now().nanoseconds / 1e9 - t0) > self._odom_lin_timeout:
                if self.current_odom is not None:
                    x = self.current_odom.pose.pose.position.x
                    y = self.current_odom.pose.pose.position.y
                    traveled = math.hypot(x - x0, y - y0)
                    self.get_logger().warn(
                        f"[odom] linear timeout -> stop (traveled={traveled:.3f} m of {target:.3f} m; "
                        f"dx={x - x0:.4f} dy={y - y0:.4f}). "
                        "If traveled≈0: check /cmd_vel subscribers, e-stop, and that /odom updates when wheels move."
                    )
                    self._publish_exec_event(
                        f"STEP ODOM LINEAR TIMEOUT traveled_m={traveled:.3f} target_m={target:.3f}"
                    )
                else:
                    self.get_logger().warn("[odom] linear timeout -> stop (no odom)")
                    self._publish_exec_event(
                        f"STEP ODOM LINEAR TIMEOUT traveled_m=nan target_m={target:.3f}"
                    )
                self._plan_failed = True
                break
            if self.current_odom is None:
                rclpy.spin_once(self, timeout_sec=self._odom_dt)
                continue
            x = self.current_odom.pose.pose.position.x
            y = self.current_odom.pose.pose.position.y
            traveled = math.hypot(x - x0, y - y0)
            if traveled - last_traveled >= self._odom_stuck_epsilon:
                last_traveled = traveled
                last_progress_time = self.get_clock().now().nanoseconds / 1e9
            elif (self.get_clock().now().nanoseconds / 1e9 - last_progress_time) > self._odom_stuck_timeout:
                self.get_logger().warn(
                    f"[odom] linear stuck -> stop (traveled={traveled:.3f} m, "
                    f"no progress > {self._odom_stuck_epsilon:.3f} m for {self._odom_stuck_timeout:.1f} s)"
                )
                self._publish_exec_event(
                    f"STEP ODOM LINEAR STUCK traveled_m={traveled:.3f} target_m={target:.3f}"
                )
                self._plan_failed = True
                break
            if traveled + self._odom_pos_tol >= target:
                self.get_logger().info(f"[odom] linear done traveled={traveled:.3f} m")
                self._publish_exec_event(f"STEP ODOM LINEAR OK traveled_m={traveled:.3f}")
                break
            self._publish_cmd(direction * speed, 0.0)
            rclpy.spin_once(self, timeout_sec=self._odom_dt)

        self._stop_cmd_repeat()

    def _run_rotate_odom(self, delta_yaw_rad: float) -> None:
        """Blocking: in-place rotate until yaw changes by delta_yaw_rad."""
        if self._cmd_pub is None:
            return
        self._wait_odom_ready()
        yaw0 = self._yaw_from_quaternion(self.current_odom.pose.pose.orientation)
        goal = _normalize_angle(yaw0 + delta_yaw_rad)
        t0 = self.get_clock().now().nanoseconds / 1e9
        w = self._odom_ang_speed
        self.get_logger().info(f"[odom] rotate delta={delta_yaw_rad:.3f} rad goal_yaw={goal:.3f}")

        while rclpy.ok():
            if (self.get_clock().now().nanoseconds / 1e9 - t0) > self._odom_rot_timeout:
                self.get_logger().warn("[odom] rotate timeout -> stop")
                self._publish_exec_event("STEP ODOM ROTATE TIMEOUT")
                self._plan_failed = True
                break
            if self.current_odom is None:
                rclpy.spin_once(self, timeout_sec=self._odom_dt)
                continue
            yaw = self._yaw_from_quaternion(self.current_odom.pose.pose.orientation)
            err = _normalize_angle(goal - yaw)
            if abs(err) <= self._odom_ang_tol:
                self.get_logger().info(f"[odom] rotate done err={err:.3f} rad")
                self._publish_exec_event("STEP ODOM ROTATE OK")
                break
            self._publish_cmd(0.0, w if err > 0.0 else -w)
            rclpy.spin_once(self, timeout_sec=self._odom_dt)

        self._stop_cmd_repeat()

    def _run_navigate_to_odom(self, location: str, x_goal: float, y_goal: float, yaw_goal: Optional[float]) -> None:
        _ = yaw_goal
        self._wait_odom_ready()
        if self.current_odom is None:
            self.get_logger().warn("[odom] no odom available, cannot run navigate_to")
            self._publish_exec_event(f"STEP ODOM NAV SKIP loc={location} reason=no_odom")
            return

        # Holonomic XY navigation:
        # move in x/y without forcing pre/post rotation to keep heading stable.
        kp_xy = 0.8
        vmax_x = self._odom_lin_speed
        vmax_y = self._odom_lin_speed
        timeout_sec = self._odom_lin_timeout
        t0 = self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info(
            f"[odom] navigate_to '{location}' target=({x_goal:.2f}, {y_goal:.2f}) mode=xy_no_rotate"
        )
        self._publish_exec_event(
            f"STEP ODOM NAV START loc={location} target_x={x_goal:.2f} target_y={y_goal:.2f}"
        )

        while rclpy.ok():
            if (self.get_clock().now().nanoseconds / 1e9 - t0) > timeout_sec:
                self.get_logger().warn("[odom] navigate_to timeout -> stop")
                self._publish_exec_event(f"STEP ODOM NAV TIMEOUT loc={location}")
                self._plan_failed = True
                break
            if self.current_odom is None:
                rclpy.spin_once(self, timeout_sec=self._odom_dt)
                continue

            pose = self.current_odom.pose.pose
            x_now = pose.position.x
            y_now = pose.position.y
            yaw = self._yaw_from_quaternion(pose.orientation)

            dx_w = x_goal - x_now
            dy_w = y_goal - y_now
            distance = math.hypot(dx_w, dy_w)
            if distance <= self._odom_pos_tol:
                self._publish_exec_event(f"STEP ODOM NAV OK loc={location}")
                break

            # Convert world-frame error to body-frame error.
            c = math.cos(yaw)
            s = math.sin(yaw)
            dx_b = c * dx_w + s * dy_w
            dy_b = -s * dx_w + c * dy_w

            vx = max(-vmax_x, min(vmax_x, kp_xy * dx_b))
            vy = max(-vmax_y, min(vmax_y, kp_xy * dy_b))

            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.angular.z = 0.0
            self._cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=self._odom_dt)

        self._stop_cmd_repeat()

    def _find_location(self, location_id: str) -> Optional[Dict[str, Any]]:
        for location in LOCATIONS.get("locations", []):
            if location.get("id") == location_id:
                return location
        return None

    @staticmethod
    def _parse_pose_yaw_rad(pose: Dict[str, Any]) -> Optional[float]:
        theta_deg = pose.get("theta_deg")
        if theta_deg is not None:
            try:
                return math.radians(float(theta_deg))
            except (TypeError, ValueError):
                return None

        theta_rad = pose.get("theta_rad")
        if theta_rad is not None:
            try:
                return float(theta_rad)
            except (TypeError, ValueError):
                return None
        return None

    def _wait_odom_ready(self, timeout_sec: float = 5.0) -> None:
        t0 = self.get_clock().now().nanoseconds / 1e9
        while self.current_odom is None and rclpy.ok():
            if (self.get_clock().now().nanoseconds / 1e9 - t0) > timeout_sec:
                self.get_logger().warn("[odom] still no odom after wait")
                return
            rclpy.spin_once(self, timeout_sec=0.1)

    def _sleep_blocking(self, duration: float) -> None:
        end_time = self.get_clock().now().nanoseconds / 1e9 + duration
        while rclpy.ok() and (self.get_clock().now().nanoseconds / 1e9) < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _publish_exec_event(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.exec_events_pub.publish(msg)

    def _publish_executor_timing(self, *, plan_id: int, t_in: float, t_done: float, result: str) -> None:
        msg = String()
        msg.data = (
            f"EXEC TIMING id={plan_id} t_in={t_in:.3f} t_done={t_done:.3f} "
            f"total={max(0.0, t_done - t_in):.3f} result={result}"
        )
        self.executor_timing_pub.publish(msg)

    def _publish_executor_report(self, *, plan_id: int, t_in: float, t_done: float, result: str) -> None:
        msg = String()
        msg.data = (
            f"EXECUTOR REPORT id={plan_id} t_in={t_in:.3f} t_done={t_done:.3f} "
            f"total={max(0.0, t_done - t_in):.3f} result={result}"
        )
        self.executor_report_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlanExecutorNode()
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
