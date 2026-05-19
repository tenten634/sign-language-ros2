#!/usr/bin/env python3
"""Pipeline LED orchestrator for Robotont."""

from __future__ import annotations

import os
import signal
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from robotont_msgs.msg import LedModuleMode
except Exception:  # noqa: BLE001
    LedModuleMode = None

PIPELINE_STATUS_TOPIC = "/pipeline/status"
DEFAULT_SPIN_SPEED = 15
FAST_SPIN_SPEED = 150


class PipelineLedNode(Node):
    def __init__(self) -> None:
        super().__init__("pipeline_led_node")

        # Use absolute name so commands reach robotont_driver's /led_mode, not /pipeline_led_node/led_mode.
        self.declare_parameter("led_mode_topic", "/led_mode")
        self.declare_parameter("heartbeat_hz", 3.0)
        self.declare_parameter("state_hold_sec", 0.35)

        self._last_led_key: Optional[str] = None
        self._status: str = "waiting_for_palm"
        self._led_off_done: bool = False
        self._status_since_sec: float = time.time()

        if LedModuleMode is None:
            self.get_logger().error("robotont_msgs is unavailable; pipeline_led_node cannot publish LED commands.")
            raise RuntimeError("robotont_msgs unavailable")

        led_mode_topic = str(self.get_parameter("led_mode_topic").value).strip() or "/led_mode"
        heartbeat_hz = float(self.get_parameter("heartbeat_hz").value) or 3.0
        heartbeat_hz = max(0.5, min(heartbeat_hz, 20.0))
        self._state_hold_sec = max(0.0, float(self.get_parameter("state_hold_sec").value))

        # Packet format must match robotont_driver jazzy PluginLedModule::writeMode (LM:mode:params...).
        # https://github.com/robotont/robotont_driver/blob/jazzy/src/plugin_led_module.cpp
        # Publisher QoS: default RELIABLE is compatible with driver's sensor_data (BEST_EFFORT) subscription.
        self._led_pub = self.create_publisher(LedModuleMode, led_mode_topic, 10)
        self._status_sub = self.create_subscription(String, PIPELINE_STATUS_TOPIC, self._on_status, 10)
        self._timer = self.create_timer(1.0 / heartbeat_hz, self._on_timer)

        # Ctrl+C / launch teardown: ensure NONE reaches the driver before the context dies.
        try:
            self.context.add_on_shutdown_callback(self._burst_led_off)
        except AttributeError:
            pass

        # Launch sends SIGINT to all children; burst here too so NONE is sent even if
        # destroy_node ordering is tight with DDS teardown.
        def _sig_led_off(signum: int, frame: object) -> None:  # noqa: ARG001
            try:
                self._burst_led_off()
            except Exception:  # noqa: BLE001
                pass
            signal.signal(signum, signal.SIG_DFL)
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            os.kill(os.getpid(), signum)

        try:
            signal.signal(signal.SIGINT, _sig_led_off)
            signal.signal(signal.SIGTERM, _sig_led_off)
        except ValueError:
            pass  # not main thread (e.g. tests)

        self.get_logger().info(
            f"LED orchestrator started: led={led_mode_topic}, status={PIPELINE_STATUS_TOPIC}"
        )

    def _publish_led_mode(self, mode: int, params: List[int]) -> None:
        msg = LedModuleMode()
        msg.mode = int(mode)
        msg.params = [int(v) for v in params]
        self._led_pub.publish(msg)

    def _burst_led_off(self) -> None:
        """Stop heartbeat republish and send LedModuleMode.NONE a few times (interrupt-safe)."""
        if self._led_off_done:
            return
        self._led_off_done = True
        try:
            self._timer.cancel()
        except Exception:  # noqa: BLE001
            pass
        self._last_led_key = None
        for _ in range(25):
            try:
                self._publish_led_mode(LedModuleMode.NONE, [])
                rclpy.spin_once(self, timeout_sec=0.08)
            except Exception:  # noqa: BLE001
                break
            time.sleep(0.04)
        time.sleep(0.12)

    def _on_status(self, msg: String) -> None:
        prev = self._status
        next_status = (msg.data or "").strip() or "waiting_for_palm"
        now = time.time()
        if next_status != prev:
            prev_pri = self._status_priority(prev)
            next_pri = self._status_priority(next_status)
            held_for = now - self._status_since_sec
            # Avoid rapid flicker/collision by rejecting short-lived downgrades.
            if next_pri < prev_pri and held_for < self._state_hold_sec:
                self.get_logger().debug(
                    f"ignore status downgrade: {prev}({prev_pri}) -> {next_status}({next_pri}) "
                    f"held_for={held_for:.3f}s < hold={self._state_hold_sec:.3f}s"
                )
                return
            self._status = next_status
            self._status_since_sec = now
        else:
            self._status = next_status
        if self._status != prev:
            self.get_logger().info(f"pipeline_status: {prev} -> {self._status}")
        self._update_led(force=True)

    @staticmethod
    def _status_priority(status: str) -> int:
        """Higher number wins during short collision windows."""
        high = {"plan_executing"}
        medium = {
            "llm_planning",
            "predicting",
            "text_processing",
            "video_processing",
            "text_waiting_plan",
            "video_waiting_plan",
            "recording_end_palm_hold",
        }
        low = {
            "waiting_for_palm",
            "open_palm_hold",
            "countdown",
            "recording",
            "recording_ready_end",
            "text_waiting",
            "video_waiting",
            "plan_ready",
        }
        if status in high:
            return 30
        if status in medium:
            return 20
        if status in low:
            return 10
        return 0

    def _on_timer(self) -> None:
        self._update_led(force=False)

    def _update_led(self, *, force: bool) -> None:
        key = self._status
        if key == self._last_led_key and not force:
            return

        if self._status == "waiting_for_palm":
            self._publish_led_mode(LedModuleMode.SPIN, [0, 40, 255, DEFAULT_SPIN_SPEED])
        elif self._status == "open_palm_hold":
            self._publish_led_mode(LedModuleMode.SPIN, [0, 40, 255, FAST_SPIN_SPEED])
        elif self._status == "countdown":
            self._publish_led_mode(LedModuleMode.PULSE, [0, 40, 255, 20])
        elif self._status == "recording":
            self._publish_led_mode(LedModuleMode.NONE, [])
        elif self._status == "recording_ready_end":
            self._publish_led_mode(LedModuleMode.PULSE, [0, 40, 255, 20])
        elif self._status == "recording_end_palm_hold":
            self._publish_led_mode(LedModuleMode.SPIN, [0, 40, 255, FAST_SPIN_SPEED])
        elif self._status == "text_waiting":
            self._publish_led_mode(LedModuleMode.SPIN, [0, 40, 255, DEFAULT_SPIN_SPEED])
        elif self._status == "text_processing":
            self._publish_led_mode(LedModuleMode.SPIN, [255, 0, 0, DEFAULT_SPIN_SPEED])
        elif self._status == "video_waiting":
            self._publish_led_mode(LedModuleMode.SPIN, [0, 40, 255, DEFAULT_SPIN_SPEED])
        elif self._status == "video_processing":
            self._publish_led_mode(LedModuleMode.SPIN, [255, 0, 0, DEFAULT_SPIN_SPEED])
        elif self._status == "predicting":
            self._publish_led_mode(LedModuleMode.SPIN, [255, 0, 0, DEFAULT_SPIN_SPEED])
        elif self._status == "showing_result":
            self._publish_led_mode(LedModuleMode.NONE, [])
        elif self._status == "llm_planning":
            self._publish_led_mode(LedModuleMode.SPIN, [255, 0, 0, DEFAULT_SPIN_SPEED])
        elif self._status == "text_waiting_plan":
            self._publish_led_mode(LedModuleMode.SPIN, [255, 0, 0, DEFAULT_SPIN_SPEED])
        elif self._status == "video_waiting_plan":
            self._publish_led_mode(LedModuleMode.SPIN, [255, 0, 0, DEFAULT_SPIN_SPEED])
        elif self._status == "plan_executing":
            self._publish_led_mode(LedModuleMode.SPIN, [0, 255, 0, DEFAULT_SPIN_SPEED])
        else:
            self._publish_led_mode(LedModuleMode.NONE, [])

        self._last_led_key = key

    def destroy_node(self) -> bool:
        try:
            self._burst_led_off()
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineLedNode()
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

