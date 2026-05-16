#!/usr/bin/env python3

import csv
import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage


class HandCalibrationCsvReplay(Node):
    def __init__(self):
        super().__init__("hand_calibration_csv_replay")

        self.declare_and_read_params()
        self.rows, self.columns = self.load_csv(self.csv_path)
        self.column_map = self.detect_columns(self.columns)

        self.tf_pub = (
            self.create_publisher(TFMessage, self.tf_topic, 10)
            if self.publish_tf
            else None
        )
        self.current_step_pub = self.create_publisher(
            String,
            "/replay/current_step",
            10,
        )
        self.left_pose_pub = None
        self.right_pose_pub = None
        if self.publish_pose_topics:
            self.left_pose_pub = self.create_publisher(
                PoseStamped,
                self.left_pose_topic,
                10,
            )
            self.right_pose_pub = self.create_publisher(
                PoseStamped,
                self.right_pose_topic,
                10,
            )

        self.reset_srv = self.create_service(
            Trigger,
            "~/reset_replay",
            self.reset_replay_callback,
        )
        self.pause_srv = self.create_service(
            Trigger,
            "~/pause_replay",
            self.pause_replay_callback,
        )

        self.row_index = 0
        self.paused = self.start_paused
        self.pause_until = 0.0
        self.current_step_name = None
        self.last_log_time = 0.0
        self.next_publish_time = time.monotonic()
        self.initial_hold_until = self.next_publish_time + self.initial_hold_sec

        timer_period = 1.0 / max(self.publish_rate, 1e-6)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.log_startup_summary()

    def declare_and_read_params(self):
        self.declare_parameter(
            "csv_path",
            "/home/zaid/humanoid_ws/hand_axis_calibration.csv",
        )
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("initial_hold_sec", 5.0)
        self.declare_parameter("pause_between_steps_sec", 1.0)
        self.declare_parameter("parent_frame", "replay_vr_origin")
        self.declare_parameter("left_child_frame", "hand_left")
        self.declare_parameter("right_child_frame", "hand_right")
        self.declare_parameter("tf_topic", "/tf")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_pose_topics", False)
        self.declare_parameter("left_pose_topic", "/replay/left_hand_pose")
        self.declare_parameter("right_pose_topic", "/replay/right_hand_pose")
        self.declare_parameter("use_csv_timestamps", False)
        self.declare_parameter("start_paused", False)
        self.declare_parameter("debug", True)

        self.csv_path = Path(str(self.get_parameter("csv_path").value)).expanduser()
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.initial_hold_sec = max(
            0.0,
            float(self.get_parameter("initial_hold_sec").value),
        )
        self.pause_between_steps_sec = max(
            0.0,
            float(self.get_parameter("pause_between_steps_sec").value),
        )
        self.parent_frame = str(self.get_parameter("parent_frame").value)
        self.left_child_frame = str(self.get_parameter("left_child_frame").value)
        self.right_child_frame = str(self.get_parameter("right_child_frame").value)
        self.tf_topic = str(self.get_parameter("tf_topic").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.publish_pose_topics = bool(
            self.get_parameter("publish_pose_topics").value
        )
        self.left_pose_topic = str(self.get_parameter("left_pose_topic").value)
        self.right_pose_topic = str(self.get_parameter("right_pose_topic").value)
        self.use_csv_timestamps = bool(
            self.get_parameter("use_csv_timestamps").value
        )
        self.start_paused = bool(self.get_parameter("start_paused").value)
        self.debug = bool(self.get_parameter("debug").value)

    def load_csv(self, csv_path):
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise RuntimeError(f"CSV file has no header: {csv_path}")
            rows = list(reader)

        if not rows:
            raise RuntimeError(f"CSV file has no data rows: {csv_path}")
        return rows, list(reader.fieldnames)

    def detect_columns(self, columns):
        available = {name.strip().lower(): name for name in columns}
        column_map = {}
        for side in ("left", "right"):
            for component in ("x", "y", "z"):
                column_map[f"{side}_{component}"] = self.find_column(
                    available,
                    self.position_candidates(side, component),
                )
            for component in ("qx", "qy", "qz", "qw"):
                column_map[f"{side}_{component}"] = self.find_column(
                    available,
                    self.quaternion_candidates(side, component),
                )

        column_map["timestamp"] = self.find_column(
            available,
            ("timestamp", "time", "stamp", "t"),
            required=False,
        )
        column_map["step_name"] = self.find_column(
            available,
            ("step_name", "step", "name", "requested_direction", "requested_rotation"),
            required=False,
        )
        return column_map

    @staticmethod
    def position_candidates(side, component):
        return (
            f"{side}_{component}",
            f"{side}_wrist_{component}",
            f"{side}_position_{component}",
            f"{side}_wrist_position_{component}",
            f"{side}_hand_{component}",
            f"{side}_hand_position_{component}",
        )

    @staticmethod
    def quaternion_candidates(side, component):
        axis = component[1]
        return (
            f"{side}_{component}",
            f"{side}_wrist_{component}",
            f"{side}_orientation_{component}",
            f"{side}_quat_{component}",
            f"{side}_quaternion_{component}",
            f"{side}_wrist_orientation_{component}",
            f"{side}_wrist_quat_{component}",
            f"{side}_wrist_quaternion_{component}",
            f"{side}_q{axis}",
        )

    @staticmethod
    def find_column(available, candidates, required=True):
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in available:
                return available[key]
        if not required:
            return None
        raise RuntimeError(
            "Missing required CSV column. Tried: " + ", ".join(candidates)
        )

    def log_startup_summary(self):
        detected = ", ".join(
            f"{key}={value}" for key, value in sorted(self.column_map.items()) if value
        )
        self.get_logger().info(
            "hand_calibration_csv_replay ready: "
            f"csv_path={self.csv_path}, rows={len(self.rows)}, "
            f"publish_rate={self.publish_rate}, loop={self.loop}, "
            f"initial_hold_sec={self.initial_hold_sec}, "
            f"use_csv_timestamps={self.use_csv_timestamps}, "
            f"publish_tf={self.publish_tf}, publish_pose_topics={self.publish_pose_topics}"
        )
        self.get_logger().info(
            "TF frames: "
            f"topic={self.tf_topic}, parent={self.parent_frame}, "
            f"left_child={self.left_child_frame}, right_child={self.right_child_frame}"
        )
        self.get_logger().info("Detected CSV columns: " + detected)
        if self.paused:
            self.get_logger().info(
                "Replay starts paused. Call ~/pause_replay to resume."
            )

    def timer_callback(self):
        if self.paused:
            return
        if not self.rows:
            return

        now = time.monotonic()
        if now < self.pause_until or now < self.next_publish_time:
            return

        row = self.rows[self.row_index]
        step_name = self.get_step_name(row)
        if step_name != self.current_step_name:
            self.current_step_name = step_name
            self.get_logger().info(f"Replaying step: {step_name}")
        self.current_step_pub.publish(String(data=step_name))

        left_pose = self.row_to_pose(row, "left")
        right_pose = self.row_to_pose(row, "right")
        stamp = self.get_clock().now().to_msg()

        if self.publish_tf:
            self.tf_pub.publish(
                TFMessage(
                    transforms=[
                        self.make_transform(stamp, self.left_child_frame, left_pose),
                        self.make_transform(stamp, self.right_child_frame, right_pose),
                    ]
                )
            )
        if self.publish_pose_topics:
            self.left_pose_pub.publish(
                self.make_pose_stamped(stamp, self.left_child_frame, left_pose)
            )
            self.right_pose_pub.publish(
                self.make_pose_stamped(stamp, self.right_child_frame, right_pose)
            )

        self.log_replay_debug(step_name, left_pose, right_pose)
        if self.should_hold_initial_row(now):
            return
        self.advance_replay_index()

    def should_hold_initial_row(self, now):
        return self.row_index == 0 and now < self.initial_hold_until

    def row_to_pose(self, row, side):
        x = self.read_float(row, self.column_map[f"{side}_x"])
        y = self.read_float(row, self.column_map[f"{side}_y"])
        z = self.read_float(row, self.column_map[f"{side}_z"])
        qx = self.read_float(row, self.column_map[f"{side}_qx"])
        qy = self.read_float(row, self.column_map[f"{side}_qy"])
        qz = self.read_float(row, self.column_map[f"{side}_qz"])
        qw = self.read_float(row, self.column_map[f"{side}_qw"])
        qx, qy, qz, qw = self.normalize_quaternion(side, qx, qy, qz, qw)
        return {
            "position": (x, y, z),
            "orientation": (qx, qy, qz, qw),
        }

    @staticmethod
    def read_float(row, column):
        try:
            return float(row[column])
        except Exception as exc:
            raise RuntimeError(f"Invalid float in column {column!r}: {row}") from exc

    def normalize_quaternion(self, side, qx, qy, qz, qw):
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-8 or not math.isfinite(norm):
            self.throttled_warning(
                f"{side}_bad_quaternion",
                f"Invalid {side} quaternion at row {self.row_index}; using identity.",
            )
            return 0.0, 0.0, 0.0, 1.0
        return qx / norm, qy / norm, qz / norm, qw / norm

    def make_transform(self, stamp, child_frame, pose):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = child_frame
        x, y, z = pose["position"]
        qx, qy, qz, qw = pose["orientation"]
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        return transform

    def make_pose_stamped(self, stamp, child_frame, pose):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.parent_frame
        x, y, z = pose["position"]
        qx, qy, qz, qw = pose["orientation"]
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def get_step_name(self, row):
        column = self.column_map.get("step_name")
        if column is None:
            return "unknown"
        value = row.get(column, "")
        return value if value else "unknown"

    def advance_replay_index(self):
        previous_index = self.row_index
        self.row_index += 1

        if self.row_index >= len(self.rows):
            if self.loop:
                self.row_index = 0
            else:
                self.row_index = len(self.rows) - 1
                self.paused = True
                self.get_logger().info("Replay finished; paused at final row.")
                return

        delay = self.compute_next_delay(previous_index, self.row_index)
        if self.step_changed(previous_index, self.row_index):
            delay = max(delay, self.pause_between_steps_sec)
        self.next_publish_time = time.monotonic() + delay

    def compute_next_delay(self, previous_index, next_index):
        if not self.use_csv_timestamps:
            return 0.0

        timestamp_column = self.column_map.get("timestamp")
        if timestamp_column is None:
            self.throttled_warning(
                "missing_timestamp",
                "use_csv_timestamps=true but no timestamp column was detected; using fixed-rate timer.",
            )
            return 0.0

        try:
            previous_timestamp = float(self.rows[previous_index][timestamp_column])
            next_timestamp = float(self.rows[next_index][timestamp_column])
        except Exception:
            self.throttled_warning(
                "bad_timestamp",
                "Could not parse CSV timestamp; using fixed-rate timer.",
            )
            return 0.0

        if next_index == 0 and next_timestamp < previous_timestamp:
            return self.pause_between_steps_sec
        return max(0.0, min(next_timestamp - previous_timestamp, 10.0))

    def step_changed(self, previous_index, next_index):
        previous_step = self.get_step_name(self.rows[previous_index])
        next_step = self.get_step_name(self.rows[next_index])
        return previous_step != next_step

    def log_replay_debug(self, step_name, left_pose, right_pose):
        if not self.debug:
            return
        now = time.monotonic()
        if now - self.last_log_time < 1.0:
            return
        self.last_log_time = now
        left_xyz = left_pose["position"]
        right_xyz = right_pose["position"]
        self.get_logger().info(
            f"row={self.row_index}/{len(self.rows) - 1}, step_name={step_name}, "
            f"left_xyz=({left_xyz[0]:.3f}, {left_xyz[1]:.3f}, {left_xyz[2]:.3f}), "
            f"right_xyz=({right_xyz[0]:.3f}, {right_xyz[1]:.3f}, {right_xyz[2]:.3f})"
        )

    def throttled_warning(self, key, message, interval_sec=2.0):
        now = time.monotonic()
        last = getattr(self, "_last_warning_time", {})
        if now - last.get(key, 0.0) < interval_sec:
            return
        last[key] = now
        self._last_warning_time = last
        self.get_logger().warning(message)

    def reset_replay_callback(self, _request, response):
        self.row_index = 0
        self.pause_until = 0.0
        self.next_publish_time = time.monotonic()
        self.initial_hold_until = self.next_publish_time + self.initial_hold_sec
        self.current_step_name = None
        response.success = True
        response.message = "Replay reset to row 0."
        self.get_logger().info(response.message)
        return response

    def pause_replay_callback(self, _request, response):
        self.paused = not self.paused
        state = "paused" if self.paused else "resumed"
        response.success = True
        response.message = f"Replay {state}."
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HandCalibrationCsvReplay()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
