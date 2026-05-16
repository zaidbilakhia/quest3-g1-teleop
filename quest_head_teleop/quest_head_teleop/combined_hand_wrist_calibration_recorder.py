#!/usr/bin/env python3

import csv
import math
import shutil
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


BASE_CSV_FIELDS = [
    "timestamp",
    "step_index",
    "step_name",
    "active_hand",
    "requested_direction",
    "requested_rotation",
    "sample_index",
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_z",
    "left_wrist_qx",
    "left_wrist_qy",
    "left_wrist_qz",
    "left_wrist_qw",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_z",
    "right_wrist_qx",
    "right_wrist_qy",
    "right_wrist_qz",
    "right_wrist_qw",
]

MATRIX_FIELDS = [
    f"{hand}_wrist_r{row}{col}"
    for hand in ("left", "right")
    for row in range(3)
    for col in range(3)
]

DELTA_FIELDS = [
    f"{hand}_delta_{axis}"
    for hand in ("left", "right")
    for axis in ("x", "y", "z", "distance")
]


def normalize_quaternion(q):
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in q)


def quaternion_to_matrix(q):
    x, y, z, w = normalize_quaternion(q)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def vector_delta(a, b):
    return tuple(float(a[i] - b[i]) for i in range(3))


def vector_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


class CombinedHandWristCalibrationRecorder(Node):
    def __init__(self):
        super().__init__("combined_hand_wrist_calibration_recorder")

        self.declare_parameter("tf_topic", "/tf")
        self.declare_parameter("left_hand_tf_frame", "hand_left")
        self.declare_parameter("right_hand_tf_frame", "hand_right")
        self.declare_parameter(
            "output_csv",
            "/home/zaid/humanoid_ws/combined_hand_wrist_calibration.csv",
        )
        self.declare_parameter(
            "summary_csv",
            "/home/zaid/humanoid_ws/combined_hand_wrist_calibration_summary.csv",
        )
        self.declare_parameter("samples_per_step", 8)
        self.declare_parameter("sample_interval_sec", 0.20)
        self.declare_parameter("prepare_countdown_sec", 3.0)
        self.declare_parameter("use_speech", True)
        self.declare_parameter("speech_command", "auto")
        self.declare_parameter("debug", True)
        self.declare_parameter("include_rotation_matrix", True)
        self.declare_parameter("stationary_warning_m", 0.06)

        self.tf_topic = str(self.get_parameter("tf_topic").value)
        self.left_frame = str(self.get_parameter("left_hand_tf_frame").value)
        self.right_frame = str(self.get_parameter("right_hand_tf_frame").value)
        self.output_csv = Path(str(self.get_parameter("output_csv").value)).expanduser()
        self.summary_csv = Path(str(self.get_parameter("summary_csv").value)).expanduser()
        self.samples_per_step = int(self.get_parameter("samples_per_step").value)
        self.sample_interval_sec = float(
            self.get_parameter("sample_interval_sec").value
        )
        self.prepare_countdown_sec = float(
            self.get_parameter("prepare_countdown_sec").value
        )
        self.use_speech = bool(self.get_parameter("use_speech").value)
        self.speech_command_param = self.get_parameter("speech_command").value
        self.debug = bool(self.get_parameter("debug").value)
        self.include_rotation_matrix = bool(
            self.get_parameter("include_rotation_matrix").value
        )
        self.stationary_warning_m = float(
            self.get_parameter("stationary_warning_m").value
        )

        self.csv_fields = list(BASE_CSV_FIELDS)
        if self.include_rotation_matrix:
            self.csv_fields.extend(MATRIX_FIELDS)
        self.csv_fields.extend(DELTA_FIELDS)

        self.latest_transforms = {}
        self.last_missing_report = ""
        self.last_missing_report_time = 0.0
        self.samples_written = 0
        self.neutral_positions = {}
        self.step_summaries = []

        self.speech_command = self.detect_speech_command()
        self.tf_sub = self.create_subscription(
            TFMessage,
            self.tf_topic,
            self.tf_callback,
            50,
        )
        self.steps = self.build_steps()

        self.get_logger().info("Combined hand/wrist calibration recorder ready.")
        self.get_logger().info(f"Subscribing to TF topic: {self.tf_topic}")
        self.get_logger().info(
            f"Using frames: left={self.left_frame}, right={self.right_frame}"
        )
        self.get_logger().info(f"Output CSV: {self.output_csv}")
        self.get_logger().info(f"Summary CSV: {self.summary_csv}")
        self.get_logger().info(
            f"Countdown between prompts: {self.prepare_countdown_sec:.1f} seconds"
        )
        if self.use_speech and self.speech_command is None:
            self.get_logger().warn(
                "Speech command not found. Printing only. "
                "Install with: sudo apt install espeak-ng"
            )

    def normalize_frame_id(self, frame_id):
        return frame_id.lstrip("/")

    def tf_callback(self, msg):
        wanted = {
            self.normalize_frame_id(self.left_frame),
            self.normalize_frame_id(self.right_frame),
        }
        for transform in msg.transforms:
            child = self.normalize_frame_id(transform.child_frame_id)
            if child in wanted:
                self.latest_transforms[child] = transform

    def detect_speech_command(self):
        if not self.use_speech:
            return None

        requested = str(self.speech_command_param).strip()
        if requested and requested != "auto":
            if shutil.which(requested):
                return requested
            self.get_logger().warn(
                f"speech_command={requested} not found. Printing only."
            )
            return None

        for command in ("spd-say", "espeak-ng", "espeak"):
            if shutil.which(command):
                return command
        return None

    def say(self, text):
        print(text, flush=True)
        self.get_logger().info(text)
        if not self.use_speech or self.speech_command is None:
            return

        try:
            command = [self.speech_command, text]
            if self.speech_command == "spd-say":
                command = [self.speech_command, "--wait", text]
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as exc:
            self.get_logger().warn(f"Voice output failed: {exc}")

    def build_steps(self):
        steps = [
            {
                "step_name": "neutral",
                "active_hand": "both",
                "requested_direction": "neutral",
                "requested_rotation": "neutral",
                "instruction": "neutral",
            }
        ]

        directions = ("forward", "backward", "up", "down", "left", "right")
        rotations = ("clockwise", "anticlockwise")

        for hand in ("right", "left"):
            other = "left" if hand == "right" else "right"
            for direction in directions:
                steps.append(
                    {
                        "step_name": f"{hand}_{direction}",
                        "active_hand": hand,
                        "requested_direction": direction,
                        "requested_rotation": "none",
                        "instruction": f"{hand} hand {direction}",
                    }
                )
            for rotation in rotations:
                steps.append(
                    {
                        "step_name": f"{hand}_{rotation}",
                        "active_hand": hand,
                        "requested_direction": "none",
                        "requested_rotation": rotation,
                        "instruction": f"{hand} wrist {rotation}",
                    }
                )
        return steps

    def get_missing_frames(self):
        missing = []
        left_frame = self.normalize_frame_id(self.left_frame)
        right_frame = self.normalize_frame_id(self.right_frame)
        if left_frame not in self.latest_transforms:
            missing.append(self.left_frame)
        if right_frame not in self.latest_transforms:
            missing.append(self.right_frame)
        return missing

    def report_missing_frames(self, missing):
        message = "Waiting for missing TF frame(s): " + ", ".join(missing)
        now = time.monotonic()
        should_report = (
            message != self.last_missing_report
            or now - self.last_missing_report_time > 2.0
        )
        if should_report:
            print(message, flush=True)
            self.get_logger().warn(message)
            self.last_missing_report = message
            self.last_missing_report_time = now

    def spin_for(self, duration_sec):
        end_time = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < end_time:
            timeout = min(0.05, max(0.0, end_time - time.monotonic()))
            rclpy.spin_once(self, timeout_sec=timeout)

    def wait_for_frames(self):
        while rclpy.ok():
            missing = self.get_missing_frames()
            if not missing:
                return True
            self.report_missing_frames(missing)
            self.spin_for(0.1)
        return False

    def run_prepare_countdown(self):
        seconds = max(0, int(round(self.prepare_countdown_sec)))
        for remaining in range(seconds, 0, -1):
            self.say(str(remaining))
            self.spin_for(1.0)

    def get_current_pose(self):
        poses = {}
        for hand, frame in (("left", self.left_frame), ("right", self.right_frame)):
            transform = self.latest_transforms[self.normalize_frame_id(frame)]
            t = transform.transform.translation
            q_msg = transform.transform.rotation
            q = normalize_quaternion((q_msg.x, q_msg.y, q_msg.z, q_msg.w))
            poses[hand] = {
                "position": (float(t.x), float(t.y), float(t.z)),
                "quaternion": q,
                "matrix": quaternion_to_matrix(q),
            }
        return poses

    def make_pose_row(self, step_index, step, sample_index, poses):
        row = {
            "timestamp": f"{self.get_clock().now().nanoseconds * 1e-9:.9f}",
            "step_index": step_index,
            "step_name": step["step_name"],
            "active_hand": step["active_hand"],
            "requested_direction": step["requested_direction"],
            "requested_rotation": step["requested_rotation"],
            "sample_index": sample_index,
        }

        for hand in ("left", "right"):
            p = poses[hand]["position"]
            q = poses[hand]["quaternion"]
            row[f"{hand}_wrist_x"] = p[0]
            row[f"{hand}_wrist_y"] = p[1]
            row[f"{hand}_wrist_z"] = p[2]
            row[f"{hand}_wrist_qx"] = q[0]
            row[f"{hand}_wrist_qy"] = q[1]
            row[f"{hand}_wrist_qz"] = q[2]
            row[f"{hand}_wrist_qw"] = q[3]
            if self.include_rotation_matrix:
                matrix = poses[hand]["matrix"]
                for matrix_row in range(3):
                    for col in range(3):
                        row[f"{hand}_wrist_r{matrix_row}{col}"] = (
                            matrix[matrix_row][col]
                        )

            neutral = self.neutral_positions.get(hand)
            if neutral is None:
                delta = (0.0, 0.0, 0.0)
                distance = 0.0
            else:
                delta = vector_delta(p, neutral)
                distance = vector_distance(p, neutral)
            row[f"{hand}_delta_x"] = delta[0]
            row[f"{hand}_delta_y"] = delta[1]
            row[f"{hand}_delta_z"] = delta[2]
            row[f"{hand}_delta_distance"] = distance
        return row

    @staticmethod
    def mean_position(positions):
        count = float(len(positions))
        return tuple(
            sum(position[i] for position in positions) / count for i in range(3)
        )

    def warn_if_drifted(self, step, poses, step_start_positions):
        if step["active_hand"] == "both":
            return

        active = step["active_hand"]
        inactive = "left" if active == "right" else "right"
        for hand, label in ((active, "active"), (inactive, "inactive")):
            current = poses[hand]["position"]
            step_start = step_start_positions.get(hand)
            step_drift = (
                vector_distance(current, step_start) if step_start is not None else 0.0
            )
            if label == "inactive" and step_drift > self.stationary_warning_m:
                message = (
                    f"{hand} inactive hand moved {step_drift:.3f} m during "
                    f"{step['step_name']}."
                )
                print(message, flush=True)
                self.get_logger().warn(message)

    def summarize_step(self, step_index, step, captured_positions):
        means = {
            hand: self.mean_position(captured_positions[hand])
            for hand in ("left", "right")
        }
        summary = {
            "step_index": step_index,
            "step_name": step["step_name"],
            "active_hand": step["active_hand"],
            "requested_direction": step["requested_direction"],
            "requested_rotation": step["requested_rotation"],
        }
        for hand in ("left", "right"):
            neutral = self.neutral_positions.get(hand, means[hand])
            delta = vector_delta(means[hand], neutral)
            distance = vector_distance(means[hand], neutral)
            summary[f"{hand}_mean_x"] = means[hand][0]
            summary[f"{hand}_mean_y"] = means[hand][1]
            summary[f"{hand}_mean_z"] = means[hand][2]
            summary[f"{hand}_delta_x"] = delta[0]
            summary[f"{hand}_delta_y"] = delta[1]
            summary[f"{hand}_delta_z"] = delta[2]
            summary[f"{hand}_distance_m"] = distance
        self.step_summaries.append(summary)

        active = step["active_hand"]
        if active in ("left", "right"):
            self.get_logger().info(
                f"{step['step_name']} {active} movement from neutral: "
                f"dx={summary[f'{active}_delta_x']:.3f}, "
                f"dy={summary[f'{active}_delta_y']:.3f}, "
                f"dz={summary[f'{active}_delta_z']:.3f}, "
                f"distance={summary[f'{active}_distance_m']:.3f} m"
            )

    def record_step(self, csv_file, writer, step_index, step):
        self.say(step["instruction"])
        self.run_prepare_countdown()
        self.say("recording")

        if not self.wait_for_frames():
            return False
        step_start_positions = {
            hand: pose["position"] for hand, pose in self.get_current_pose().items()
        }

        captured_positions = {"left": [], "right": []}
        for sample_index in range(self.samples_per_step):
            if not self.wait_for_frames():
                return False

            poses = self.get_current_pose()
            self.warn_if_drifted(step, poses, step_start_positions)
            for hand in ("left", "right"):
                captured_positions[hand].append(poses[hand]["position"])

            row = self.make_pose_row(step_index, step, sample_index, poses)
            writer.writerow(row)
            csv_file.flush()
            self.samples_written += 1

            if self.debug:
                self.get_logger().info(
                    f"Saved sample {sample_index + 1}/{self.samples_per_step} "
                    f"for {step['step_name']}"
                )
            if sample_index < self.samples_per_step - 1:
                self.spin_for(self.sample_interval_sec)

        if step["step_name"] == "neutral":
            self.neutral_positions = {
                hand: self.mean_position(captured_positions[hand])
                for hand in ("left", "right")
            }
            self.get_logger().info(
                "Neutral wrist positions: "
                f"left={self.neutral_positions['left']}, "
                f"right={self.neutral_positions['right']}"
            )

        self.summarize_step(step_index, step, captured_positions)
        self.say("Saved")
        return True

    def write_summary_csv(self):
        if not self.step_summaries:
            return
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.step_summaries[0].keys())
        with self.summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.step_summaries)

    def run_routine(self):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.wait_for_frames():
            return False

        with self.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fields)
            writer.writeheader()
            csv_file.flush()

            for step_index, step in enumerate(self.steps):
                if not rclpy.ok():
                    return False
                if not self.record_step(csv_file, writer, step_index, step):
                    return False

        self.write_summary_csv()
        self.say("Combined hand and wrist calibration complete.")
        self.print_summary()
        return True

    def print_summary(self):
        lines = [
            "Combined hand/wrist calibration summary:",
            f"  output_csv: {self.output_csv}",
            f"  summary_csv: {self.summary_csv}",
            f"  steps: {len(self.steps)}",
            f"  samples: {self.samples_written}",
            f"  left_hand_tf_frame: {self.left_frame}",
            f"  right_hand_tf_frame: {self.right_frame}",
        ]
        for line in lines:
            print(line, flush=True)
            self.get_logger().info(line)


def main(args=None):
    rclpy.init(args=args)
    node = CombinedHandWristCalibrationRecorder()
    try:
        node.run_routine()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Combined hand/wrist calibration stopped.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
