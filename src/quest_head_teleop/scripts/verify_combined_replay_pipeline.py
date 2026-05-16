#!/usr/bin/env python3

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


NEW_ROOM_BASIS = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

POSITION_STEPS = {
    "forward": ("x", "+"),
    "backward": ("x", "-"),
    "left": ("y", "+"),
    "right": ("y", "-"),
    "up": ("z", "+"),
    "down": ("z", "-"),
}


def read_csv_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"No rows in CSV: {csv_path}")
    return rows, fieldnames


def wrist_position(row, side):
    return np.array(
        [
            float(row[f"{side}_wrist_x"]),
            float(row[f"{side}_wrist_y"]),
            float(row[f"{side}_wrist_z"]),
        ],
        dtype=float,
    )


def group_rows_by_step(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["step_index"]), row["step_name"])].append(row)
    return dict(sorted(grouped.items()))


def mean_position(step_rows, side):
    return np.mean([wrist_position(row, side) for row in step_rows], axis=0)


def sign_text(value, tolerance=0.01):
    if value > tolerance:
        return "+"
    if value < -tolerance:
        return "-"
    return "0"


def analyze_expected_mapping(csv_path):
    rows, fieldnames = read_csv_rows(csv_path)
    grouped = group_rows_by_step(rows)
    neutral_key = next((key for key in grouped if key[1] == "neutral"), None)
    if neutral_key is None:
        raise RuntimeError("CSV does not contain a neutral step.")

    neutral = {
        side: mean_position(grouped[neutral_key], side)
        for side in ("left", "right")
    }

    lines = [
        f"CSV: {csv_path}",
        f"Rows: {len(rows)}",
        f"Columns: {len(fieldnames)}",
        "First step: "
        f"{rows[0].get('step_name')} active={rows[0].get('active_hand')} "
        f"direction={rows[0].get('requested_direction')} "
        f"rotation={rows[0].get('requested_rotation')}",
        "Required wrist pose fields present: "
        + str(
            all(
                field in fieldnames
                for side in ("left", "right")
                for field in (
                    f"{side}_wrist_x",
                    f"{side}_wrist_y",
                    f"{side}_wrist_z",
                    f"{side}_wrist_qx",
                    f"{side}_wrist_qy",
                    f"{side}_wrist_qz",
                    f"{side}_wrist_qw",
                )
            )
        ),
        "",
        "Expected robot movement signs using position_basis_variant=new_room:",
    ]

    expected = {}
    for (step_index, step_name), step_rows in grouped.items():
        if step_name == "neutral":
            continue
        first = step_rows[0]
        active = first["active_hand"]
        direction = first.get("requested_direction", "")
        if active not in ("left", "right") or direction not in POSITION_STEPS:
            continue
        hand_delta = mean_position(step_rows, active) - neutral[active]
        robot_delta = NEW_ROOM_BASIS @ hand_delta
        dominant_index = int(np.argmax(np.abs(robot_delta)))
        dominant_axis = "xyz"[dominant_index]
        dominant_sign = sign_text(robot_delta[dominant_index])
        expected_axis, expected_sign = POSITION_STEPS[direction]
        ok = dominant_axis == expected_axis and dominant_sign == expected_sign
        expected[step_name] = {
            "active": active,
            "direction": direction,
            "hand_delta": hand_delta,
            "robot_delta": robot_delta,
            "dominant": dominant_axis + dominant_sign,
            "expected": expected_axis + expected_sign,
            "ok": ok,
        }
        lines.append(
            f"{step_index:02d} {step_name:16s} active={active:5s} "
            f"hand_delta={np.array2string(hand_delta, precision=3)} "
            f"robot_delta={np.array2string(robot_delta, precision=3)} "
            f"dominant={dominant_axis}{dominant_sign} "
            f"expected={expected_axis}{expected_sign} ok={ok}"
        )

    return lines, expected


class PipelineMonitor(Node):
    def __init__(self, tf_topic):
        super().__init__("combined_replay_pipeline_monitor")
        self.tf_messages = 0
        self.left_tf = 0
        self.right_tf = 0
        self.left_cmds = []
        self.right_cmds = []
        self.joint_states = 0
        self.first_time = time.monotonic()
        self.last_time = self.first_time
        self.create_subscription(TFMessage, tf_topic, self.tf_callback, 50)
        self.create_subscription(
            JointState,
            "/g1/left_arm_cmd",
            lambda msg: self.command_callback("left", msg),
            50,
        )
        self.create_subscription(
            JointState,
            "/g1/right_arm_cmd",
            lambda msg: self.command_callback("right", msg),
            50,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            50,
        )

    def tf_callback(self, msg):
        self.tf_messages += 1
        self.last_time = time.monotonic()
        for transform in msg.transforms:
            child = transform.child_frame_id.lstrip("/")
            if child == "hand_left":
                self.left_tf += 1
            elif child == "hand_right":
                self.right_tf += 1

    def command_callback(self, side, msg):
        values = np.asarray(msg.position, dtype=float)
        if side == "left":
            self.left_cmds.append(values)
        else:
            self.right_cmds.append(values)
        self.last_time = time.monotonic()

    def joint_state_callback(self, _msg):
        self.joint_states += 1
        self.last_time = time.monotonic()


def command_stats(values):
    if not values:
        return "count=0 changed=False range_norm=0.000 nan=False"
    count = len(values)
    finite = [v for v in values if np.all(np.isfinite(v))]
    has_nan = len(finite) != len(values)
    if not finite:
        return f"count={count} changed=False range_norm=nan nan=True"
    min_len = min(len(v) for v in finite)
    arr = np.asarray([v[:min_len] for v in finite], dtype=float)
    range_norm = float(np.linalg.norm(np.ptp(arr, axis=0)))
    changed = range_norm > 1e-4
    return (
        f"count={count} changed={changed} "
        f"range_norm={range_norm:.6f} nan={has_nan}"
    )


def monitor_topics(tf_topic, duration_sec):
    rclpy.init()
    node = PipelineMonitor(tf_topic)
    try:
        end_time = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(node, timeout_sec=0.1)
        elapsed = max(1e-6, time.monotonic() - node.first_time)
        lines = [
            "",
            f"Monitor duration: {elapsed:.2f} sec",
            f"TF received: {node.tf_messages > 0}",
            f"TF messages: {node.tf_messages}",
            f"hand_left TF count: {node.left_tf}",
            f"hand_right TF count: {node.right_tf}",
            f"Approx TF rate: {node.tf_messages / elapsed:.2f} Hz",
            f"Left arm commands received: {len(node.left_cmds) > 0}",
            "Left arm command stats: " + command_stats(node.left_cmds),
            f"Right arm commands received: {len(node.right_cmds) > 0}",
            "Right arm command stats: " + command_stats(node.right_cmds),
            f"Joint states received: {node.joint_states > 0}",
            f"Joint state count: {node.joint_states}",
            f"Approx joint state rate: {node.joint_states / elapsed:.2f} Hz",
        ]
        return lines
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Verify combined hand/wrist replay CSV and optionally monitor ROS topics."
    )
    parser.add_argument(
        "--csv",
        default="/home/zaid/humanoid_ws/combined_hand_wrist_calibration.csv",
    )
    parser.add_argument(
        "--report",
        default="/home/zaid/humanoid_ws/replay_verification_report.txt",
    )
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--monitor-sec", type=float, default=0.0)
    args = parser.parse_args()

    lines, _expected = analyze_expected_mapping(args.csv)
    if args.monitor_sec > 0.0:
        lines.extend(monitor_topics(args.tf_topic, args.monitor_sec))

    text = "\n".join(lines) + "\n"
    Path(args.report).write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Report written to: {args.report}")


if __name__ == "__main__":
    main()
