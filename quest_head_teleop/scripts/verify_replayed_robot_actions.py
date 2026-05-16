#!/usr/bin/env python3

import argparse
import csv
import math
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


EXPECTED_ACTIONS = {
    "right_forward": ("right", "x", 1),
    "right_backward": ("right", "x", -1),
    "right_left": ("right", "y", 1),
    "right_right": ("right", "y", -1),
    "right_up": ("right", "z", 1),
    "right_down": ("right", "z", -1),
    "left_forward": ("left", "x", 1),
    "left_backward": ("left", "x", -1),
    "left_left": ("left", "y", 1),
    "left_right": ("left", "y", -1),
    "left_up": ("left", "z", 1),
    "left_down": ("left", "z", -1),
}

AXES = ("x", "y", "z")
SIGN_TEXT = {1: "+", -1: "-", 0: "0"}
NEW_ROOM_BASIS = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def ordered_steps_from_csv(csv_path):
    csv_path = Path(csv_path).expanduser()
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "step_name" not in reader.fieldnames:
            raise RuntimeError(f"CSV must contain a step_name column: {csv_path}")
        steps = OrderedDict()
        for row in reader:
            step_name = row.get("step_name", "").strip()
            if step_name:
                steps.setdefault(step_name, None)

    if "neutral" not in steps:
        raise RuntimeError("CSV does not contain a neutral step.")

    return list(steps.keys())


def detect_position_column(fieldnames, side, component):
    candidates = (
        f"{side}_wrist_{component}",
        f"{side}_{component}",
        f"{side}_position_{component}",
        f"{side}_hand_{component}",
    )
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise RuntimeError(f"Missing {side} {component} position column in CSV.")


def csv_step_means(csv_path):
    csv_path = Path(csv_path).expanduser()
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        columns = {
            side: {
                axis: detect_position_column(fieldnames, side, axis)
                for axis in AXES
            }
            for side in ("left", "right")
        }
        grouped = OrderedDict()
        for row in reader:
            step_name = row.get("step_name", "").strip()
            if not step_name:
                continue
            grouped.setdefault(step_name, {"left": [], "right": []})
            for side in ("left", "right"):
                grouped[step_name][side].append(
                    np.asarray(
                        [
                            float(row[columns[side]["x"]]),
                            float(row[columns[side]["y"]]),
                            float(row[columns[side]["z"]]),
                        ],
                        dtype=float,
                    )
                )

    means = {}
    for step_name, side_values in grouped.items():
        means[step_name] = {}
        for side in ("left", "right"):
            if side_values[side]:
                means[step_name][side] = np.mean(side_values[side], axis=0)
    return means


def dominant_axis_and_sign(delta, sign_tolerance):
    abs_delta = np.abs(delta)
    dominant_index = int(np.argmax(abs_delta))
    dominant_axis = AXES[dominant_index]
    dominant_sign = sign(float(delta[dominant_index]), sign_tolerance)
    total = float(np.linalg.norm(delta))
    confidence = float(abs_delta[dominant_index] / total) if total > 1e-9 else 0.0
    sorted_abs = np.sort(abs_delta)
    margin = float(sorted_abs[-1] - sorted_abs[-2]) if len(sorted_abs) > 1 else 0.0
    return dominant_axis, dominant_sign, confidence, margin


def side_vector(args, side, name):
    return np.asarray(
        [
            getattr(args, f"{side}_{name}_x"),
            getattr(args, f"{side}_{name}_y"),
            getattr(args, f"{side}_{name}_z"),
        ],
        dtype=float,
    )


def build_mapper_target_analysis(args):
    means = csv_step_means(args.csv)
    if "neutral" not in means:
        return {}

    gain = np.asarray(
        [
            args.position_gain_x,
            args.position_gain_y,
            args.position_gain_z,
        ],
        dtype=float,
    ) * args.global_position_gain
    neutral_robot = {
        "left": side_vector(args, "left", "neutral"),
        "right": side_vector(args, "right", "neutral"),
    }
    workspace_min = {
        "left": side_vector(args, "left", "workspace_min"),
        "right": side_vector(args, "right", "workspace_min"),
    }
    workspace_max = {
        "left": side_vector(args, "left", "workspace_max"),
        "right": side_vector(args, "right", "workspace_max"),
    }

    analysis = {}
    for step_name, (side, _axis, _sign) in EXPECTED_ACTIONS.items():
        if step_name not in means or side not in means[step_name]:
            continue
        if side not in means["neutral"]:
            continue
        delta_hand = means[step_name][side] - means["neutral"][side]
        delta_before_gain = NEW_ROOM_BASIS @ delta_hand
        delta_after_gain = delta_before_gain * gain
        target_before_clip = neutral_robot[side] + delta_after_gain
        lower = np.minimum(workspace_min[side], workspace_max[side])
        upper = np.maximum(workspace_min[side], workspace_max[side])
        target_after_clip = np.clip(target_before_clip, lower, upper)
        target_delta_after_clip = target_after_clip - neutral_robot[side]
        dominant_axis, dominant_sign, confidence, margin = dominant_axis_and_sign(
            target_delta_after_clip,
            args.sign_tolerance,
        )
        analysis[step_name] = {
            "side": side,
            "delta_hand": delta_hand,
            "delta_before_gain": delta_before_gain,
            "delta_after_gain": delta_after_gain,
            "target_before_clip": target_before_clip,
            "target_after_clip": target_after_clip,
            "clipped": not np.allclose(target_before_clip, target_after_clip),
            "dominant_axis": dominant_axis,
            "dominant_sign": SIGN_TEXT[dominant_sign],
            "confidence": confidence,
            "margin": margin,
        }
    return analysis


def vector_from_pose(msg):
    return np.asarray(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
        dtype=float,
    )


def sign(value, tolerance):
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def format_vec(vec):
    return f"[{vec[0]:+.4f}, {vec[1]:+.4f}, {vec[2]:+.4f}]"


class ReplayedActionVerifier(Node):
    def __init__(self, args, expected_steps):
        super().__init__("replayed_robot_action_verifier")
        self.args = args
        self.expected_steps = expected_steps
        self.image_dir = Path(args.image_dir).expanduser()
        self.current_step = None
        self.previous_step = None
        self.completed_steps = set()
        self.samples = defaultdict(lambda: {"left": [], "right": []})
        self.seen_steps = OrderedDict()
        self.screenshot_paths = {}
        self.screenshot_futures = {}
        self.service_warned = False
        self.mapper_target_analysis = build_mapper_target_analysis(args)

        self.create_subscription(
            String,
            "/replay/current_step",
            self.step_callback,
            50,
        )
        self.create_subscription(
            PoseStamped,
            "/g1/left_wrist_pose",
            lambda msg: self.pose_callback("left", msg),
            100,
        )
        self.create_subscription(
            PoseStamped,
            "/g1/right_wrist_pose",
            lambda msg: self.pose_callback("right", msg),
            100,
        )
        self.screenshot_client = self.create_client(
            Trigger,
            "/g1_ros2_bridge/save_screenshot",
        )

    def step_callback(self, msg):
        step_name = msg.data.strip()
        if not step_name:
            return
        if self.previous_step is not None and step_name != self.previous_step:
            if self.previous_step in self.expected_steps:
                self.completed_steps.add(self.previous_step)
        self.previous_step = step_name
        self.current_step = step_name
        self.seen_steps.setdefault(step_name, time.monotonic())

    def pose_callback(self, side, msg):
        if not self.current_step:
            return
        if self.current_step not in self.expected_steps:
            return
        self.samples[self.current_step][side].append(vector_from_pose(msg))

    def maybe_request_screenshots(self):
        for step_name in self.expected_steps:
            if step_name != self.current_step:
                continue
            if step_name in self.screenshot_paths:
                continue
            if step_name in self.screenshot_futures:
                continue
            if not self.has_screenshot_samples(step_name):
                continue
            if not self.screenshot_client.service_is_ready():
                if not self.service_warned:
                    self.get_logger().warn(
                        "Screenshot service is not ready; numeric verification will continue."
                    )
                    self.service_warned = True
                continue

            future = self.screenshot_client.call_async(Trigger.Request())
            self.screenshot_futures[step_name] = future

        done = []
        for step_name, future in self.screenshot_futures.items():
            if not future.done():
                continue
            done.append(step_name)
            try:
                response = future.result()
            except Exception as exc:
                self.screenshot_paths[step_name] = ""
                self.get_logger().warn(f"Screenshot failed for {step_name}: {exc}")
                continue

            if response.success:
                self.screenshot_paths[step_name] = response.message
            else:
                self.screenshot_paths[step_name] = ""
                self.get_logger().warn(
                    f"Screenshot failed for {step_name}: {response.message}"
                )

        for step_name in done:
            del self.screenshot_futures[step_name]

    def has_min_samples(self, step_name):
        sides = self.samples.get(step_name, {})
        if step_name == "neutral":
            return (
                len(sides.get("left", [])) >= self.args.min_samples
                and len(sides.get("right", [])) >= self.args.min_samples
            )
        side = EXPECTED_ACTIONS.get(step_name, ("", "", 0))[0]
        return len(sides.get(side, [])) >= self.args.min_samples

    def has_screenshot_samples(self, step_name):
        sides = self.samples.get(step_name, {})
        if step_name == "neutral":
            return (
                len(sides.get("left", [])) >= self.args.screenshot_samples
                and len(sides.get("right", [])) >= self.args.screenshot_samples
            )
        side = EXPECTED_ACTIONS.get(step_name, ("", "", 0))[0]
        return len(sides.get(side, [])) >= self.args.screenshot_samples

    def complete(self):
        required = ["neutral"] + [
            step for step in self.expected_steps if step in EXPECTED_ACTIONS
        ]
        return all(
            self.has_min_samples(step) and step in self.completed_steps
            for step in required
        )

    def mean_pose(self, step_name, side):
        values = self.samples.get(step_name, {}).get(side, [])
        if not values:
            return None
        return np.mean(values, axis=0)

    def build_results(self):
        rows = []
        neutral_left = self.mean_pose("neutral", "left")
        neutral_right = self.mean_pose("neutral", "right")
        if neutral_left is None or neutral_right is None:
            raise RuntimeError("No neutral left/right MuJoCo wrist pose samples were collected.")

        rows.append(
            {
                "step_name": "neutral",
                "side": "both",
                "expected_axis": "",
                "expected_sign": "",
                "actual_dx": 0.0,
                "actual_dy": 0.0,
                "actual_dz": 0.0,
                "dominant_axis": "",
                "dominant_sign": "",
                "pass_fail": "BASELINE",
                "confidence": 1.0,
                "image_path": self.image_path_for("neutral"),
                "notes": (
                    f"left_samples={len(self.samples['neutral']['left'])}; "
                    f"right_samples={len(self.samples['neutral']['right'])}"
                ),
            }
        )

        for step_name in self.expected_steps:
            if step_name not in EXPECTED_ACTIONS:
                continue
            side, expected_axis, expected_sign = EXPECTED_ACTIONS[step_name]
            neutral = neutral_right if side == "right" else neutral_left
            action_mean = self.mean_pose(step_name, side)
            if action_mean is None:
                rows.append(self.missing_row(step_name, side, expected_axis, expected_sign))
                continue

            delta = action_mean - neutral
            dominant_axis, dominant_sign, confidence, margin = dominant_axis_and_sign(
                delta,
                self.args.sign_tolerance,
            )
            total = float(np.linalg.norm(delta))
            mapper = self.mapper_target_analysis.get(step_name, {})

            if total < self.args.min_movement:
                result = "UNCLEAR"
                notes = self.add_mapper_note(
                    mapper,
                    f"movement below threshold; samples={len(self.samples[step_name][side])}",
                )
            elif confidence < self.args.min_confidence or margin < self.args.min_axis_margin:
                result = "UNCLEAR"
                notes = self.add_mapper_note(
                    mapper,
                    f"dominant axis not strong enough; margin={margin:.4f}; "
                    f"samples={len(self.samples[step_name][side])}"
                )
            elif dominant_axis == expected_axis and dominant_sign == expected_sign:
                result = "PASS"
                notes = self.add_mapper_note(
                    mapper,
                    f"samples={len(self.samples[step_name][side])}",
                )
            else:
                result = "FAIL"
                notes = self.add_mapper_note(
                    mapper,
                    f"samples={len(self.samples[step_name][side])}",
                )

            rows.append(
                {
                    "step_name": step_name,
                    "side": side,
                    "expected_axis": expected_axis,
                    "expected_sign": SIGN_TEXT[expected_sign],
                    "actual_dx": float(delta[0]),
                    "actual_dy": float(delta[1]),
                    "actual_dz": float(delta[2]),
                    "dominant_axis": dominant_axis,
                    "dominant_sign": SIGN_TEXT[dominant_sign],
                    "mapper_dx": mapper.get("delta_after_gain", [math.nan] * 3)[0],
                    "mapper_dy": mapper.get("delta_after_gain", [math.nan] * 3)[1],
                    "mapper_dz": mapper.get("delta_after_gain", [math.nan] * 3)[2],
                    "mapper_dominant_axis": mapper.get("dominant_axis", ""),
                    "mapper_dominant_sign": mapper.get("dominant_sign", ""),
                    "target_before_clip_x": mapper.get("target_before_clip", [math.nan] * 3)[0],
                    "target_before_clip_y": mapper.get("target_before_clip", [math.nan] * 3)[1],
                    "target_before_clip_z": mapper.get("target_before_clip", [math.nan] * 3)[2],
                    "target_after_clip_x": mapper.get("target_after_clip", [math.nan] * 3)[0],
                    "target_after_clip_y": mapper.get("target_after_clip", [math.nan] * 3)[1],
                    "target_after_clip_z": mapper.get("target_after_clip", [math.nan] * 3)[2],
                    "mapper_clipped": mapper.get("clipped", ""),
                    "pass_fail": result,
                    "confidence": confidence,
                    "image_path": self.image_path_for(step_name),
                    "notes": notes,
                }
            )

        return rows

    @staticmethod
    def add_mapper_note(mapper, note):
        if not mapper:
            return note
        pieces = [
            note,
            (
                "mapper_target="
                f"{mapper.get('dominant_sign', '')}{str(mapper.get('dominant_axis', '')).upper()}"
            ),
        ]
        if mapper.get("clipped"):
            pieces.append("mapper_target_clipped=True")
        return "; ".join(pieces)

    def image_path_for(self, step_name):
        return self.screenshot_paths.get(
            step_name,
            str(self.image_dir / f"{step_name}.png"),
        )

    def missing_row(self, step_name, side, expected_axis, expected_sign):
        return {
            "step_name": step_name,
            "side": side,
            "expected_axis": expected_axis,
            "expected_sign": SIGN_TEXT[expected_sign],
            "actual_dx": math.nan,
            "actual_dy": math.nan,
            "actual_dz": math.nan,
            "dominant_axis": "",
            "dominant_sign": "",
            "mapper_dx": math.nan,
            "mapper_dy": math.nan,
            "mapper_dz": math.nan,
            "mapper_dominant_axis": "",
            "mapper_dominant_sign": "",
            "target_before_clip_x": math.nan,
            "target_before_clip_y": math.nan,
            "target_before_clip_z": math.nan,
            "target_after_clip_x": math.nan,
            "target_after_clip_y": math.nan,
            "target_after_clip_z": math.nan,
            "mapper_clipped": "",
            "pass_fail": "UNCLEAR",
            "confidence": 0.0,
            "image_path": self.image_path_for(step_name),
            "notes": "no MuJoCo wrist pose samples collected for this step",
        }


def write_reports(rows, text_report, csv_report):
    text_report = Path(text_report).expanduser()
    csv_report = Path(csv_report).expanduser()
    text_report.parent.mkdir(parents=True, exist_ok=True)
    csv_report.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Replayed Robot Action Verification",
        "",
        "Numeric result is based on actual MuJoCo wrist pose topics:",
        "/g1/left_wrist_pose and /g1/right_wrist_pose",
        "",
    ]
    for row in rows:
        if row["step_name"] == "neutral":
            lines.extend(
                [
                    "neutral:",
                    f"  result: {row['pass_fail']}",
                    f"  image: {row['image_path']}",
                    f"  notes: {row['notes']}",
                    "",
                ]
            )
            continue
        delta = np.asarray([row["actual_dx"], row["actual_dy"], row["actual_dz"]])
        mapper_delta = np.asarray([row["mapper_dx"], row["mapper_dy"], row["mapper_dz"]])
        lines.extend(
            [
                f"{row['step_name']}:",
                (
                    f"  expected: {row['side']} wrist "
                    f"{row['expected_sign']}{row['expected_axis'].upper()}"
                ),
                f"  mapper_target_delta: {format_vec(mapper_delta)}",
                (
                    f"  mapper_target_dominant: {row['mapper_dominant_sign']}"
                    f"{str(row['mapper_dominant_axis']).upper()}"
                ),
                (
                    "  mapper_target_clipped: "
                    f"{row['mapper_clipped']}"
                ),
                f"  actual_delta: {format_vec(delta)}",
                (
                    f"  dominant: {row['dominant_sign']}"
                    f"{row['dominant_axis'].upper()}"
                ),
                f"  result: {row['pass_fail']}",
                f"  confidence: {row['confidence']:.3f}",
                f"  image: {row['image_path']}",
                f"  notes: {row['notes']}",
                "",
            ]
        )

    text_report.write_text("\n".join(lines), encoding="utf-8")

    fieldnames = [
        "step_name",
        "side",
        "expected_axis",
        "expected_sign",
        "actual_dx",
        "actual_dy",
        "actual_dz",
        "dominant_axis",
        "dominant_sign",
        "mapper_dx",
        "mapper_dy",
        "mapper_dz",
        "mapper_dominant_axis",
        "mapper_dominant_sign",
        "target_before_clip_x",
        "target_before_clip_y",
        "target_before_clip_z",
        "target_after_clip_x",
        "target_after_clip_y",
        "target_after_clip_z",
        "mapper_clipped",
        "pass_fail",
        "confidence",
        "image_path",
        "notes",
    ]
    with csv_report.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Verify recorded teleoperation actions using actual MuJoCo wrist poses."
    )
    parser.add_argument(
        "--csv",
        default="/home/zaid/humanoid_ws/combined_hand_wrist_calibration.csv",
    )
    parser.add_argument(
        "--image-dir",
        default="/home/zaid/humanoid_ws/replay_action_images",
    )
    parser.add_argument(
        "--report",
        default="/home/zaid/humanoid_ws/replayed_robot_action_report.txt",
    )
    parser.add_argument(
        "--csv-report",
        default="/home/zaid/humanoid_ws/replayed_robot_action_report.csv",
    )
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--screenshot-samples", type=int, default=20)
    parser.add_argument("--min-movement", type=float, default=0.01)
    parser.add_argument("--sign-tolerance", type=float, default=0.003)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-axis-margin", type=float, default=0.005)
    parser.add_argument("--global-position-gain", type=float, default=0.6)
    parser.add_argument("--position-gain-x", type=float, default=1.3)
    parser.add_argument("--position-gain-y", type=float, default=1.3)
    parser.add_argument("--position-gain-z", type=float, default=1.2)
    parser.add_argument("--left-neutral-x", type=float, default=0.32)
    parser.add_argument("--left-neutral-y", type=float, default=0.20)
    parser.add_argument("--left-neutral-z", type=float, default=0.20)
    parser.add_argument("--right-neutral-x", type=float, default=0.32)
    parser.add_argument("--right-neutral-y", type=float, default=-0.20)
    parser.add_argument("--right-neutral-z", type=float, default=0.20)
    parser.add_argument("--left-workspace-min-x", type=float, default=0.08)
    parser.add_argument("--left-workspace-min-y", type=float, default=0.02)
    parser.add_argument("--left-workspace-min-z", type=float, default=-0.10)
    parser.add_argument("--left-workspace-max-x", type=float, default=0.65)
    parser.add_argument("--left-workspace-max-y", type=float, default=0.55)
    parser.add_argument("--left-workspace-max-z", type=float, default=0.60)
    parser.add_argument("--right-workspace-min-x", type=float, default=0.08)
    parser.add_argument("--right-workspace-min-y", type=float, default=-0.55)
    parser.add_argument("--right-workspace-min-z", type=float, default=-0.10)
    parser.add_argument("--right-workspace-max-x", type=float, default=0.65)
    parser.add_argument("--right-workspace-max-y", type=float, default=-0.02)
    parser.add_argument("--right-workspace-max-z", type=float, default=0.60)
    args = parser.parse_args()

    expected_steps = ordered_steps_from_csv(args.csv)
    expected_steps = [
        step for step in expected_steps if step == "neutral" or step in EXPECTED_ACTIONS
    ]

    rclpy.init()
    node = ReplayedActionVerifier(args, expected_steps)
    deadline = time.monotonic() + max(1.0, args.timeout_sec)
    try:
        node.get_logger().info(
            "Waiting for replay steps and actual MuJoCo wrist poses. "
            "Start this before or during CSV replay."
        )
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.maybe_request_screenshots()
            if node.complete():
                break

        # Give pending screenshot service calls one short chance to finish.
        grace_deadline = time.monotonic() + 3.0
        while rclpy.ok() and node.screenshot_futures and time.monotonic() < grace_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.maybe_request_screenshots()

        rows = node.build_results()
        write_reports(rows, args.report, args.csv_report)

        print(f"Text report written to: {args.report}")
        print(f"CSV report written to: {args.csv_report}")
        print(f"Images directory: {args.image_dir}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
