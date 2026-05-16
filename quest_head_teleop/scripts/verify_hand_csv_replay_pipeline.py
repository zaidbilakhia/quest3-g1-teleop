#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict

import numpy as np


BASIS_VARIANTS = {
    "identity": np.eye(3, dtype=float),
    "old": np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    ),
    "flip_x": np.diag([-1.0, 1.0, 1.0]),
    "flip_y": np.diag([1.0, -1.0, 1.0]),
    "flip_z": np.diag([1.0, 1.0, -1.0]),
}


def mean(values):
    return np.mean(np.asarray(values, dtype=float), axis=0)


def load_step_means(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows_by_step = defaultdict(lambda: {"left": [], "right": []})
        for row in reader:
            step = row["step_name"]
            rows_by_step[step]["left"].append(
                [
                    float(row["left_wrist_x"]),
                    float(row["left_wrist_y"]),
                    float(row["left_wrist_z"]),
                ]
            )
            rows_by_step[step]["right"].append(
                [
                    float(row["right_wrist_x"]),
                    float(row["right_wrist_y"]),
                    float(row["right_wrist_z"]),
                ]
            )

    return {
        step: {side: mean(values) for side, values in sides.items()}
        for step, sides in rows_by_step.items()
    }


def sign_text(value, tolerance=0.01):
    if value > tolerance:
        return "+"
    if value < -tolerance:
        return "-"
    return "0"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize expected robot target deltas from hand calibration CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/home/zaid/humanoid_ws/hand_axis_calibration.csv",
    )
    parser.add_argument("--position-basis-variant", default="identity")
    parser.add_argument("--gain-x", type=float, default=1.3)
    parser.add_argument("--gain-y", type=float, default=1.3)
    parser.add_argument("--gain-z", type=float, default=1.2)
    args = parser.parse_args()

    basis = BASIS_VARIANTS.get(args.position_basis_variant)
    if basis is None:
        raise SystemExit(
            f"Unsupported basis {args.position_basis_variant!r}; "
            f"choose one of {sorted(BASIS_VARIANTS)}"
        )

    step_means = load_step_means(args.csv_path)
    neutral = step_means.get("neutral")
    if neutral is None:
        raise SystemExit("CSV does not contain a neutral step.")

    gain = np.array([args.gain_x, args.gain_y, args.gain_z], dtype=float)
    print(f"CSV: {args.csv_path}")
    print(f"position_basis_variant: {args.position_basis_variant}")
    print(f"gain: {gain}")
    print("")
    print("Expected robot target deltas from neutral")
    for step in sorted(step_means):
        if step == "neutral":
            continue
        active_side = "left" if step.startswith("left_") else "right"
        hand_delta = step_means[step][active_side] - neutral[active_side]
        robot_delta = (basis @ hand_delta) * gain
        signs = "".join(sign_text(value) for value in robot_delta)
        print(
            f"{step:15s} side={active_side:5s} "
            f"hand_delta={np.array2string(hand_delta, precision=3)} "
            f"robot_delta={np.array2string(robot_delta, precision=3)} "
            f"sign_xyz={signs}"
        )


if __name__ == "__main__":
    main()
