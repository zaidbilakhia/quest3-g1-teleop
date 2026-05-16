#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict


POSITION_FIELDS = {
    'left': ('left_wrist_x', 'left_wrist_y', 'left_wrist_z'),
    'right': ('right_wrist_x', 'right_wrist_y', 'right_wrist_z'),
}


def vector_length(delta):
    return math.sqrt(sum(value * value for value in delta))


def format_vec(values):
    return f'x={values[0]: .5f}, y={values[1]: .5f}, z={values[2]: .5f}'


def dominant_axis(delta):
    labels = ('x', 'y', 'z')
    index = max(range(3), key=lambda i: abs(delta[i]))
    return labels[index], delta[index]


def mean_position(rows, hand):
    fields = POSITION_FIELDS[hand]
    count = float(len(rows))
    return tuple(
        sum(float(row[field]) for row in rows) / count
        for field in fields
    )


def subtract(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def load_rows(csv_path):
    with open(csv_path, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f'No rows found in {csv_path}')

    required = {
        'step_index',
        'step_name',
        'active_hand',
        'requested_direction',
        *POSITION_FIELDS['left'],
        *POSITION_FIELDS['right'],
    }
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise RuntimeError(
            'CSV is missing required column(s): ' + ', '.join(missing)
        )

    return rows


def group_by_step(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (int(row['step_index']), row['step_name'])
        groups[key].append(row)
    return dict(sorted(groups.items()))


def print_step_means(step_means):
    print('Mean wrist position per step')
    for (step_index, step_name), means in step_means.items():
        print(
            f'  {step_index:02d} {step_name:16s} '
            f'left({format_vec(means["left"])})  '
            f'right({format_vec(means["right"])})'
        )


def print_deltas(step_means, neutral):
    print()
    print('Delta from neutral')
    for (step_index, step_name), means in step_means.items():
        left_delta = subtract(means['left'], neutral['left'])
        right_delta = subtract(means['right'], neutral['right'])
        print(
            f'  {step_index:02d} {step_name:16s} '
            f'left_d({format_vec(left_delta)})  '
            f'right_d({format_vec(right_delta)})'
        )


def print_hand_motion_summary(step_means, rows_by_step, neutral):
    print()
    print('Movement summary')
    for key, means in step_means.items():
        step_index, step_name = key
        first = rows_by_step[key][0]
        active = first['active_hand']
        direction = first['requested_direction']

        if active not in ('left', 'right'):
            continue

        still = 'left' if active == 'right' else 'right'
        active_delta = subtract(means[active], neutral[active])
        still_delta = subtract(means[still], neutral[still])
        axis, value = dominant_axis(active_delta)

        print(
            f'  {step_index:02d} {active:5s} {direction:8s}: '
            f'{active}_delta({format_vec(active_delta)}), '
            f'dominant={axis} {value:+.5f}, '
            f'{still}_accidental_move={vector_length(still_delta):.5f} m '
            f'({format_vec(still_delta)})'
        )


def print_mirror_hints(step_means):
    right_by_direction = {}
    left_by_direction = {}
    neutral = step_means[(0, 'neutral')]

    for (step_index, step_name), means in step_means.items():
        del step_index
        if step_name.startswith('right_'):
            direction = step_name.removeprefix('right_')
            right_by_direction[direction] = subtract(
                means['right'],
                neutral['right'],
            )
        elif step_name.startswith('left_'):
            direction = step_name.removeprefix('left_')
            left_by_direction[direction] = subtract(
                means['left'],
                neutral['left'],
            )

    print()
    print('Left/right mirror hints')
    for direction in ('right', 'left', 'forward', 'backward', 'up', 'down'):
        if (
            direction not in right_by_direction
            or direction not in left_by_direction
        ):
            continue
        right_delta = right_by_direction[direction]
        left_delta = left_by_direction[direction]
        signs = []
        axis_values = zip(('x', 'y', 'z'), left_delta, right_delta)
        for axis, left_value, right_value in axis_values:
            if abs(left_value) < 1e-6 or abs(right_value) < 1e-6:
                signs.append(f'{axis}: weak')
            elif left_value * right_value < 0.0:
                signs.append(f'{axis}: mirrored')
            else:
                signs.append(f'{axis}: same')
        print(
            f'  {direction:8s}: '
            f'right({format_vec(right_delta)}), '
            f'left({format_vec(left_delta)})  '
            + ', '.join(signs)
        )


def analyze(csv_path):
    rows = load_rows(csv_path)
    rows_by_step = group_by_step(rows)
    step_means = {}

    for key, step_rows in rows_by_step.items():
        step_means[key] = {
            'left': mean_position(step_rows, 'left'),
            'right': mean_position(step_rows, 'right'),
        }

    neutral_key = (0, 'neutral')
    if neutral_key not in step_means:
        raise RuntimeError(
            'Neutral step not found. Expected step_index=0, step_name=neutral'
        )
    neutral = step_means[neutral_key]

    print(f'Loaded {len(rows)} samples from {csv_path}')
    print(f'Found {len(step_means)} recorded steps')
    print()
    print_step_means(step_means)
    print_deltas(step_means, neutral)
    print_hand_motion_summary(step_means, rows_by_step, neutral)
    print_mirror_hints(step_means)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Quest hand axis calibration CSV data.'
    )
    parser.add_argument(
        'csv_path',
        help='Path to hand_axis_calibration CSV file',
    )
    args = parser.parse_args()
    analyze(args.csv_path)


if __name__ == '__main__':
    main()
