#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict


ROTATION_NAMES = ('up', 'down', 'left', 'right', 'clockwise', 'anticlockwise')
AXIS_LABELS = ('x', 'y', 'z')


def normalize_quaternion(q):
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in q)


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def mean_quaternion(quaternions):
    if not quaternions:
        return (0.0, 0.0, 0.0, 1.0)

    reference = normalize_quaternion(quaternions[0])
    aligned = []
    for q in quaternions:
        q = normalize_quaternion(q)
        if dot(reference, q) < 0.0:
            q = tuple(-value for value in q)
        aligned.append(q)

    mean = tuple(
        sum(q[i] for q in aligned) / float(len(aligned))
        for i in range(4)
    )
    return normalize_quaternion(mean)


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


def matmul(a, b):
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(3))
              for col in range(3))
        for row in range(3)
    )


def transpose(matrix):
    return tuple(
        tuple(matrix[col][row] for col in range(3))
        for row in range(3)
    )


def mat_vec_mul(matrix, vector):
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3))
                 for row in range(3))


def matrix_trace(matrix):
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def axis_angle_from_matrix(matrix):
    cos_angle = (matrix_trace(matrix) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.acos(cos_angle)
    if abs(angle) < 1e-8:
        return (1.0, 0.0, 0.0), 0.0

    denom = 2.0 * math.sin(angle)
    if abs(denom) < 1e-8:
        axis = (
            math.sqrt(max(0.0, (matrix[0][0] + 1.0) * 0.5)),
            math.sqrt(max(0.0, (matrix[1][1] + 1.0) * 0.5)),
            math.sqrt(max(0.0, (matrix[2][2] + 1.0) * 0.5)),
        )
        return normalize_vector(axis), angle

    axis = (
        (matrix[2][1] - matrix[1][2]) / denom,
        (matrix[0][2] - matrix[2][0]) / denom,
        (matrix[1][0] - matrix[0][1]) / denom,
    )
    return normalize_vector(axis), angle


def normalize_vector(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-9:
        return (1.0, 0.0, 0.0)
    return tuple(value / norm for value in vector)


def rpy_from_matrix(matrix):
    sy = math.sqrt(matrix[0][0] * matrix[0][0]
                   + matrix[1][0] * matrix[1][0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def vector_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def mean_position(rows, hand):
    fields = [f'{hand}_wrist_{axis}' for axis in AXIS_LABELS]
    count = float(len(rows))
    return tuple(sum(float(row[field]) for row in rows) / count
                 for field in fields)


def mean_hand_quaternion(rows, hand):
    quaternions = []
    for row in rows:
        quaternions.append(
            tuple(float(row[f'{hand}_wrist_q{axis}'])
                  for axis in ('x', 'y', 'z', 'w'))
        )
    return mean_quaternion(quaternions)


def format_vec(values):
    return '(' + ', '.join(f'{value:+.3f}' for value in values) + ')'


def dominant_axis(axis, angle):
    rotation_vector = tuple(axis[i] * angle for i in range(3))
    index = max(range(3), key=lambda i: abs(rotation_vector[i]))
    return AXIS_LABELS[index], rotation_vector[index]


def frobenius_norm_delta(a, b):
    return math.sqrt(
        sum((a[row][col] - b[row][col]) ** 2
            for row in range(3)
            for col in range(3))
    )


def transform_delta_with_basis(delta, basis):
    return matmul(matmul(basis, delta), transpose(basis))


def load_rows(csv_path):
    with open(csv_path, newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError(f'No rows found in {csv_path}')
    required = {
        'step_index',
        'step_name',
        'active_hand',
        'requested_rotation',
        'left_wrist_x',
        'left_wrist_y',
        'left_wrist_z',
        'right_wrist_x',
        'right_wrist_y',
        'right_wrist_z',
    }
    for hand in ('left', 'right'):
        for axis in ('x', 'y', 'z', 'w'):
            required.add(f'{hand}_wrist_q{axis}')
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise RuntimeError('CSV is missing column(s): ' + ', '.join(missing))
    return rows


def group_by_step(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (int(row['step_index']), row['step_name'])
        groups[key].append(row)
    return dict(sorted(groups.items()))


def build_step_stats(rows_by_step):
    stats = {}
    for key, rows in rows_by_step.items():
        stats[key] = {}
        for hand in ('left', 'right'):
            q = mean_hand_quaternion(rows, hand)
            stats[key][hand] = {
                'position': mean_position(rows, hand),
                'quaternion': q,
                'matrix': quaternion_to_matrix(q),
            }
        first = rows[0]
        stats[key]['active_hand'] = first['active_hand']
        stats[key]['requested_rotation'] = first['requested_rotation']
    return stats


def print_per_step_analysis(stats, neutral_key):
    neutral = stats[neutral_key]
    deltas = {}
    print('Per-step relative orientation from neutral')
    for key in sorted(stats):
        step_index, step_name = key
        if key == neutral_key:
            continue
        active = stats[key]['active_hand']
        if active not in ('left', 'right'):
            continue
        requested = stats[key]['requested_rotation']
        neutral_matrix = neutral[active]['matrix']
        step_matrix = stats[key][active]['matrix']
        delta = matmul(transpose(neutral_matrix), step_matrix)
        axis, angle = axis_angle_from_matrix(delta)
        roll, pitch, yaw = rpy_from_matrix(delta)
        dominant, dominant_value = dominant_axis(axis, angle)
        drift = vector_distance(
            stats[key][active]['position'],
            neutral[active]['position'],
        )
        deltas[(active, requested)] = delta
        rpy_degrees = tuple(math.degrees(v) for v in (roll, pitch, yaw))
        print(
            f'  {step_index:02d} {step_name:20s} '
            f'active={active:5s} requested={requested:13s} '
            f'drift={drift:.3f} m '
            f'axis={format_vec(axis)} angle={math.degrees(angle):+.1f} deg '
            f'rpy_deg={format_vec(rpy_degrees)} '
            f'dominant={dominant} {math.degrees(dominant_value):+.1f} deg'
        )
    return deltas


def print_left_right_comparison(deltas):
    print()
    print('Left/right requested-rotation comparison')
    for rotation in ROTATION_NAMES:
        right = deltas.get(('right', rotation))
        left = deltas.get(('left', rotation))
        if right is None or left is None:
            continue
        right_axis, right_angle = axis_angle_from_matrix(right)
        left_axis, left_angle = axis_angle_from_matrix(left)
        right_vec = tuple(right_axis[i] * right_angle for i in range(3))
        left_vec = tuple(left_axis[i] * left_angle for i in range(3))
        signs = []
        for index, label in enumerate(AXIS_LABELS):
            if abs(right_vec[index]) < math.radians(2.0):
                signs.append(f'{label}: weak')
            elif right_vec[index] * left_vec[index] < 0.0:
                signs.append(f'{label}: opposite')
            else:
                signs.append(f'{label}: same')
        print(
            f'  {rotation:13s} right_rotvec_deg='
            f'{format_vec(tuple(math.degrees(v) for v in right_vec))} '
            f'left_rotvec_deg='
            f'{format_vec(tuple(math.degrees(v) for v in left_vec))} '
            + ', '.join(signs)
        )


def print_basis_recommendation(deltas):
    basis_variants = {
        'same': (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        'flip_x': (
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        'flip_y': (
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        'flip_z': (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
        ),
        'mirror_y': (
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    }
    scores = {name: [] for name in basis_variants}
    for rotation in ROTATION_NAMES:
        right = deltas.get(('right', rotation))
        left = deltas.get(('left', rotation))
        if right is None or left is None:
            continue
        for name, basis in basis_variants.items():
            transformed_left = transform_delta_with_basis(left, basis)
            scores[name].append(frobenius_norm_delta(right, transformed_left))

    print()
    print('Candidate left orientation basis scores')
    ranked = []
    for name, values in scores.items():
        if not values:
            continue
        score = sum(values) / float(len(values))
        ranked.append((score, name))
        print(f'  {name:8s}: mean matrix error {score:.4f}')

    if not ranked:
        print('  Not enough paired left/right data for a recommendation.')
        return

    preference = {
        'same': 0,
        'flip_x': 1,
        'flip_y': 2,
        'flip_z': 3,
        'mirror_y': 4,
    }
    ranked.sort(key=lambda item: (item[0], preference.get(item[1], 99)))
    best_score, best_name = ranked[0]
    print()
    print('Simple recommendation')
    print('  Right orientation basis can probably stay as the reference.')
    print(
        f'  Best candidate for left orientation correction: '
        f'{best_name} (score {best_score:.4f})'
    )
    clockwise = deltas.get(('right', 'clockwise'))
    anticlockwise = deltas.get(('left', 'clockwise'))
    if clockwise is not None and anticlockwise is not None:
        right_axis, right_angle = axis_angle_from_matrix(clockwise)
        left_axis, left_angle = axis_angle_from_matrix(anticlockwise)
        right_vec = tuple(right_axis[i] * right_angle for i in range(3))
        left_vec = tuple(left_axis[i] * left_angle for i in range(3))
        if dot(right_vec, left_vec) < 0.0:
            print('  Clockwise direction appears inverted between hands.')
        else:
            print('  Clockwise direction does not look globally inverted.')


def analyze(csv_path):
    rows = load_rows(csv_path)
    rows_by_step = group_by_step(rows)
    stats = build_step_stats(rows_by_step)
    neutral_key = (0, 'neutral')
    if neutral_key not in stats:
        raise RuntimeError('Missing neutral step at step_index=0.')

    print(f'Loaded {len(rows)} samples from {csv_path}')
    print(f'Found {len(stats)} recorded steps')
    print()
    deltas = print_per_step_analysis(stats, neutral_key)
    print_left_right_comparison(deltas)
    print_basis_recommendation(deltas)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Quest wrist orientation calibration CSV data.'
    )
    parser.add_argument(
        'csv_path',
        help='Path to wrist_orientation_calibration CSV file',
    )
    args = parser.parse_args()
    analyze(args.csv_path)


if __name__ == '__main__':
    main()
