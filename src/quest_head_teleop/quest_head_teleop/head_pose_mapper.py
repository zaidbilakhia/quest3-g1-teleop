#!/usr/bin/env python3
import math
import json
from pathlib import Path

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from sensor_msgs.msg import JointState


def quat_normalize(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return None
    return (x / n, y / n, z / n, w / n)


def quat_inverse(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return (x, y, z, w)


def quat_to_rpy(q):
    x, y, z, w = q

    # roll
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    # yaw
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def apply_deadband(v, db):
    if abs(v) < db:
        return 0.0
    return v


def piecewise_normalize(value, pos_limit, neg_limit):
    """
    Returns value normalized into [-1, 1] using asymmetric positive/negative limits.
    pos_limit should be > 0
    neg_limit should be < 0
    """
    eps = 1e-6

    if value >= 0.0:
        if abs(pos_limit) < eps:
            return 0.0
        return clamp(value / pos_limit, 0.0, 1.0)
    else:
        if abs(neg_limit) < eps:
            return 0.0
        return clamp(value / abs(neg_limit), -1.0, 0.0)


class QuestHeadRuntime(Node):
    def __init__(self):
        super().__init__('quest_head_runtime')

        self.input_topic = '/quest/pose/headset'
        self.debug_rpy_topic = '/quest/head_rpy_relative'
        self.debug_norm_topic = '/quest/head_cmd_normalized'

        # Keep topic name as head_cmd, but internally drive waist joints
        self.output_topic = '/g1/head_cmd'

        # Current G1 29DOF model has waist joints, not separate head joints
        self.joint_names = ['waist_yaw_joint', 'waist_pitch_joint']

        self.calibration_path = str(Path.home() / 'humanoid_ws' / 'quest_head_calibration.json')

        # Safer torso-look limits
        self.robot_yaw_limit = 0.40
        self.robot_pitch_up_limit = 0.25
        self.robot_pitch_down_limit = 0.20

        # Sign tuning
        self.yaw_sign = 1.0
        self.pitch_sign = 1.0

        # Noise handling
        self.yaw_deadband = 0.03
        self.pitch_deadband = 0.03

        # Low-pass smoothing
        self.alpha = 0.2
        self.filtered_yaw_cmd = 0.0
        self.filtered_pitch_cmd = 0.0
        self.have_filter_state = False

        self.msg_count = 0
        self.q_ref = None

        self.load_calibration()

        self.sub = self.create_subscription(
            PoseStamped,
            self.input_topic,
            self.pose_callback,
            50
        )

        self.debug_rpy_pub = self.create_publisher(Vector3Stamped, self.debug_rpy_topic, 10)
        self.debug_norm_pub = self.create_publisher(Vector3Stamped, self.debug_norm_topic, 10)
        self.joint_pub = self.create_publisher(JointState, self.output_topic, 10)

        self.get_logger().info(f'Loaded calibration from: {self.calibration_path}')
        self.get_logger().info(f'Subscribing to: {self.input_topic}')
        self.get_logger().info(f'Publishing torso-look cmd to: {self.output_topic}')
        self.get_logger().info(f'Joint names: {self.joint_names}')

    def load_calibration(self):
        with open(self.calibration_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        q_ref = data['neutral_quaternion_xyzw']
        self.q_ref = quat_normalize(tuple(q_ref))
        if self.q_ref is None:
            raise RuntimeError('Invalid neutral quaternion in calibration file.')

        self.yaw_left_limit = float(data['derived']['yaw_left_limit_rad'])       # positive
        self.yaw_right_limit = float(data['derived']['yaw_right_limit_rad'])     # negative
        self.pitch_up_limit = float(data['derived']['pitch_up_limit_rad'])       # usually negative
        self.pitch_down_limit = float(data['derived']['pitch_down_limit_rad'])   # usually positive

        self.get_logger().info(
            f'Calibration limits loaded: '
            f'yaw_left={self.yaw_left_limit:.3f}, '
            f'yaw_right={self.yaw_right_limit:.3f}, '
            f'pitch_up={self.pitch_up_limit:.3f}, '
            f'pitch_down={self.pitch_down_limit:.3f}'
        )

    def low_pass(self, prev, new):
        return (1.0 - self.alpha) * prev + self.alpha * new

    def pose_callback(self, msg: PoseStamped):
        q_curr = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        q_curr = quat_normalize(q_curr)
        if q_curr is None:
            return

        q_rel = quat_multiply(quat_inverse(self.q_ref), q_curr)
        q_rel = quat_normalize(q_rel)
        if q_rel is None:
            return

        roll, pitch, yaw = quat_to_rpy(q_rel)

        # Deadband on raw relative angles
        yaw = apply_deadband(yaw, self.yaw_deadband)
        pitch = apply_deadband(pitch, self.pitch_deadband)

        # Normalize using actual calibrated human range
        yaw_norm = piecewise_normalize(
            yaw,
            pos_limit=self.yaw_left_limit,
            neg_limit=self.yaw_right_limit
        )

        # In your calibration:
        # up   = negative pitch
        # down = positive pitch
        pitch_norm = piecewise_normalize(
            pitch,
            pos_limit=self.pitch_down_limit,
            neg_limit=self.pitch_up_limit
        )

        # Map normalized motion to waist commands
        yaw_cmd = self.yaw_sign * yaw_norm * self.robot_yaw_limit

        if pitch_norm >= 0.0:
            pitch_cmd = self.pitch_sign * pitch_norm * self.robot_pitch_down_limit
        else:
            pitch_cmd = self.pitch_sign * abs(pitch_norm) * (-self.robot_pitch_up_limit)

        # Smoothing
        if not self.have_filter_state:
            self.filtered_yaw_cmd = yaw_cmd
            self.filtered_pitch_cmd = pitch_cmd
            self.have_filter_state = True
        else:
            self.filtered_yaw_cmd = self.low_pass(self.filtered_yaw_cmd, yaw_cmd)
            self.filtered_pitch_cmd = self.low_pass(self.filtered_pitch_cmd, pitch_cmd)

        # Final clamp
        self.filtered_yaw_cmd = clamp(self.filtered_yaw_cmd, -self.robot_yaw_limit, self.robot_yaw_limit)
        self.filtered_pitch_cmd = clamp(
            self.filtered_pitch_cmd,
            -self.robot_pitch_up_limit,
            self.robot_pitch_down_limit
        )

        # Debug RPY publisher
        dbg_rpy = Vector3Stamped()
        dbg_rpy.header.stamp = self.get_clock().now().to_msg()
        dbg_rpy.header.frame_id = 'quest_head_relative_rpy'
        dbg_rpy.vector.x = roll
        dbg_rpy.vector.y = pitch
        dbg_rpy.vector.z = yaw
        self.debug_rpy_pub.publish(dbg_rpy)

        # Debug normalized publisher
        dbg_norm = Vector3Stamped()
        dbg_norm.header.stamp = self.get_clock().now().to_msg()
        dbg_norm.header.frame_id = 'quest_head_normalized_cmd'
        dbg_norm.vector.x = 0.0
        dbg_norm.vector.y = pitch_norm
        dbg_norm.vector.z = yaw_norm
        self.debug_norm_pub.publish(dbg_norm)

        # Joint command publisher
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = [self.filtered_yaw_cmd, self.filtered_pitch_cmd]
        self.joint_pub.publish(js)

        self.msg_count += 1
        if self.msg_count % 100 == 0:
            self.get_logger().info(
                f'raw yaw={yaw:.3f}, raw pitch={pitch:.3f} | '
                f'norm yaw={yaw_norm:.3f}, norm pitch={pitch_norm:.3f} | '
                f'cmd waist_yaw={self.filtered_yaw_cmd:.3f}, '
                f'cmd waist_pitch={self.filtered_pitch_cmd:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = QuestHeadRuntime()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()