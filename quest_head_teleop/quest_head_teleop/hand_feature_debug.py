#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from vr_haptic_msgs.msg import ManoLandmarks


def dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def finger_curl(msg, base_idx, j1_idx, j2_idx, tip_idx):
    """
    Simple curl estimate:
    - compute finger chain length
    - compute straight-line distance from base to tip
    - if finger is straight, straight distance ~= chain length
    - if finger is curled, straight distance becomes smaller

    Returns:
        0.0 -> open/straight
        1.0 -> curled more
    """
    lm = msg.landmarks

    base = lm[base_idx]
    j1 = lm[j1_idx]
    j2 = lm[j2_idx]
    tip = lm[tip_idx]

    chain_len = dist(base, j1) + dist(j1, j2) + dist(j2, tip)
    straight_len = dist(base, tip)

    if chain_len < 1e-6:
        return 0.0

    openness = straight_len / chain_len
    openness = clamp(openness, 0.0, 1.0)

    curl = 1.0 - openness
    return clamp(curl, 0.0, 1.0)


class HandFeatureDebug(Node):
    def __init__(self):
        super().__init__('hand_feature_debug')

        self.sub = self.create_subscription(
            ManoLandmarks,
            '/quest/hand_pose',
            self.callback,
            10
        )

        self.count = 0
        self.get_logger().info('Hand feature debug node started.')
        self.get_logger().info('Subscribing to /quest/hand_pose')

    def callback(self, msg):
        if len(msg.landmarks) < 21:
            self.get_logger().warn(f'Expected 21 landmarks, got {len(msg.landmarks)}')
            return

        thumb_close = finger_curl(msg, 1, 2, 3, 4)
        index_close = finger_curl(msg, 5, 6, 7, 8)
        middle_close = finger_curl(msg, 9, 10, 11, 12)

        self.count += 1

        # Print every 10 messages so terminal stays readable
        if self.count % 10 == 0:
            self.get_logger().info(
                f'thumb_close={thumb_close:.3f}, '
                f'index_close={index_close:.3f}, '
                f'middle_close={middle_close:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = HandFeatureDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
