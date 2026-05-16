#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class HeadPoseViz(Node):
    def __init__(self):
        super().__init__('head_pose_viz')

        self.sub = self.create_subscription(
            PoseStamped,
            '/quest/pose/headset',
            self.cb,
            10
        )

        self.pub = self.create_publisher(
            PoseStamped,
            '/quest/head_pose_viz',
            10
        )

        self.get_logger().info('Head pose viz node started.')

    def cb(self, msg: PoseStamped):
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'vr_origin'

        # keep object fixed in space, only orientation changes
        out.pose.position.x = 0.0
        out.pose.position.y = 0.0
        out.pose.position.z = 0.0

        out.pose.orientation = msg.pose.orientation

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HeadPoseViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()