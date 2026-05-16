#!/usr/bin/env python3

import sys
import select
import termios
import tty
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class HeadKeyboardTeleop(Node):
    def __init__(self):
        super().__init__('head_keyboard_teleop')

        # ===== CHANGE THESE =====
        self.command_topic = '/g1/joint_cmd'
        self.head_yaw_joint = 'waist_yaw_joint'
        self.head_pitch_joint = 'waist_pitch_joint'
        # ========================

        self.publisher_ = self.create_publisher(JointState, self.command_topic, 10)

        self.current_yaw = 0.0
        self.current_pitch = 0.0

        self.yaw_dir = 0
        self.pitch_dir = 0

        self.yaw_speed = 0.8
        self.pitch_speed = 0.6

        self.yaw_min = -0.5
        self.yaw_max = 0.5
        self.pitch_min = -0.3
        self.pitch_max = 0.3

        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        self.running = True
        self.lock = threading.Lock()

        self.key_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.key_thread.start()

        self.get_logger().info('Head keyboard teleop started.')
        self.get_logger().info('Controls:')
        self.get_logger().info('  LEFT  arrow -> turn head left')
        self.get_logger().info('  RIGHT arrow -> turn head right')
        self.get_logger().info('  UP    arrow -> look up')
        self.get_logger().info('  DOWN  arrow -> look down')
        self.get_logger().info('  SPACE       -> stop head motion')
        self.get_logger().info('  E           -> exit')

    def clamp(self, value, vmin, vmax):
        return max(vmin, min(value, vmax))

    def keyboard_loop(self):
        while self.running:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue

            ch1 = sys.stdin.read(1)

            if ch1 == '\x1b':
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                seq = ch1 + ch2 + ch3

                with self.lock:
                    if seq == '\x1b[D':   # LEFT
                        self.yaw_dir = 1
                        self.get_logger().info('Turning head LEFT')
                    elif seq == '\x1b[C': # RIGHT
                        self.yaw_dir = -1
                        self.get_logger().info('Turning head RIGHT')
                    elif seq == '\x1b[A': # UP
                        self.pitch_dir = 1
                        self.get_logger().info('Looking UP')
                    elif seq == '\x1b[B': # DOWN
                        self.pitch_dir = -1
                        self.get_logger().info('Looking DOWN')

            elif ch1 == ' ':
                with self.lock:
                    self.yaw_dir = 0
                    self.pitch_dir = 0
                self.get_logger().info('Stopping head motion')

            elif ch1 in ['e', 'E']:
                self.get_logger().info('Exiting head teleop...')
                self.running = False
                break

    def publish_head_command(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.head_yaw_joint, self.head_pitch_joint]
        msg.position = [self.current_yaw, self.current_pitch]
        self.publisher_.publish(msg)

    def run_loop(self):
        rate_hz = 20.0
        dt = 1.0 / rate_hz

        while rclpy.ok() and self.running:
            with self.lock:
                self.current_yaw += self.yaw_dir * self.yaw_speed * dt
                self.current_pitch += self.pitch_dir * self.pitch_speed * dt

                self.current_yaw = self.clamp(self.current_yaw, self.yaw_min, self.yaw_max)
                self.current_pitch = self.clamp(self.current_pitch, self.pitch_min, self.pitch_max)

            self.publish_head_command()
            time.sleep(dt)

    def cleanup(self):
        self.running = False
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = HeadKeyboardTeleop()

    try:
        node.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()