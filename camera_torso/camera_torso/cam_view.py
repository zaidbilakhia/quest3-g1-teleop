#!/usr/bin/env python3

import os
from typing import Dict

import cv2
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


class FaceCameraViewer(Node):
    def __init__(self):
        super().__init__('cam_view')

        default_xml = os.path.expanduser(
            '~/humanoid_ws/assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml'
        )

        self.declare_parameter('xml_path', default_xml)
        self.declare_parameter('camera_name', 'face_camera')
        self.declare_parameter('image_topic', '/face_camera/image_raw')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 20.0)

        self.xml_path = self.get_parameter('xml_path').get_parameter_value().string_value
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.joint_states_topic = self.get_parameter('joint_states_topic').get_parameter_value().string_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value

        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f'XML file not found: {self.xml_path}')

        self.get_logger().info(f'Loading MuJoCo scene from: {self.xml_path}')
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id == -1:
            raise RuntimeError(f'Camera "{self.camera_name}" not found in model XML.')
        self.cam_id = cam_id
        self.get_logger().info(f'Using camera "{self.camera_name}" with id {self.cam_id}')

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.joint_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_states_callback,
            10
        )

        self.joint_qpos_index: Dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None:
                continue

            joint_type = self.model.jnt_type[joint_id]
            qpos_adr = self.model.jnt_qposadr[joint_id]

            if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self.joint_qpos_index[joint_name] = qpos_adr

        self.get_logger().info(
            f'Mapped {len(self.joint_qpos_index)} MuJoCo joints for /joint_states syncing.'
        )

        self.latest_joint_positions: Dict[str, float] = {}

        period = 1.0 / self.fps
        self.timer = self.create_timer(period, self.render_loop)

        cv2.namedWindow('Face Camera', cv2.WINDOW_NORMAL)
        self.get_logger().info('Camera node started.')
        self.get_logger().info('OpenCV window: Face Camera')
        self.get_logger().info(f'Publishing images on: {self.image_topic}')

    def joint_states_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joint_positions[name] = pos

    def apply_joint_states_to_model(self):
        for joint_name, joint_pos in self.latest_joint_positions.items():
            if joint_name in self.joint_qpos_index:
                qpos_idx = self.joint_qpos_index[joint_name]
                self.data.qpos[qpos_idx] = joint_pos

        mujoco.mj_forward(self.model, self.data)

    def publish_image_msg(self, rgb_image: np.ndarray):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_name
        msg.height = rgb_image.shape[0]
        msg.width = rgb_image.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = rgb_image.shape[1] * 3
        msg.data = rgb_image.tobytes()
        self.image_pub.publish(msg)

    def render_loop(self):
        self.apply_joint_states_to_model()

        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('Face Camera', bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            self.get_logger().info('Closing camera node...')
            rclpy.shutdown()
            return

        self.publish_image_msg(rgb)

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FaceCameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()