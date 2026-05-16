#!/usr/bin/env python3
import os
import re
import time
from pathlib import Path

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

import mujoco
import mujoco.viewer
import numpy as np


class G1Ros2Bridge(Node):
    def __init__(self):
        super().__init__('g1_ros2_bridge')

        default_model_path = os.path.expanduser(
            '~/humanoid_ws/assets/ark_unitree_g1/unitree_g1/mjcf/scene_29dof_with_hand.xml'
        )

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('initial_pose', 'boxer_calibration_pose')
        self.declare_parameter('left_ee_body_name', 'auto')
        self.declare_parameter('right_ee_body_name', 'auto')
        self.declare_parameter(
            'screenshot_dir',
            '/home/zaid/humanoid_ws/replay_action_images'
        )
        self.declare_parameter('screenshot_width', 640)
        self.declare_parameter('screenshot_height', 480)
        self.declare_parameter('screenshot_camera_name', '')
        self.declare_parameter('enable_periodic_screenshots', False)
        self.declare_parameter('screenshot_interval_sec', 1.0)
        self.model_path = self.get_parameter('model_path').value
        self.initial_pose = str(self.get_parameter('initial_pose').value).strip().lower()
        self.left_ee_body_name = str(self.get_parameter('left_ee_body_name').value)
        self.right_ee_body_name = str(self.get_parameter('right_ee_body_name').value)
        self.screenshot_dir = Path(
            str(self.get_parameter('screenshot_dir').value)
        ).expanduser()
        self.screenshot_width = int(self.get_parameter('screenshot_width').value)
        self.screenshot_height = int(self.get_parameter('screenshot_height').value)
        self.screenshot_camera_name = str(
            self.get_parameter('screenshot_camera_name').value
        ).strip()
        self.enable_periodic_screenshots = bool(
            self.get_parameter('enable_periodic_screenshots').value
        )
        self.screenshot_interval_sec = max(
            0.1,
            float(self.get_parameter('screenshot_interval_sec').value)
        )
        self.current_replay_step = 'unknown'
        self.last_screenshot_time = 0.0
        self.renderer = None

        self.get_logger().info(f'Loading MuJoCo model: {self.model_path}')

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.screenshot_width = min(
            self.screenshot_width,
            int(self.model.vis.global_.offwidth)
        )
        self.screenshot_height = min(
            self.screenshot_height,
            int(self.model.vis.global_.offheight)
        )
        self.left_ee_body_id = self.resolve_ee_body('left', self.left_ee_body_name)
        self.right_ee_body_id = self.resolve_ee_body('right', self.right_ee_body_name)

        # Collect all non-free joints
        self.joint_names = []
        self.joint_qpos_index = {}
        self.joint_qvel_index = {}

        for jid in range(self.model.njnt):
            joint_type = self.model.jnt_type[jid]

            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                continue

            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not name:
                name = f'joint_{jid}'

            self.joint_names.append(name)
            self.joint_qpos_index[name] = int(self.model.jnt_qposadr[jid])
            self.joint_qvel_index[name] = int(self.model.jnt_dofadr[jid])

        self.desired_positions = {}
        for name in self.joint_names:
            qidx = self.joint_qpos_index[name]
            self.desired_positions[name] = float(self.data.qpos[qidx])

        self.get_logger().info('Controllable joints:')
        self.get_logger().info(', '.join(self.joint_names))

        # Set robot to the requested initial pose before the viewer starts.
        if self.initial_pose == 'boxer_calibration_pose':
            self.set_initial_keyframe_pose('boxer_calibration_pose')
        elif self.initial_pose == 'standing':
            self.set_initial_standing_pose()
        else:
            self.set_initial_seated_pose()

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.left_wrist_pose_pub = self.create_publisher(
            PoseStamped,
            '/g1/left_wrist_pose',
            10
        )
        self.right_wrist_pose_pub = self.create_publisher(
            PoseStamped,
            '/g1/right_wrist_pose',
            10
        )

        # CHANGED: subscribe to Quest-mapped head command topic
        self.head_cmd_sub = self.create_subscription(
            JointState,
            '/g1/head_cmd',
            self.joint_cmd_callback,
            10
        )

        self.right_hand_cmd_sub = self.create_subscription(
            JointState,
            '/g1/right_hand_cmd',
            self.joint_cmd_callback,
            10
        )

        self.right_arm_cmd_sub = self.create_subscription(
            JointState,
            '/g1/right_arm_cmd',
            self.joint_cmd_callback,
            10
        )

        self.left_arm_cmd_sub = self.create_subscription(
            JointState,
            '/g1/left_arm_cmd',
            self.joint_cmd_callback,
            10
        )
        self.current_step_sub = self.create_subscription(
            String,
            '/replay/current_step',
            self.current_step_callback,
            10
        )
        self.screenshot_srv = self.create_service(
            Trigger,
            '~/save_screenshot',
            self.save_screenshot_callback
        )

        self.get_logger().info('Subscribed to /g1/head_cmd')
        self.get_logger().info('Subscribed to /g1/right_hand_cmd')
        self.get_logger().info('Subscribed to /g1/right_arm_cmd')
        self.get_logger().info('Subscribed to /g1/left_arm_cmd')
        self.get_logger().info('Subscribed to /replay/current_step')
        self.get_logger().info(
            f'Screenshot service ready: {self.get_fully_qualified_name()}/save_screenshot'
        )

        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False
        )

        self.timer = self.create_timer(0.02, self.update_loop)  # 50 Hz

    def resolve_ee_body(self, side, requested_name):
        requested_name = str(requested_name).strip()
        if requested_name and requested_name.lower() != 'auto':
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                requested_name
            )
            if body_id < 0:
                raise RuntimeError(
                    f'{side}_ee_body_name={requested_name!r} was not found in model.'
                )
            self.get_logger().info(
                f'Selected {side} end-effector body from parameter: {requested_name}'
            )
            return body_id

        candidates = (
            f'{side}_hand_palm_link',
            f'{side}_hand',
            f'{side}_dex3_hand',
            f'{side}_wrist_yaw_link',
            f'{side}_wrist_pitch_link',
            f'{side}_wrist_roll_link',
        )
        for candidate in candidates:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                candidate
            )
            if body_id >= 0:
                self.get_logger().info(
                    f'Selected {side} end-effector body: {candidate} (auto)'
                )
                return body_id

        available = []
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id
            )
            if name and side in name and ('wrist' in name or 'hand' in name):
                available.append(name)
        raise RuntimeError(
            f'Could not auto-select {side} end-effector body. '
            f'Available wrist/hand bodies: {available}'
        )

    def set_initial_keyframe_pose(self, keyframe_name):
        """
        Reset MuJoCo state from a named keyframe and hold its joint positions.
        """

        key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            keyframe_name
        )
        if key_id < 0:
            self.get_logger().warn(
                f'Initial keyframe not found: {keyframe_name}. Falling back to standing.'
            )
            self.set_initial_standing_pose()
            return

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.data.qvel[:] = 0.0

        for joint_name in self.joint_names:
            qidx = self.joint_qpos_index[joint_name]
            self.desired_positions[joint_name] = float(self.data.qpos[qidx])

        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(f'Initial pose set to keyframe: {keyframe_name}.')

    def set_initial_standing_pose(self):
        """
        Place the robot upright on the floor and hold a neutral standing pose.
        """

        # 1) Move floating base (root) if model has a free joint
        free_joint_found = False
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = int(self.model.jnt_qposadr[jid])

                # Base position: x, y, z
                self.data.qpos[qadr:qadr + 3] = [0.0, 0.0, 0.793]

                # Base orientation quaternion: w, x, y, z
                self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

                free_joint_found = True
                break

        if not free_joint_found:
            self.get_logger().warn('No free joint found. Cannot reposition robot base.')

        # 2) Neutral standing joint pose
        standing_pose = {
            'hip_pitch': 0.0,
            'knee': 0.0,
            'ankle_pitch': 0.0,
            'hip_roll': 0.0,
            'hip_yaw': 0.0,
            'ankle_roll': 0.0,
            'waist': 0.0,

            # Keep arms relaxed and close to neutral
            'shoulder_pitch': 0.0,
            'shoulder_roll': 0.0,
            'shoulder_yaw': 0.0,
            'elbow': 0.0,

            'wrist_roll': 0.0,
            'wrist_pitch': 0.0,
            'wrist_yaw': 0.0,
        }

        # Apply pose by partial name matching
        for key, value in standing_pose.items():
            for joint_name in self.joint_names:
                if key in joint_name:
                    qidx = self.joint_qpos_index[joint_name]
                    self.data.qpos[qidx] = value
                    self.desired_positions[joint_name] = value

        # Zero velocities for a clean start
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def set_initial_seated_pose(self):
        """
        Place the robot on the chair from scene_29dof_with_hand.xml.
        """

        free_joint_found = False
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = int(self.model.jnt_qposadr[jid])

                # Chair center is (1.0, -0.55), with the seat top at z ~= 0.48.
                # Yaw +90 deg makes the robot face the table.
                self.data.qpos[qadr:qadr + 3] = [1.0, -0.55, 0.72]
                self.data.qpos[qadr + 3:qadr + 7] = [
                    0.70710678,
                    0.0,
                    0.0,
                    0.70710678,
                ]

                free_joint_found = True
                break

        if not free_joint_found:
            self.get_logger().warn('No free joint found. Cannot reposition robot base.')

        seated_pose = {
            'left_hip_pitch_joint': -1.15,
            'right_hip_pitch_joint': -1.15,
            'left_hip_roll_joint': 0.08,
            'right_hip_roll_joint': -0.08,
            'left_hip_yaw_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'left_knee_joint': 2.05,
            'right_knee_joint': 2.05,
            'left_ankle_pitch_joint': -0.75,
            'right_ankle_pitch_joint': -0.75,
            'left_ankle_roll_joint': 0.0,
            'right_ankle_roll_joint': 0.0,
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.15,
            'left_shoulder_pitch_joint': -1.4,
            'right_shoulder_pitch_joint': -1.4,
            'left_shoulder_roll_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.25,
            'right_elbow_joint': 1.25,
            'left_wrist_roll_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }

        for joint_name, value in seated_pose.items():
            if joint_name not in self.joint_qpos_index:
                self.get_logger().warn(f'Initial seated pose joint not found: {joint_name}')
                continue
            qidx = self.joint_qpos_index[joint_name]
            self.data.qpos[qidx] = value
            self.desired_positions[joint_name] = value

        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info('Initial pose set to seated_on_chair.')

    def joint_cmd_callback(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warn(
                'Received JointState with mismatched name and position lengths.'
            )
            return

        for name, pos in zip(msg.name, msg.position):
            if name not in self.desired_positions:
                self.get_logger().warn(f'Unknown joint name: {name}')
                continue
            self.desired_positions[name] = float(pos)

    def current_step_callback(self, msg: String):
        value = msg.data.strip()
        if value:
            self.current_replay_step = value

    def update_loop(self):
        if self.viewer is None or not self.viewer.is_running():
            self.get_logger().info('Viewer closed. Shutting down node.')
            rclpy.shutdown()
            return

        with self.viewer.lock():
            for name, pos in self.desired_positions.items():
                qidx = self.joint_qpos_index[name]
                self.data.qpos[qidx] = pos

            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync(state_only=True)

        stamp = self.get_clock().now().to_msg()
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(self.joint_names)
        msg.position = [float(self.data.qpos[self.joint_qpos_index[n]]) for n in self.joint_names]
        msg.velocity = [float(self.data.qvel[self.joint_qvel_index[n]]) for n in self.joint_names]
        msg.effort = []
        self.joint_state_pub.publish(msg)
        self.left_wrist_pose_pub.publish(
            self.make_body_pose_stamped(stamp, self.left_ee_body_id)
        )
        self.right_wrist_pose_pub.publish(
            self.make_body_pose_stamped(stamp, self.right_ee_body_id)
        )

        if self.enable_periodic_screenshots:
            now = time.monotonic()
            if now - self.last_screenshot_time >= self.screenshot_interval_sec:
                self.last_screenshot_time = now
                try:
                    self.save_screenshot()
                except Exception as exc:
                    self.get_logger().warn(f'Periodic screenshot failed: {exc}')

    def make_body_pose_stamped(self, stamp, body_id):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(self.data.xpos[body_id][0])
        msg.pose.position.y = float(self.data.xpos[body_id][1])
        msg.pose.position.z = float(self.data.xpos[body_id][2])
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, self.data.xmat[body_id])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        return msg

    def save_screenshot_callback(self, _request, response):
        try:
            image_path = self.save_screenshot()
        except Exception as exc:
            response.success = False
            response.message = f'Failed to save screenshot: {exc}'
            self.get_logger().error(response.message)
            return response

        response.success = True
        response.message = str(image_path)
        self.get_logger().info(f'Saved MuJoCo screenshot: {image_path}')
        return response

    def save_screenshot(self):
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        filename = self.safe_step_filename(self.current_replay_step)
        image_path = self.screenshot_dir / f'{filename}.png'

        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.screenshot_height,
                width=self.screenshot_width
            )

        camera = self.screenshot_camera_name or None
        try:
            self.renderer.update_scene(self.data, camera=camera)
        except TypeError:
            self.renderer.update_scene(self.data)
        image = self.renderer.render()
        self.write_png(image_path, image)
        return image_path

    @staticmethod
    def safe_step_filename(step_name):
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', step_name.strip())
        return name.strip('._') or 'unknown'

    @staticmethod
    def write_png(path, image):
        try:
            from PIL import Image
            Image.fromarray(image).save(path)
            return
        except Exception:
            pass

        try:
            import imageio.v2 as imageio
            imageio.imwrite(path, image)
            return
        except Exception as exc:
            raise RuntimeError(
                'PNG writing requires either pillow or imageio to be installed.'
            ) from exc

    def destroy_node(self):
        try:
            if self.viewer is not None:
                self.viewer.close()
        except Exception:
            pass
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = G1Ros2Bridge()
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
