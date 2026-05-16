#!/usr/bin/env python3

import ctypes
import os
import sys
import time


# The Unitree xr_teleoperate IK imports `from pinocchio import casadi as cpin`.
# In this workspace that binding is available from the local conda-forge env.
# Override with XR_TELEOP_IK_PREFIX if the env moves.
DEFAULT_IK_PYTHON_PREFIX = "/home/zaid/micromamba/envs/xr-teleop-ik"


def prefer_ik_python_prefix(prefix):
    if not prefix or not os.path.isdir(prefix):
        return

    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(prefix, "lib", python_tag, "site-packages")
    lib_dir = os.path.join(prefix, "lib")
    bin_dir = os.path.join(prefix, "bin")

    if os.path.isdir(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    if os.path.isdir(lib_dir):
        os.environ["LD_LIBRARY_PATH"] = (
            lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )
        for library_name in ("libstdc++.so.6", "libgcc_s.so.1"):
            library_path = os.path.join(lib_dir, library_name)
            if os.path.exists(library_path):
                try:
                    ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    if os.path.isdir(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


prefer_ik_python_prefix(os.environ.get("XR_TELEOP_IK_PREFIX", DEFAULT_IK_PYTHON_PREFIX))

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

try:
    from vr_haptic_msgs.msg import ManoLandmarks
except Exception:
    ManoLandmarks = None

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]


HAND_TO_ROBOT_BASIS = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


class LiveArmMapper(Node):
    """ROS 2 communication adapter for xr_teleoperate G1_29_ArmIK.

    This node keeps the Unitree arm method intact:
    G1_29_ArmIK, Pinocchio + CasADi + IPOPT, 4x4 wrist targets, dual-arm
    solve_ik(), and the internal weighted moving filter.

    Only the external communication is replaced:
    TeleVuer input becomes a ROS ManoLandmarks subscription, and Unitree DDS
    output becomes a ROS JointState publisher for the MuJoCo bridge.
    """

    def __init__(
        self,
        node_name="live_arm_mapper",
        default_arm_side="right",
        default_hand_topic=None,
    ):
        super().__init__(node_name)
        self.default_arm_side = default_arm_side
        self.default_hand_topic = default_hand_topic

        self.declare_and_read_params()
        self.setup_xr_repo_import()
        self.arm_ik = self.create_arm_ik()

        self.left_wrist_neutral, self.right_wrist_neutral = self.make_neutral_targets()

        self.latest_hand_pose = None
        self.neutral_hand_pose = None
        self.filtered_target_pose = None
        self.filtered_arm_q = None
        self.last_command_time = None
        self.current_lr_arm_q = np.zeros(14, dtype=float)
        self.current_lr_arm_dq = np.zeros(14, dtype=float)
        self.have_joint_state = False
        self.last_log_time = {}

        self.hand_pose_sub = None
        if ManoLandmarks is None:
            self.get_logger().warning(
                "vr_haptic_msgs/ManoLandmarks is not importable; "
                f"cannot subscribe to {self.quest_hand_pose_topic}."
            )
        else:
            self.hand_pose_sub = self.create_subscription(
                ManoLandmarks,
                self.quest_hand_pose_topic,
                self.hand_pose_callback,
                10,
            )
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            20,
        )
        self.cmd_pub = self.create_publisher(JointState, self.cmd_topic, 10)
        self.calibrate_srv = self.create_service(
            Trigger,
            "~/calibrate_neutral",
            self.calibrate_neutral_callback,
        )

        timer_period = 1.0 / max(self.publish_rate, 1e-6)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            "live_arm_mapper ready: "
            f"arm_side={self.arm_side}, "
            f"quest_hand_pose_topic={self.quest_hand_pose_topic}, "
            f"joint_state_topic={self.joint_state_topic}, "
            f"cmd_topic={self.cmd_topic}, "
            f"publish_rate={self.publish_rate}, "
            "calibration_service=~/calibrate_neutral"
        )

    def declare_and_read_params(self):
        self.declare_parameter("arm_side", self.default_arm_side)
        self.arm_side = str(self.get_parameter("arm_side").value).strip().lower()
        if self.arm_side not in ("left", "right"):
            self.get_logger().warning(
                f"Unsupported arm_side={self.arm_side!r}; using 'right'."
            )
            self.arm_side = "right"

        default_cmd_topic = f"/g1/{self.arm_side}_arm_cmd"
        if self.arm_side == "left":
            default_workspace_y = (0.05, 0.45)
        else:
            default_workspace_y = (-0.45, -0.05)

        default_hand_topic = self.default_hand_topic
        if default_hand_topic is None:
            default_hand_topic = (
                "/quest/left_hand_pose"
                if self.arm_side == "left"
                else "/quest/hand_pose"
            )

        self.declare_parameter("quest_hand_pose_topic", default_hand_topic)
        self.declare_parameter("hand_wrist_landmark_index", 0)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cmd_topic", default_cmd_topic)
        self.declare_parameter(
            "xr_repo_path",
            "/home/zaid/humanoid_ws/assets/xr_teleoperate",
        )
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("use_orientation", True)
        self.declare_parameter("debug", True)
        self.declare_parameter("right_neutral_x", 0.32)
        self.declare_parameter("right_neutral_y", -0.20)
        self.declare_parameter("right_neutral_z", 0.20)
        self.declare_parameter("left_neutral_x", 0.32)
        self.declare_parameter("left_neutral_y", 0.20)
        self.declare_parameter("left_neutral_z", 0.20)
        self.declare_parameter("workspace_min_x", 0.12)
        self.declare_parameter("workspace_max_x", 0.55)
        self.declare_parameter("workspace_min_y", default_workspace_y[0])
        self.declare_parameter("workspace_max_y", default_workspace_y[1])
        self.declare_parameter("workspace_min_z", -0.05)
        self.declare_parameter("workspace_max_z", 0.50)
        self.declare_parameter("target_filter_alpha", 0.30)
        self.declare_parameter("joint_filter_alpha", 0.35)
        self.declare_parameter("max_joint_velocity", 1.2)

        self.quest_hand_pose_topic = self.get_parameter("quest_hand_pose_topic").value
        self.hand_wrist_landmark_index = int(
            self.get_parameter("hand_wrist_landmark_index").value
        )
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.xr_repo_path = os.path.abspath(self.get_parameter("xr_repo_path").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.position_scale = float(self.get_parameter("position_scale").value)
        self.use_orientation = bool(self.get_parameter("use_orientation").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.target_filter_alpha = self.clamp_float(
            float(self.get_parameter("target_filter_alpha").value), 0.0, 1.0
        )
        self.joint_filter_alpha = self.clamp_float(
            float(self.get_parameter("joint_filter_alpha").value), 0.0, 1.0
        )
        self.max_joint_velocity = max(
            0.0,
            float(self.get_parameter("max_joint_velocity").value),
        )

        self.right_neutral_position = np.array(
            [
                float(self.get_parameter("right_neutral_x").value),
                float(self.get_parameter("right_neutral_y").value),
                float(self.get_parameter("right_neutral_z").value),
            ],
            dtype=float,
        )
        self.left_neutral_position = np.array(
            [
                float(self.get_parameter("left_neutral_x").value),
                float(self.get_parameter("left_neutral_y").value),
                float(self.get_parameter("left_neutral_z").value),
            ],
            dtype=float,
        )
        self.workspace_min = np.array(
            [
                float(self.get_parameter("workspace_min_x").value),
                float(self.get_parameter("workspace_min_y").value),
                float(self.get_parameter("workspace_min_z").value),
            ],
            dtype=float,
        )
        self.workspace_max = np.array(
            [
                float(self.get_parameter("workspace_max_x").value),
                float(self.get_parameter("workspace_max_y").value),
                float(self.get_parameter("workspace_max_z").value),
            ],
            dtype=float,
        )
        self.workspace_min, self.workspace_max = (
            np.minimum(self.workspace_min, self.workspace_max),
            np.maximum(self.workspace_min, self.workspace_max),
        )

        if self.arm_side == "left":
            self.active_joint_names = LEFT_ARM_JOINT_NAMES
            self.active_slice = slice(0, 7)
            self.active_neutral_position = self.left_neutral_position
        else:
            self.active_joint_names = RIGHT_ARM_JOINT_NAMES
            self.active_slice = slice(7, 14)
            self.active_neutral_position = self.right_neutral_position

    def setup_xr_repo_import(self):
        self.xr_teleop_path = os.path.join(self.xr_repo_path, "teleop")

        # Required for: from robot_control.robot_arm_ik import G1_29_ArmIK
        if self.xr_teleop_path not in sys.path:
            sys.path.append(self.xr_teleop_path)

        # Required because robot_arm_ik.py imports teleop.utils.weighted_moving_filter.
        if self.xr_repo_path not in sys.path:
            sys.path.append(self.xr_repo_path)

    def create_arm_ik(self):
        try:
            from robot_control.robot_arm_ik import G1_29_ArmIK
        except Exception as exc:
            raise RuntimeError(
                "Failed to import G1_29_ArmIK from xr_teleoperate. "
                f"Added sys.path: {self.xr_teleop_path}, {self.xr_repo_path}. "
                f"Original error: {exc}"
            ) from exc

        # Match xr_teleoperate's working directory convention, because
        # G1_29_ArmIK uses relative paths for the G1 URDF and model cache.
        old_cwd = os.getcwd()
        try:
            os.chdir(self.xr_teleop_path)
            return G1_29_ArmIK()
        finally:
            os.chdir(old_cwd)

    @staticmethod
    def clamp_float(value, lower, upper):
        return max(lower, min(upper, value))

    @staticmethod
    def normalize_vector(vector, min_norm=1e-6):
        norm = np.linalg.norm(vector)
        if norm < min_norm:
            return None
        return vector / norm

    def hand_landmarks_msg_to_matrix(self, msg):
        if self.hand_wrist_landmark_index < 0:
            raise ValueError("hand_wrist_landmark_index must be >= 0")
        if len(msg.landmarks) <= self.hand_wrist_landmark_index:
            raise ValueError(
                "not enough landmarks: "
                f"got {len(msg.landmarks)}, need index {self.hand_wrist_landmark_index}"
            )

        wrist = msg.landmarks[self.hand_wrist_landmark_index]
        T = np.eye(4, dtype=float)
        T[:3, 3] = [wrist.x, wrist.y, wrist.z]
        T[:3, :3] = self.hand_landmarks_to_orientation(msg)

        return T

    def hand_landmarks_to_orientation(self, msg):
        required_indices = [0, 5, 9, 17]
        if len(msg.landmarks) <= max(required_indices):
            raise ValueError(
                "not enough landmarks for wrist orientation: "
                f"got {len(msg.landmarks)}, need index {max(required_indices)}"
            )

        points = []
        for index in required_indices:
            point = msg.landmarks[index]
            points.append(np.array([point.x, point.y, point.z], dtype=float))

        wrist, index_mcp, middle_mcp, pinky_mcp = points
        palm_x = self.normalize_vector(index_mcp - pinky_mcp)
        finger_axis = self.normalize_vector(middle_mcp - wrist)
        if palm_x is None or finger_axis is None:
            return np.eye(3, dtype=float)

        palm_z = self.normalize_vector(np.cross(palm_x, finger_axis))
        if palm_z is None:
            return np.eye(3, dtype=float)

        palm_y = self.normalize_vector(np.cross(palm_z, palm_x))
        if palm_y is None:
            return np.eye(3, dtype=float)

        return np.column_stack((palm_x, palm_y, palm_z))

    def matrix_to_pose_debug(self, T):
        p = T[:3, 3]
        return f"x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}"

    def make_transform(self, position, rotation=None):
        T = np.eye(4, dtype=float)
        T[:3, 3] = np.asarray(position, dtype=float)
        if rotation is not None:
            T[:3, :3] = np.asarray(rotation, dtype=float)
        return T

    def make_neutral_targets(self):
        left = self.make_transform(self.left_neutral_position)
        right = self.make_transform(self.right_neutral_position)
        return left, right

    def hand_pose_callback(self, msg):
        try:
            self.latest_hand_pose = self.hand_landmarks_msg_to_matrix(msg)
        except Exception as exc:
            self.throttled_log(
                "hand_pose_parse_error",
                2.0,
                "error",
                f"Failed to convert ManoLandmarks to wrist matrix: {exc}",
            )

    def calibrate_neutral_callback(self, _request, response):
        if self.latest_hand_pose is None:
            response.success = False
            response.message = (
                f"No hand landmark input received yet on {self.quest_hand_pose_topic}."
            )
            return response

        self.neutral_hand_pose = self.latest_hand_pose.copy()
        self.filtered_target_pose = None
        self.filtered_arm_q = None
        self.last_command_time = None
        response.success = True
        response.message = (
            "Captured current wrist landmark pose as neutral reference: "
            + self.matrix_to_pose_debug(self.neutral_hand_pose)
        )
        self.get_logger().info(response.message)
        return response

    def joint_state_callback(self, msg):
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        lr_arm_q = np.zeros(14, dtype=float)
        lr_arm_dq = np.zeros(14, dtype=float)
        missing = []

        for offset, joint_names in (
            (0, LEFT_ARM_JOINT_NAMES),
            (7, RIGHT_ARM_JOINT_NAMES),
        ):
            for joint_index, joint_name in enumerate(joint_names):
                msg_index = name_to_index.get(joint_name)
                if msg_index is None:
                    if joint_name in self.active_joint_names:
                        missing.append(joint_name)
                    continue

                out_index = offset + joint_index
                if msg_index < len(msg.position):
                    lr_arm_q[out_index] = msg.position[msg_index]
                if msg_index < len(msg.velocity):
                    lr_arm_dq[out_index] = msg.velocity[msg_index]

        self.current_lr_arm_q = lr_arm_q
        self.current_lr_arm_dq = lr_arm_dq
        self.have_joint_state = len(missing) == 0

        if missing:
            self.throttled_log(
                "missing_joint_states",
                5.0,
                "warning",
                f"Missing {self.arm_side} arm joints in JointState: "
                + ", ".join(missing),
            )

    def timer_callback(self):
        if self.latest_hand_pose is None:
            self.throttled_log(
                "waiting_for_hand_pose",
                5.0,
                "info",
                f"Waiting for wrist landmark input on {self.quest_hand_pose_topic}",
            )
            return

        if self.neutral_hand_pose is None:
            self.throttled_log(
                "waiting_for_calibration",
                5.0,
                "info",
                "Waiting for intentional neutral calibration. Run: "
                f"ros2 service call /{self.get_name()}/calibrate_neutral "
                "std_srvs/srv/Trigger {}",
            )
            return

        if not self.have_joint_state:
            self.throttled_log(
                "using_zero_joint_state",
                5.0,
                "warning",
                "No complete right arm joint state yet; using zeros.",
            )

        active_wrist_pose = self.convert_hand_pose_to_robot_target(
            self.latest_hand_pose
        )
        if self.arm_side == "left":
            left_wrist_pose = active_wrist_pose
            right_wrist_pose = self.right_wrist_neutral
        else:
            left_wrist_pose = self.left_wrist_neutral
            right_wrist_pose = active_wrist_pose

        current_lr_arm_q = self.current_lr_arm_q.copy()
        current_lr_arm_dq = self.current_lr_arm_dq.copy()

        try:
            sol_q, _sol_tauff = self.arm_ik.solve_ik(
                left_wrist_pose,
                right_wrist_pose,
                current_lr_arm_q,
                current_lr_arm_dq,
            )
        except Exception as exc:
            self.throttled_log("ik_solve_error", 2.0, "error", f"IK solve failed: {exc}")
            return

        sol_q = np.asarray(sol_q, dtype=float)
        if sol_q.shape[0] < 14:
            self.throttled_log(
                "ik_bad_shape",
                2.0,
                "error",
                f"IK returned unexpected q shape {sol_q.shape}; not publishing.",
            )
            return

        active_q = sol_q[self.active_slice].copy()
        if not np.all(np.isfinite(active_q)):
            self.throttled_log(
                "ik_nonfinite",
                2.0,
                "error",
                f"IK returned NaN/Inf {self.arm_side}_q={active_q}; not publishing.",
            )
            return

        active_q = self.clip_active_q_to_ik_limits(active_q)
        active_q = self.filter_and_limit_active_q(active_q)
        self.publish_active_arm(active_q)

        if self.debug:
            self.throttled_log(
                "debug_publish",
                1.0,
                "info",
                "target "
                + self.matrix_to_pose_debug(active_wrist_pose)
                + f", {self.arm_side}_q={np.array2string(active_q, precision=3)}",
            )

    def publish_active_arm(self, active_q):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.active_joint_names)
        msg.position = [float(q) for q in active_q]
        msg.velocity = []
        msg.effort = []
        self.cmd_pub.publish(msg)

    def convert_hand_pose_to_robot_target(self, T_hand):
        if self.arm_side == "left":
            T_robot_target = self.left_wrist_neutral.copy()
        else:
            T_robot_target = self.right_wrist_neutral.copy()

        delta_pos_hand = T_hand[:3, 3] - self.neutral_hand_pose[:3, 3]
        delta_robot = np.array(
            [
                delta_pos_hand[1],
                -delta_pos_hand[0],
                delta_pos_hand[2],
            ],
            dtype=float,
        )
        unclipped_position = (
            self.active_neutral_position + delta_robot * self.position_scale
        )
        clipped_position = np.clip(
            unclipped_position,
            self.workspace_min,
            self.workspace_max,
        )
        if not np.allclose(unclipped_position, clipped_position):
            self.throttled_log(
                "workspace_clipped",
                1.0,
                "warning",
                "Clipped arm target from "
                + np.array2string(unclipped_position, precision=3)
                + " to "
                + np.array2string(clipped_position, precision=3),
            )
        T_robot_target[:3, 3] = clipped_position

        if self.use_orientation:
            delta_rot_hand = self.neutral_hand_pose[:3, :3].T @ T_hand[:3, :3]
            delta_rot_robot = (
                HAND_TO_ROBOT_BASIS @ delta_rot_hand @ HAND_TO_ROBOT_BASIS.T
            )
            T_robot_target[:3, :3] = T_robot_target[:3, :3] @ delta_rot_robot

        return self.filter_robot_target(T_robot_target)

    def filter_robot_target(self, target):
        if self.filtered_target_pose is None:
            self.filtered_target_pose = target.copy()
            return target

        filtered = target.copy()
        filtered[:3, 3] = (
            (1.0 - self.target_filter_alpha) * self.filtered_target_pose[:3, 3]
            + self.target_filter_alpha * target[:3, 3]
        )

        if self.use_orientation:
            rotation = (
                (1.0 - self.target_filter_alpha) * self.filtered_target_pose[:3, :3]
                + self.target_filter_alpha * target[:3, :3]
            )
            u, _s, vt = np.linalg.svd(rotation)
            filtered[:3, :3] = u @ vt

        self.filtered_target_pose = filtered.copy()
        return filtered

    def filter_and_limit_active_q(self, active_q):
        now = time.monotonic()
        if self.filtered_arm_q is None:
            self.filtered_arm_q = active_q.copy()
            self.last_command_time = now
            return active_q

        dt = now - self.last_command_time if self.last_command_time is not None else 0.0
        self.last_command_time = now
        dt = max(dt, 1e-3)

        filtered = (
            (1.0 - self.joint_filter_alpha) * self.filtered_arm_q
            + self.joint_filter_alpha * active_q
        )

        if self.max_joint_velocity > 0.0:
            max_delta = self.max_joint_velocity * dt
            delta = np.clip(
                filtered - self.filtered_arm_q,
                -max_delta,
                max_delta,
            )
            filtered = self.filtered_arm_q + delta

        self.filtered_arm_q = self.clip_active_q_to_ik_limits(filtered)
        return self.filtered_arm_q.copy()

    def clip_active_q_to_ik_limits(self, active_q):
        try:
            lower = np.asarray(self.arm_ik.reduced_robot.model.lowerPositionLimit)[
                self.active_slice
            ]
            upper = np.asarray(self.arm_ik.reduced_robot.model.upperPositionLimit)[
                self.active_slice
            ]
        except Exception:
            return active_q

        if lower.shape != (7,) or upper.shape != (7,):
            return active_q
        return np.clip(active_q, lower, upper)

    def throttled_log(self, key, interval_sec, level, message):
        now = time.monotonic()
        last = self.last_log_time.get(key, 0.0)
        if now - last < interval_sec:
            return

        self.last_log_time[key] = now
        logger = self.get_logger()
        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LiveArmMapper()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


RightArmLiveMapper = LiveArmMapper

