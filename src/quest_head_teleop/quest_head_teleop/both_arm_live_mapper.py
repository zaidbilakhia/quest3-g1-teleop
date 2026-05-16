#!/usr/bin/env python3

import ctypes
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

try:
    from vr_haptic_msgs.msg import ManoLandmarks
except Exception:
    ManoLandmarks = None


# The Unitree xr_teleoperate IK imports `from pinocchio import casadi as cpin`.
# In this workspace that binding is available from the local conda-forge env.
# Override with XR_TELEOP_IK_PREFIX if the env moves.
DEFAULT_IK_PYTHON_PREFIX = "/home/zaid/micromamba/envs/xr-teleop-ik"
IK_REEXEC_ENV = "XR_TELEOP_IK_REEXECED"


def reexec_with_ik_library_path(prefix):
    if os.environ.get(IK_REEXEC_ENV) == "1":
        return
    if not prefix or not os.path.isdir(prefix):
        return

    lib_dir = os.path.join(prefix, "lib")
    if not os.path.isdir(lib_dir):
        return

    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(prefix, "lib", python_tag, "site-packages")
    ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    python_paths = [
        p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    if ld_paths and ld_paths[0] == lib_dir and python_paths and python_paths[0] == site_packages:
        return

    os.environ["LD_LIBRARY_PATH"] = (
        lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )
    if os.path.isdir(site_packages):
        os.environ["PYTHONPATH"] = (
            site_packages + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
    os.environ[IK_REEXEC_ENV] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


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


IK_PYTHON_PREFIX = os.environ.get("XR_TELEOP_IK_PREFIX", DEFAULT_IK_PYTHON_PREFIX)
reexec_with_ik_library_path(IK_PYTHON_PREFIX)
prefer_ik_python_prefix(IK_PYTHON_PREFIX)


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

ARM_SIDES = ("left", "right")
JOINT_NAMES_BY_SIDE = {
    "left": LEFT_ARM_JOINT_NAMES,
    "right": RIGHT_ARM_JOINT_NAMES,
}
JOINT_SLICE_BY_SIDE = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}

IDENTITY_BASIS = np.eye(3, dtype=float)

OLD_HUMAN_TO_ROBOT_BASIS = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

NEW_ROOM_HUMAN_TO_ROBOT_BASIS = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

SWAP_YZ_ORIENTATION_BASIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

SWAP_YZ_INVERSE_ORIENTATION_BASIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)

POSITION_BASIS_VARIANTS = {
    "identity": IDENTITY_BASIS,
    "old": OLD_HUMAN_TO_ROBOT_BASIS,
    "new_room": NEW_ROOM_HUMAN_TO_ROBOT_BASIS,
    "flip_x": np.diag([-1.0, 1.0, 1.0]),
    "flip_y": np.diag([1.0, -1.0, 1.0]),
    "flip_z": np.diag([1.0, 1.0, -1.0]),
}

ORIENTATION_BASIS_VARIANTS = {
    **POSITION_BASIS_VARIANTS,
    "swap_yz": SWAP_YZ_ORIENTATION_BASIS,
    "swap_yz_inverse": SWAP_YZ_INVERSE_ORIENTATION_BASIS,
    "flip_x_swap_yz": SWAP_YZ_ORIENTATION_BASIS @ POSITION_BASIS_VARIANTS["flip_x"],
    "flip_x_swap_yz_inverse": (
        SWAP_YZ_INVERSE_ORIENTATION_BASIS @ POSITION_BASIS_VARIANTS["flip_x"]
    ),
}

DEFAULT_HUMAN_TO_ROBOT_POSITION_BASIS = IDENTITY_BASIS.copy()

HAND_TO_ROBOT_ORIENTATION_BASIS_BY_SIDE = {
    "left": OLD_HUMAN_TO_ROBOT_BASIS.copy(),
    "right": OLD_HUMAN_TO_ROBOT_BASIS.copy(),
}


class LiveArmMapper(Node):
    """Quest/Unity wrist-pose to Unitree G1 arm JointState mapper.

    The node keeps Unitree's G1_29_ArmIK as the only IK solver. It builds
    dual-arm 4x4 wrist targets, solves both arms every cycle, and publishes
    only the requested active arm command topics.
    """

    def __init__(self, node_name="both_arm_live_mapper", default_arm_side="both"):
        super().__init__(node_name)
        self.default_arm_side = default_arm_side

        self.declare_and_read_params()
        self.setup_xr_repo_import()
        self.arm_ik = self.create_arm_ik()

        self.left_wrist_neutral, self.right_wrist_neutral = self.make_neutral_targets()
        self.neutral_target_by_side = {
            "left": self.left_wrist_neutral,
            "right": self.right_wrist_neutral,
        }

        self.latest_hand_pose = {side: None for side in ARM_SIDES}
        self.neutral_hand_pose = {side: None for side in ARM_SIDES}
        self.filtered_target_pose = {side: None for side in ARM_SIDES}
        self.last_unfiltered_target_position = {side: None for side in ARM_SIDES}
        self.filtered_arm_q = {side: None for side in ARM_SIDES}
        self.last_command_time = {side: None for side in ARM_SIDES}
        self.current_lr_arm_q = np.zeros(14, dtype=float)
        self.current_lr_arm_dq = np.zeros(14, dtype=float)
        self.have_joint_state = {side: False for side in ARM_SIDES}
        self.deadman_active = not self.deadman_enabled
        self.last_log_time = {}
        self.test_start_time = time.monotonic()
        self.candidate_results = []
        self.target_clip_count = {side: 0 for side in ARM_SIDES}
        self.target_total_count = {side: 0 for side in ARM_SIDES}
        self.last_requested_target_position = {side: None for side in ARM_SIDES}
        self.current_replay_step = "unknown"

        self.setup_input_subscriptions()
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            20,
        )
        self.cmd_pubs = {
            "left": self.create_publisher(JointState, self.left_cmd_topic, 10),
            "right": self.create_publisher(JointState, self.right_cmd_topic, 10),
        }
        if self.deadman_enabled:
            self.deadman_sub = self.create_subscription(
                Bool,
                self.deadman_topic,
                self.deadman_callback,
                10,
            )
        else:
            self.deadman_sub = None
        self.current_step_sub = self.create_subscription(
            String,
            "/replay/current_step",
            self.current_step_callback,
            10,
        )

        self.calibrate_srv = self.create_service(
            Trigger,
            "~/calibrate_neutral",
            self.calibrate_neutral_callback,
        )

        if self.test_mode == "candidate_search":
            self.candidate_results = self.run_candidate_search()

        timer_period = 1.0 / max(self.publish_rate, 1e-6)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            "both_arm_live_mapper ready: "
            f"arm_side={self.arm_side}, active_sides={self.active_sides}, "
            f"pose_source={self.pose_source}, test_mode={self.test_mode}, "
            f"position_basis_variant={self.position_basis_variant}, "
            f"global_position_gain={self.global_position_gain}, "
            f"joint_state_topic={self.joint_state_topic}, "
            f"left_cmd_topic={self.left_cmd_topic}, "
            f"right_cmd_topic={self.right_cmd_topic}, "
            f"publish_rate={self.publish_rate}, "
            "calibration_service=~/calibrate_neutral"
        )
        self.get_logger().info(
            "Position basis:\n"
            + np.array2string(
                self.human_to_robot_position_basis,
                precision=3,
                suppress_small=True,
            )
        )
        self.get_logger().info(
            "Orientation enabled: "
            f"use_orientation={self.use_orientation}, "
            f"left_use_orientation={self.use_orientation_by_side['left']}, "
            f"right_use_orientation={self.use_orientation_by_side['right']}"
        )
        self.get_logger().info(
            "Position gains: "
            f"left={np.array2string(self.position_gain_by_side['left'], precision=3)}, "
            f"right={np.array2string(self.position_gain_by_side['right'], precision=3)}"
        )

    def declare_and_read_params(self):
        self.declare_parameter("arm_side", self.default_arm_side)
        self.arm_side = str(self.get_parameter("arm_side").value).strip().lower()
        if self.arm_side not in ("left", "right", "both"):
            self.get_logger().warning(
                f"Unsupported arm_side={self.arm_side!r}; using 'both'."
            )
            self.arm_side = "both"

        self.active_sides = (
            list(ARM_SIDES) if self.arm_side == "both" else [self.arm_side]
        )

        self.declare_parameter("pose_source", "tf")
        self.declare_parameter("test_mode", "none")
        self.declare_parameter("tf_topic", "/tf")
        self.declare_parameter("right_hand_tf_frame", "hand_right")
        self.declare_parameter("left_hand_tf_frame", "hand_left")
        self.declare_parameter("right_hand_pose_topic", "/quest/hand_pose")
        self.declare_parameter("left_hand_pose_topic", "/quest/left_hand_pose")
        self.declare_parameter("hand_wrist_landmark_index", 0)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("right_cmd_topic", "/g1/right_arm_cmd")
        self.declare_parameter("left_cmd_topic", "/g1/left_arm_cmd")
        self.declare_parameter(
            "xr_repo_path",
            "/home/zaid/humanoid_ws/assets/xr_teleoperate",
        )
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("global_position_gain", 1.0)
        self.declare_parameter("position_gain_x", 1.3)
        self.declare_parameter("position_gain_y", 1.3)
        self.declare_parameter("position_gain_z", 1.2)
        self.declare_parameter("left_position_gain_x", 1.3)
        self.declare_parameter("left_position_gain_y", 1.3)
        self.declare_parameter("left_position_gain_z", 1.2)
        self.declare_parameter("right_position_gain_x", 1.3)
        self.declare_parameter("right_position_gain_y", 1.3)
        self.declare_parameter("right_position_gain_z", 1.2)
        self.declare_parameter("position_basis_variant", "identity")
        for row in range(3):
            for col in range(3):
                default_value = float(IDENTITY_BASIS[row, col])
                self.declare_parameter(
                    f"position_basis_r{row}{col}",
                    default_value,
                )
        self.declare_parameter("use_orientation", False)
        self.declare_parameter("left_use_orientation", False)
        self.declare_parameter("right_use_orientation", False)
        self.declare_parameter("left_orientation_basis_variant", "flip_x_swap_yz")
        self.declare_parameter("right_orientation_basis_variant", "swap_yz")
        self.declare_parameter("debug", True)
        self.declare_parameter("right_neutral_x", 0.32)
        self.declare_parameter("right_neutral_y", -0.20)
        self.declare_parameter("right_neutral_z", 0.20)
        self.declare_parameter("left_neutral_x", 0.32)
        self.declare_parameter("left_neutral_y", 0.20)
        self.declare_parameter("left_neutral_z", 0.20)
        self.declare_parameter("left_neutral_roll", math.pi / 2.0)
        self.declare_parameter("left_neutral_pitch", 0.0)
        self.declare_parameter("left_neutral_yaw", 0.0)
        self.declare_parameter("right_neutral_roll", -math.pi / 2.0)
        self.declare_parameter("right_neutral_pitch", 0.0)
        self.declare_parameter("right_neutral_yaw", 0.0)
        self.declare_parameter(
            "human_to_robot_position_basis",
            DEFAULT_HUMAN_TO_ROBOT_POSITION_BASIS.reshape(-1).tolist(),
        )
        self.declare_parameter("left_workspace_min_x", 0.08)
        self.declare_parameter("left_workspace_max_x", 0.65)
        self.declare_parameter("left_workspace_min_y", 0.02)
        self.declare_parameter("left_workspace_max_y", 0.55)
        self.declare_parameter("left_workspace_min_z", -0.10)
        self.declare_parameter("left_workspace_max_z", 0.60)
        self.declare_parameter("right_workspace_min_x", 0.08)
        self.declare_parameter("right_workspace_max_x", 0.65)
        self.declare_parameter("right_workspace_min_y", -0.55)
        self.declare_parameter("right_workspace_max_y", -0.02)
        self.declare_parameter("right_workspace_min_z", -0.10)
        self.declare_parameter("right_workspace_max_z", 0.60)
        # Higher target_filter_alpha follows the hand faster but less smoothly.
        self.declare_parameter("target_filter_alpha", 0.35)
        # Higher joint_filter_alpha follows IK faster but less smoothly.
        self.declare_parameter("joint_filter_alpha", 0.45)
        # Higher max_joint_velocity makes the robot respond faster.
        self.declare_parameter("max_joint_velocity", 1.8)
        self.declare_parameter("max_target_jump", 0.25)
        self.declare_parameter("max_target_jump_m", 0.35)
        self.declare_parameter("deadman_enabled", False)
        self.declare_parameter("deadman_topic", "/teleop/deadman")
        self.declare_parameter("test_amplitude", 0.05)
        self.declare_parameter("test_period", 4.0)
        self.declare_parameter("candidate_publish_duration", 3.0)

        self.pose_source = str(self.get_parameter("pose_source").value).strip().lower()
        if self.pose_source not in ("tf", "landmarks"):
            self.get_logger().warning(
                f"Unsupported pose_source={self.pose_source!r}; using 'tf'."
            )
            self.pose_source = "tf"

        self.test_mode = str(self.get_parameter("test_mode").value).strip().lower()
        supported_test_modes = (
            "none",
            "static",
            "sweep",
            "left_sweep",
            "right_sweep",
            "gain_sweep",
            "circle",
            "candidate_search",
        )
        if self.test_mode not in supported_test_modes:
            self.get_logger().warning(
                f"Unsupported test_mode={self.test_mode!r}; using 'none'."
            )
            self.test_mode = "none"

        self.tf_topic = self.get_parameter("tf_topic").value
        self.hand_tf_frames = {
            "left": self.get_parameter("left_hand_tf_frame").value,
            "right": self.get_parameter("right_hand_tf_frame").value,
        }
        self.hand_pose_topics = {
            "left": self.get_parameter("left_hand_pose_topic").value,
            "right": self.get_parameter("right_hand_pose_topic").value,
        }
        self.hand_wrist_landmark_index = int(
            self.get_parameter("hand_wrist_landmark_index").value
        )
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.left_cmd_topic = self.get_parameter("left_cmd_topic").value
        self.right_cmd_topic = self.get_parameter("right_cmd_topic").value
        self.xr_repo_path = os.path.abspath(self.get_parameter("xr_repo_path").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.position_scale = float(self.get_parameter("position_scale").value)
        self.global_position_gain = float(
            self.get_parameter("global_position_gain").value
        )
        self.generic_position_gain = self.read_vector_param("position_gain")
        self.position_gain_by_side = {
            "left": self.read_side_position_gain("left"),
            "right": self.read_side_position_gain("right"),
        }
        self.position_basis_variant = str(
            self.get_parameter("position_basis_variant").value
        ).strip().lower()
        self.use_orientation = bool(self.get_parameter("use_orientation").value)
        self.use_orientation_by_side = {
            "left": bool(self.get_parameter("left_use_orientation").value),
            "right": bool(self.get_parameter("right_use_orientation").value),
        }
        self.orientation_basis_variant_by_side = {
            "left": str(
                self.get_parameter("left_orientation_basis_variant").value
            ).strip().lower(),
            "right": str(
                self.get_parameter("right_orientation_basis_variant").value
            ).strip().lower(),
        }
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
        self.max_target_jump = max(
            0.0,
            float(self.get_parameter("max_target_jump_m").value),
        )
        self.deadman_enabled = bool(self.get_parameter("deadman_enabled").value)
        self.deadman_topic = self.get_parameter("deadman_topic").value
        self.test_amplitude = float(self.get_parameter("test_amplitude").value)
        self.test_period = max(0.1, float(self.get_parameter("test_period").value))
        self.candidate_publish_duration = max(
            0.5,
            float(self.get_parameter("candidate_publish_duration").value),
        )

        self.neutral_position_by_side = {
            "left": self.read_vector_param("left_neutral"),
            "right": self.read_vector_param("right_neutral"),
        }
        self.neutral_rpy_by_side = {
            "left": np.array(
                [
                    float(self.get_parameter("left_neutral_roll").value),
                    float(self.get_parameter("left_neutral_pitch").value),
                    float(self.get_parameter("left_neutral_yaw").value),
                ],
                dtype=float,
            ),
            "right": np.array(
                [
                    float(self.get_parameter("right_neutral_roll").value),
                    float(self.get_parameter("right_neutral_pitch").value),
                    float(self.get_parameter("right_neutral_yaw").value),
                ],
                dtype=float,
            ),
        }
        self.left_neutral_position = self.neutral_position_by_side["left"]
        self.right_neutral_position = self.neutral_position_by_side["right"]

        self.human_to_robot_position_basis = self.read_position_basis()
        self.hand_to_robot_orientation_basis_by_side = {
            side: self.read_orientation_basis(side) for side in ARM_SIDES
        }

        self.workspace_min_by_side = {
            "left": self.read_vector_param("left_workspace_min"),
            "right": self.read_vector_param("right_workspace_min"),
        }
        self.workspace_max_by_side = {
            "left": self.read_vector_param("left_workspace_max"),
            "right": self.read_vector_param("right_workspace_max"),
        }
        for side in ARM_SIDES:
            lower = np.minimum(
                self.workspace_min_by_side[side],
                self.workspace_max_by_side[side],
            )
            upper = np.maximum(
                self.workspace_min_by_side[side],
                self.workspace_max_by_side[side],
            )
            self.workspace_min_by_side[side] = lower
            self.workspace_max_by_side[side] = upper

    def read_vector_param(self, prefix):
        return np.array(
            [
                float(self.get_parameter(f"{prefix}_x").value),
                float(self.get_parameter(f"{prefix}_y").value),
                float(self.get_parameter(f"{prefix}_z").value),
            ],
            dtype=float,
        )

    def read_side_position_gain(self, side):
        side_gain = self.read_vector_param(f"{side}_position_gain")
        side_default = np.array([1.3, 1.3, 1.2], dtype=float)
        if np.allclose(side_gain, side_default):
            return self.generic_position_gain.copy()
        return side_gain

    def read_matrix_param(self, name, default):
        value = self.get_parameter(name).value
        try:
            matrix = np.asarray(value, dtype=float).reshape(3, 3)
        except Exception:
            self.get_logger().warning(
                f"Invalid {name}; expected 9 floats. Using default."
            )
            matrix = default.copy()
        return matrix

    def read_position_basis(self):
        if self.position_basis_variant == "custom":
            values = []
            for row in range(3):
                for col in range(3):
                    values.append(
                        float(
                            self.get_parameter(
                                f"position_basis_r{row}{col}"
                            ).value
                        )
                    )
            return np.asarray(values, dtype=float).reshape(3, 3)

        basis = POSITION_BASIS_VARIANTS.get(self.position_basis_variant)
        if basis is None:
            self.get_logger().warning(
                f"Unsupported position_basis_variant="
                f"{self.position_basis_variant!r}; using 'identity'."
            )
            self.position_basis_variant = "identity"
            basis = POSITION_BASIS_VARIANTS[self.position_basis_variant]
        return basis.copy()

    def read_orientation_basis(self, side):
        variant = self.orientation_basis_variant_by_side[side]
        basis = ORIENTATION_BASIS_VARIANTS.get(variant)
        if basis is None:
            self.get_logger().warning(
                f"Unsupported {side}_orientation_basis_variant={variant!r}; "
                "using 'identity'."
            )
            self.orientation_basis_variant_by_side[side] = "identity"
            basis = ORIENTATION_BASIS_VARIANTS["identity"]
        return basis.copy()

    def setup_xr_repo_import(self):
        self.xr_teleop_path = os.path.join(self.xr_repo_path, "teleop")
        if self.xr_teleop_path not in sys.path:
            sys.path.append(self.xr_teleop_path)
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

        old_cwd = os.getcwd()
        try:
            os.chdir(self.xr_teleop_path)
            return G1_29_ArmIK()
        finally:
            os.chdir(old_cwd)

    def setup_input_subscriptions(self):
        self.tf_sub = None
        self.hand_pose_subs = {}
        if self.test_mode != "none":
            self.get_logger().info(
                f"test_mode={self.test_mode}; live Quest input and calibration are not required."
            )
            return

        if self.pose_source == "tf":
            self.tf_sub = self.create_subscription(
                TFMessage,
                self.tf_topic,
                self.tf_callback,
                50,
            )
            return

        if ManoLandmarks is None:
            self.get_logger().error(
                "pose_source=landmarks but vr_haptic_msgs/ManoLandmarks is not importable."
            )
            return

        for side in self.active_sides:
            self.hand_pose_subs[side] = self.create_subscription(
                ManoLandmarks,
                self.hand_pose_topics[side],
                lambda msg, side=side: self.hand_pose_callback(side, msg),
                10,
            )

    @staticmethod
    def clamp_float(value, lower, upper):
        return max(lower, min(upper, value))

    @staticmethod
    def normalize_vector(vector, min_norm=1e-6):
        norm = np.linalg.norm(vector)
        if norm < min_norm:
            return None
        return vector / norm

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion, min_norm=1e-8):
        q = np.asarray(quaternion, dtype=float)
        norm = np.linalg.norm(q)
        if norm < min_norm:
            return np.eye(3, dtype=float)
        x, y, z, w = q / norm
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def rpy_to_rotation_matrix(roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rz @ ry @ rx

    @staticmethod
    def make_transform(position, rotation=None):
        T = np.eye(4, dtype=float)
        T[:3, 3] = np.asarray(position, dtype=float)
        if rotation is not None:
            T[:3, :3] = np.asarray(rotation, dtype=float)
        return T

    def matrix_to_pose_debug(self, T):
        p = T[:3, 3]
        return f"x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}"

    def make_neutral_targets(self):
        left_rotation = self.rpy_to_rotation_matrix(*self.neutral_rpy_by_side["left"])
        right_rotation = self.rpy_to_rotation_matrix(*self.neutral_rpy_by_side["right"])
        left = self.make_transform(self.left_neutral_position, left_rotation)
        right = self.make_transform(self.right_neutral_position, right_rotation)

        self.get_logger().info(
            "Neutral wrist targets:\n"
            f"left=\n{np.array2string(left, precision=4, suppress_small=True)}\n"
            f"right=\n{np.array2string(right, precision=4, suppress_small=True)}\n"
            "Tuning notes: if fingers point upward, adjust neutral_pitch/yaw; "
            "if palms face upward, adjust neutral_roll; if a side is mirrored, "
            "tune that side's neutral yaw/roll or orientation basis."
        )
        return left, right

    def tf_callback(self, msg):
        for transform in msg.transforms:
            for side in self.active_sides:
                child_frame_id = transform.child_frame_id.lstrip("/")
                expected_frame_id = self.hand_tf_frames[side].lstrip("/")
                if child_frame_id != expected_frame_id:
                    continue
                try:
                    self.latest_hand_pose[side] = self.transform_msg_to_matrix(
                        transform.transform
                    )
                except Exception as exc:
                    self.throttled_log(
                        f"{side}_tf_parse_error",
                        2.0,
                        "error",
                        f"Failed to convert TF {transform.child_frame_id}: {exc}",
                    )

    def transform_msg_to_matrix(self, transform_msg):
        T = np.eye(4, dtype=float)
        t = transform_msg.translation
        r = transform_msg.rotation
        T[:3, 3] = [t.x, t.y, t.z]
        T[:3, :3] = self.quaternion_to_rotation_matrix([r.x, r.y, r.z, r.w])
        return T

    def hand_pose_callback(self, side, msg):
        try:
            self.latest_hand_pose[side] = self.hand_landmarks_msg_to_matrix(msg)
        except Exception as exc:
            self.throttled_log(
                f"{side}_landmarks_parse_error",
                2.0,
                "error",
                f"Failed to convert {side} ManoLandmarks: {exc}",
            )

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
                "not enough landmarks for orientation: "
                f"got {len(msg.landmarks)}, need index {max(required_indices)}"
            )
        points = []
        for index in required_indices:
            p = msg.landmarks[index]
            points.append(np.array([p.x, p.y, p.z], dtype=float))

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

    def deadman_callback(self, msg):
        self.deadman_active = bool(msg.data)

    def current_step_callback(self, msg):
        value = msg.data.strip()
        if value:
            self.current_replay_step = value

    def calibrate_neutral_callback(self, _request, response):
        if self.test_mode != "none":
            response.success = True
            response.message = (
                f"test_mode={self.test_mode}; calibration is bypassed and neutral robot targets are used."
            )
            return response

        missing = [
            side for side in self.active_sides if self.latest_hand_pose[side] is None
        ]
        if missing:
            response.success = False
            response.message = self.missing_input_message(missing)
            return response

        for side in self.active_sides:
            self.neutral_hand_pose[side] = self.latest_hand_pose[side].copy()
            self.filtered_target_pose[side] = None
            self.last_unfiltered_target_position[side] = None
            self.filtered_arm_q[side] = None
            self.last_command_time[side] = None

        response.success = True
        response.message = (
            "Captured current human wrist poses as neutral: "
            + ", ".join(
                f"{side} {self.matrix_to_pose_debug(self.neutral_hand_pose[side])}"
                for side in self.active_sides
            )
        )
        self.get_logger().info(response.message)
        return response

    def missing_input_message(self, missing):
        if self.pose_source == "tf":
            return (
                "No TF input received yet for: "
                + ", ".join(
                    f"{side} child_frame_id={self.hand_tf_frames[side]}"
                    for side in missing
                )
                + f" on {self.tf_topic}."
            )
        return (
            "No ManoLandmarks input received yet for: "
            + ", ".join(
                f"{side} topic={self.hand_pose_topics[side]}" for side in missing
            )
            + "."
        )

    def joint_state_callback(self, msg):
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        lr_arm_q = np.zeros(14, dtype=float)
        lr_arm_dq = np.zeros(14, dtype=float)
        missing = {side: [] for side in ARM_SIDES}

        for side, joint_names in JOINT_NAMES_BY_SIDE.items():
            offset = 0 if side == "left" else 7
            for joint_index, joint_name in enumerate(joint_names):
                msg_index = name_to_index.get(joint_name)
                if msg_index is None:
                    if side in self.active_sides:
                        missing[side].append(joint_name)
                    continue
                out_index = offset + joint_index
                if msg_index < len(msg.position):
                    lr_arm_q[out_index] = msg.position[msg_index]
                if msg_index < len(msg.velocity):
                    lr_arm_dq[out_index] = msg.velocity[msg_index]

        self.current_lr_arm_q = lr_arm_q
        self.current_lr_arm_dq = lr_arm_dq
        self.have_joint_state = {
            side: len(missing[side]) == 0 for side in ARM_SIDES
        }
        for side in self.active_sides:
            if missing[side]:
                self.throttled_log(
                    f"{side}_missing_joint_states",
                    5.0,
                    "warning",
                    f"Missing {side} arm joints in JointState: "
                    + ", ".join(missing[side]),
                )

    def timer_callback(self):
        if self.deadman_enabled and not self.deadman_active:
            self.throttled_log(
                "deadman_inactive",
                1.0,
                "warning",
                f"deadman_enabled=true but {self.deadman_topic} is not active; not publishing.",
            )
            return

        if self.test_mode != "none":
            left_wrist_pose, right_wrist_pose = self.make_test_targets()
        else:
            missing_input = [
                side
                for side in self.active_sides
                if self.latest_hand_pose[side] is None
            ]
            if missing_input:
                self.throttled_log(
                    "waiting_for_input",
                    5.0,
                    "info",
                    self.missing_input_message(missing_input),
                )
                return

            missing_calibration = [
                side
                for side in self.active_sides
                if self.neutral_hand_pose[side] is None
            ]
            if missing_calibration:
                self.throttled_log(
                    "waiting_for_calibration",
                    5.0,
                    "info",
                    "Waiting for intentional neutral calibration. Run: "
                    f"ros2 service call /{self.get_name()}/calibrate_neutral "
                    "std_srvs/srv/Trigger {}",
                )
                return

            try:
                left_wrist_pose = (
                    self.convert_hand_pose_to_robot_target(
                        "left",
                        self.latest_hand_pose["left"],
                    )
                    if "left" in self.active_sides
                    else self.left_wrist_neutral.copy()
                )
                right_wrist_pose = (
                    self.convert_hand_pose_to_robot_target(
                        "right",
                        self.latest_hand_pose["right"],
                    )
                    if "right" in self.active_sides
                    else self.right_wrist_neutral.copy()
                )
            except ValueError as exc:
                self.throttled_log("target_rejected", 1.0, "warning", str(exc))
                return

        for side in self.active_sides:
            if not self.have_joint_state[side]:
                self.throttled_log(
                    f"{side}_using_zero_joint_state",
                    5.0,
                    "warning",
                    f"No complete {side} arm joint state yet; using zeros.",
                )

        self.solve_and_publish(left_wrist_pose, right_wrist_pose)

    def make_test_targets(self):
        left = self.left_wrist_neutral.copy()
        right = self.right_wrist_neutral.copy()
        t = time.monotonic() - self.test_start_time
        phase = 2.0 * math.pi * t / self.test_period

        if self.test_mode in ("sweep", "left_sweep", "right_sweep"):
            offset = np.array(
                [
                    self.test_amplitude * math.sin(phase),
                    0.03 * math.sin(phase + math.pi / 2.0),
                    0.03 * math.sin(phase),
                ],
                dtype=float,
            )
            if self.test_mode in ("sweep", "left_sweep"):
                left[:3, 3] += offset
            if self.test_mode in ("sweep", "right_sweep"):
                right[:3, 3] += offset
            self.throttled_log(
                "test_target",
                1.0,
                "info",
                f"test_mode={self.test_mode}, offset="
                + np.array2string(offset, precision=3),
            )
        elif self.test_mode == "gain_sweep":
            amplitudes = (0.02, 0.05, 0.08, 0.12)
            segment = self.test_period
            index = int(t / segment) % len(amplitudes)
            amplitude = amplitudes[index]
            local_phase = 2.0 * math.pi * (t % segment) / segment
            offset = np.array(
                [
                    amplitude * math.sin(local_phase),
                    0.7 * amplitude * math.sin(local_phase + math.pi / 2.0),
                    0.6 * amplitude * math.sin(local_phase),
                ],
                dtype=float,
            )
            left[:3, 3] = self.clip_target_position(
                "left",
                self.left_neutral_position + offset,
                "gain_sweep",
            )
            right[:3, 3] = self.clip_target_position(
                "right",
                self.right_neutral_position + offset,
                "gain_sweep",
            )
            self.throttled_log(
                "test_target",
                1.0,
                "info",
                f"test_mode=gain_sweep amplitude={amplitude:.2f} m, offset="
                + np.array2string(offset, precision=3),
            )
        elif self.test_mode == "circle":
            offset = np.array(
                [
                    self.test_amplitude * math.cos(phase),
                    0.0,
                    self.test_amplitude * math.sin(phase),
                ],
                dtype=float,
            )
            left[:3, 3] += offset
            right[:3, 3] += offset
        elif self.test_mode == "candidate_search" and self.candidate_results:
            index = int(t / self.candidate_publish_duration) % len(
                self.candidate_results
            )
            left = self.candidate_results[index]["left_target"].copy()
            right = self.candidate_results[index]["right_target"].copy()
            self.throttled_log(
                "candidate_publish",
                1.0,
                "info",
                f"Publishing candidate {index}: {self.candidate_results[index]['label']}",
            )

        return left, right

    def clip_target_position(self, side, position, source):
        clipped_position = np.clip(
            position,
            self.workspace_min_by_side[side],
            self.workspace_max_by_side[side],
        )
        clipped = not np.allclose(position, clipped_position)
        self.target_total_count[side] += 1
        if clipped:
            self.target_clip_count[side] += 1
            self.throttled_log(
                f"{side}_workspace_clipped",
                1.0,
                "warning",
                f"Clipped {side} {source} target from "
                + np.array2string(position, precision=3)
                + " to "
                + np.array2string(clipped_position, precision=3),
            )
        self.report_clipping_stats(side)
        self.last_requested_target_position[side] = clipped_position.copy()
        return clipped_position

    def report_clipping_stats(self, side):
        total = self.target_total_count[side]
        if total <= 0:
            return
        percent = 100.0 * float(self.target_clip_count[side]) / float(total)
        message = (
            f"{side} workspace clipping: "
            f"{self.target_clip_count[side]}/{total} targets "
            f"({percent:.1f}%). "
            "If this is frequent, increase workspace or reduce gain."
        )
        self.throttled_log(
            f"{side}_workspace_clip_percent",
            5.0,
            "info" if percent < 10.0 else "warning",
            message,
        )

    def convert_hand_pose_to_robot_target(self, side, T_hand):
        T_robot_target = self.neutral_target_by_side[side].copy()
        delta_pos_hand = T_hand[:3, 3] - self.neutral_hand_pose[side][:3, 3]
        delta_robot_before_gain = self.human_to_robot_position_basis @ delta_pos_hand
        gain_vector = self.position_gain_by_side[side] * self.global_position_gain
        delta_robot_after_gain = (
            delta_robot_before_gain
            * gain_vector
            * self.position_scale
        )
        unclipped_position = (
            self.neutral_position_by_side[side] + delta_robot_after_gain
        )
        clipped_position = self.clip_target_position(
            side,
            unclipped_position,
            "live",
        )

        previous = self.last_unfiltered_target_position[side]
        jump = 0.0
        rejected_by_jump = False
        if previous is not None and self.max_target_jump > 0.0:
            jump = float(np.linalg.norm(clipped_position - previous))
            if jump > self.max_target_jump:
                rejected_by_jump = True
                if self.debug:
                    self.log_target_debug(
                        side,
                        delta_pos_hand,
                        delta_robot_before_gain,
                        gain_vector,
                        delta_robot_after_gain,
                        unclipped_position,
                        clipped_position,
                        jump,
                        rejected_by_jump,
                        None,
                    )
                raise ValueError(
                    f"Rejected {side} target jump {jump:.3f} m > "
                    f"max_target_jump={self.max_target_jump:.3f}."
                )
        self.last_unfiltered_target_position[side] = clipped_position.copy()
        T_robot_target[:3, 3] = clipped_position

        if self.should_use_orientation(side):
            delta_rot_hand = self.neutral_hand_pose[side][:3, :3].T @ T_hand[:3, :3]
            basis = self.hand_to_robot_orientation_basis_by_side[side]
            delta_rot_robot = basis @ delta_rot_hand @ basis.T
            T_robot_target[:3, :3] = T_robot_target[:3, :3] @ delta_rot_robot

        filtered_target = self.filter_robot_target(side, T_robot_target)
        if self.debug:
            self.log_target_debug(
                side,
                delta_pos_hand,
                delta_robot_before_gain,
                gain_vector,
                delta_robot_after_gain,
                unclipped_position,
                clipped_position,
                jump,
                rejected_by_jump,
                filtered_target[:3, 3],
            )
        return filtered_target

    def log_target_debug(
        self,
        side,
        delta_pos_hand,
        delta_robot_before_gain,
        gain_vector,
        delta_robot_after_gain,
        unclipped_position,
        clipped_position,
        jump,
        rejected_by_jump,
        filtered_position,
    ):
        filtered_text = "none"
        if filtered_position is not None:
            filtered_text = np.array2string(filtered_position, precision=3)
        self.throttled_log(
            f"{side}_target_debug",
            0.5,
            "info",
            f"step={self.current_replay_step}, side={side}, "
            "delta_pos_hand="
            + np.array2string(delta_pos_hand, precision=3)
            + ", delta_robot_before_gain="
            + np.array2string(delta_robot_before_gain, precision=3)
            + ", gain_vector="
            + np.array2string(gain_vector, precision=3)
            + ", delta_robot_after_gain="
            + np.array2string(delta_robot_after_gain, precision=3)
            + ", target_before_clip="
            + np.array2string(unclipped_position, precision=3)
            + ", target_after_clip="
            + np.array2string(clipped_position, precision=3)
            + f", clipped={not np.allclose(unclipped_position, clipped_position)}"
            + f", jump_from_previous={jump:.3f}"
            + f", rejected_by_max_target_jump={rejected_by_jump}"
            + ", final_filtered_target="
            + filtered_text,
        )

    def should_use_orientation(self, side):
        return self.use_orientation and self.use_orientation_by_side[side]

    def filter_robot_target(self, side, target):
        if self.filtered_target_pose[side] is None:
            self.filtered_target_pose[side] = target.copy()
            return target

        filtered = target.copy()
        filtered[:3, 3] = (
            (1.0 - self.target_filter_alpha)
            * self.filtered_target_pose[side][:3, 3]
            + self.target_filter_alpha * target[:3, 3]
        )
        if self.should_use_orientation(side):
            rotation = (
                (1.0 - self.target_filter_alpha)
                * self.filtered_target_pose[side][:3, :3]
                + self.target_filter_alpha * target[:3, :3]
            )
            u, _s, vt = np.linalg.svd(rotation)
            filtered[:3, :3] = u @ vt

        self.filtered_target_pose[side] = filtered.copy()
        return filtered

    def solve_and_publish(self, left_wrist_pose, right_wrist_pose):
        try:
            sol_q, _sol_tauff = self.arm_ik.solve_ik(
                left_wrist_pose,
                right_wrist_pose,
                self.current_lr_arm_q.copy(),
                self.current_lr_arm_dq.copy(),
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

        active_q_by_side = {}
        for side in self.active_sides:
            q = sol_q[JOINT_SLICE_BY_SIDE[side]].copy()
            if not np.all(np.isfinite(q)):
                self.throttled_log(
                    f"{side}_ik_nonfinite",
                    2.0,
                    "error",
                    f"IK returned NaN/Inf for {side}: {q}; not publishing.",
                )
                return
            raw_q = q.copy()
            q, clipped_joints = self.clip_active_q_to_ik_limits_with_info(side, q)
            filtered_q = self.filter_and_limit_active_q(side, q)
            active_q_by_side[side] = filtered_q
            if self.debug:
                self.throttled_log(
                    f"{side}_q_debug",
                    0.5,
                    "info",
                    f"step={self.current_replay_step}, side={side}, "
                    "active_q_before_limit="
                    + np.array2string(raw_q, precision=3)
                    + ", active_q_after_limit="
                    + np.array2string(q, precision=3)
                    + ", clipped_joints="
                    + str(clipped_joints)
                    + ", final_filtered_q="
                    + np.array2string(filtered_q, precision=3),
                )
            if self.test_mode == "gain_sweep":
                self.throttled_log(
                    f"{side}_gain_sweep_q_range",
                    1.0,
                    "info",
                    f"{side} gain_sweep raw_q_range="
                    f"[{float(np.min(raw_q)):.3f}, {float(np.max(raw_q)):.3f}], "
                    f"cmd_q_range="
                    f"[{float(np.min(q)):.3f}, {float(np.max(q)):.3f}]",
                )

        for side, q in active_q_by_side.items():
            self.publish_active_arm(side, q)

        if self.debug:
            pieces = [
                "left target " + self.matrix_to_pose_debug(left_wrist_pose),
                "right target " + self.matrix_to_pose_debug(right_wrist_pose),
            ]
            pieces.extend(
                f"{side}_q={np.array2string(q, precision=3)}"
                for side, q in active_q_by_side.items()
            )
            self.throttled_log("debug_publish", 1.0, "info", ", ".join(pieces))

    def publish_active_arm(self, side, active_q):
        if not rclpy.ok():
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES_BY_SIDE[side])
        msg.position = [float(q) for q in active_q]
        msg.velocity = []
        msg.effort = []
        try:
            self.cmd_pubs[side].publish(msg)
        except Exception as exc:
            self.throttled_log(
                f"{side}_publish_error",
                2.0,
                "warning",
                f"Failed to publish {side} arm command: {exc}",
            )

    def filter_and_limit_active_q(self, side, active_q):
        now = time.monotonic()
        if self.filtered_arm_q[side] is None:
            self.filtered_arm_q[side] = active_q.copy()
            self.last_command_time[side] = now
            return active_q

        dt = now - self.last_command_time[side]
        self.last_command_time[side] = now
        dt = max(dt, 1e-3)
        filtered = (
            (1.0 - self.joint_filter_alpha) * self.filtered_arm_q[side]
            + self.joint_filter_alpha * active_q
        )
        if self.max_joint_velocity > 0.0:
            max_delta = self.max_joint_velocity * dt
            delta = np.clip(
                filtered - self.filtered_arm_q[side],
                -max_delta,
                max_delta,
            )
            filtered = self.filtered_arm_q[side] + delta

        self.filtered_arm_q[side] = self.clip_active_q_to_ik_limits(side, filtered)
        return self.filtered_arm_q[side].copy()

    def clip_active_q_to_ik_limits(self, side, active_q):
        clipped_q, _clipped_joints = self.clip_active_q_to_ik_limits_with_info(
            side,
            active_q,
        )
        return clipped_q

    def clip_active_q_to_ik_limits_with_info(self, side, active_q):
        try:
            lower = np.asarray(self.arm_ik.reduced_robot.model.lowerPositionLimit)[
                JOINT_SLICE_BY_SIDE[side]
            ]
            upper = np.asarray(self.arm_ik.reduced_robot.model.upperPositionLimit)[
                JOINT_SLICE_BY_SIDE[side]
            ]
        except Exception:
            return active_q, []
        if lower.shape != (7,) or upper.shape != (7,):
            return active_q, []
        clipped_q = np.clip(active_q, lower, upper)
        clipped_indices = np.where(np.abs(clipped_q - active_q) > 1e-6)[0]
        joint_names = [
            JOINT_NAMES_BY_SIDE[side][int(index)]
            for index in clipped_indices
        ]
        if clipped_indices.size:
            target = self.last_requested_target_position.get(side)
            target_text = "unknown"
            if target is not None:
                target_text = np.array2string(target, precision=3)
            self.throttled_log(
                f"{side}_joint_limit_clipped",
                2.0,
                "warning",
                f"{side} IK joint limit clipping for target {target_text}: "
                f"joints={joint_names}, q_before_clip="
                + np.array2string(active_q, precision=3)
                + ", q_after_clip="
                + np.array2string(clipped_q, precision=3),
            )
        return clipped_q, joint_names

    def run_candidate_search(self):
        candidates = [
            ("roll +/-90", (math.pi / 2, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0)),
            ("identity", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ("roll +/-90 yaw +/-10", (math.pi / 2, 0.0, 0.17), (-math.pi / 2, 0.0, -0.17)),
            ("roll +/-90 yaw -/+10", (math.pi / 2, 0.0, -0.17), (-math.pi / 2, 0.0, 0.17)),
            ("roll +/-80", (1.40, 0.0, 0.0), (-1.40, 0.0, 0.0)),
            ("roll +/-100", (1.75, 0.0, 0.0), (-1.75, 0.0, 0.0)),
            ("pitch +10", (math.pi / 2, 0.17, 0.0), (-math.pi / 2, 0.17, 0.0)),
            ("pitch -10", (math.pi / 2, -0.17, 0.0), (-math.pi / 2, -0.17, 0.0)),
        ]
        results = []
        zeros = np.zeros(14, dtype=float)
        for label, left_rpy, right_rpy in candidates:
            left = self.make_transform(
                self.left_neutral_position,
                self.rpy_to_rotation_matrix(*left_rpy),
            )
            right = self.make_transform(
                self.right_neutral_position,
                self.rpy_to_rotation_matrix(*right_rpy),
            )
            try:
                sol_q, _ = self.arm_ik.solve_ik(left, right, zeros, zeros)
            except Exception as exc:
                self.get_logger().warning(f"candidate {label}: IK failed: {exc}")
                continue

            sol_q = np.asarray(sol_q, dtype=float)
            if sol_q.shape[0] < 14 or not np.all(np.isfinite(sol_q[:14])):
                self.get_logger().warning(f"candidate {label}: non-finite/bad q")
                continue

            clipped_left = self.clip_active_q_to_ik_limits("left", sol_q[:7])
            clipped_right = self.clip_active_q_to_ik_limits("right", sol_q[7:14])
            clip_error = float(
                np.linalg.norm(clipped_left - sol_q[:7])
                + np.linalg.norm(clipped_right - sol_q[7:14])
            )
            shoulder_elbow = np.r_[sol_q[0:4], sol_q[7:11]]
            if clip_error > 1e-5 or np.any(np.abs(shoulder_elbow) > 2.6):
                self.get_logger().warning(
                    f"candidate {label}: rejected clip_error={clip_error:.4f}, "
                    f"shoulder/elbow={np.array2string(shoulder_elbow, precision=3)}"
                )
                continue

            score = float(np.linalg.norm(sol_q[:14]))
            results.append(
                {
                    "label": label,
                    "score": score,
                    "q": sol_q[:14].copy(),
                    "left_target": left,
                    "right_target": right,
                }
            )

        results.sort(key=lambda item: item["score"])
        for index, result in enumerate(results[:8]):
            self.get_logger().info(
                f"candidate {index}: {result['label']} score={result['score']:.3f}, "
                f"q={np.array2string(result['q'], precision=3)}"
            )
        if not results:
            self.get_logger().error("candidate_search found no valid IK candidates.")
        return results

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


def main(args=None, default_arm_side="both"):
    rclpy.init(args=args)
    node = None
    try:
        node = LiveArmMapper(default_arm_side=default_arm_side)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def both_arm_main(args=None):
    main(args=args, default_arm_side="both")


if __name__ == "__main__":
    main()


RightArmLiveMapper = LiveArmMapper
