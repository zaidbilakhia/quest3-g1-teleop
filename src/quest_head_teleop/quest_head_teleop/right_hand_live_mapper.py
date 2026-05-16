#!/usr/bin/env python3
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from vr_haptic_msgs.msg import ManoLandmarks


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def dist_points(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def dist_xyz(ax, ay, az, bx, by, bz):
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def finger_curl(msg, base_idx, j1_idx, j2_idx, tip_idx):
    """
    0.0 -> finger mostly open
    1.0 -> finger more curled
    """
    lm = msg.landmarks

    base = lm[base_idx]
    j1 = lm[j1_idx]
    j2 = lm[j2_idx]
    tip = lm[tip_idx]

    chain_len = dist_points(base, j1) + dist_points(j1, j2) + dist_points(j2, tip)
    straight_len = dist_points(base, tip)

    if chain_len < 1e-6:
        return 0.0

    openness = straight_len / chain_len
    openness = clamp(openness, 0.0, 1.0)
    curl = 1.0 - openness
    return clamp(curl, 0.0, 1.0)


def normalize_between(raw_value, open_value, closed_value):
    """
    Works for both increasing and decreasing features.
    Returns 0.0 at open_value and 1.0 at closed_value.
    """
    denom = closed_value - open_value
    if abs(denom) < 1e-6:
        return 0.0
    value = (raw_value - open_value) / denom
    return clamp(value, 0.0, 1.0)


def average_feature_dicts(samples):
    if not samples:
        return None

    keys = samples[0].keys()
    out = {}
    for k in keys:
        out[k] = sum(s[k] for s in samples) / float(len(samples))
    return out


class RightHandMapper(Node):
    def __init__(self):
        super().__init__('right_hand_mapper')

        self.declare_parameter('mode', 'calibrate')  # calibrate or live
        self.declare_parameter(
            'calibration_path',
            str(Path.home() / 'humanoid_ws' / 'right_hand_calibration.json')
        )
        self.declare_parameter('speak', True)
        self.declare_parameter('announce_delay_sec', 5.0)
        self.declare_parameter('capture_sec', 3.0)
        self.declare_parameter('auto_switch_to_live', True)

        self.mode = self.get_parameter('mode').value
        self.calibration_path = Path(self.get_parameter('calibration_path').value)
        self.use_speech = bool(self.get_parameter('speak').value)
        self.announce_delay_sec = float(self.get_parameter('announce_delay_sec').value)
        self.capture_sec = float(self.get_parameter('capture_sec').value)
        self.auto_switch_to_live = bool(self.get_parameter('auto_switch_to_live').value)

        self.sub = self.create_subscription(
            ManoLandmarks,
            '/quest/hand_pose',
            self.hand_callback,
            10
        )

        self.pub = self.create_publisher(JointState, '/g1/right_hand_cmd', 10)

        self.latest_features = None
        self.msg_count = 0
        self.calib = None

        # Low-pass filter for live motion
        self.alpha = 0.25
        self.have_filter_state = False
        self.filtered = {
            'thumb_opp': 0.0,
            'thumb_pinch': 0.0,
            'index': 0.0,
            'middle': 0.0,
        }

        # If thumb base moves the wrong way, change to -1.0
        self.thumb0_sign = 1.0

        self.speaker_cmd = self.detect_speaker()

        self.get_logger().info('Right hand mapper started.')
        self.get_logger().info(f'Mode: {self.mode}')
        self.get_logger().info('Subscribing to /quest/hand_pose')
        self.get_logger().info('Publishing to /g1/right_hand_cmd')
        self.get_logger().info(f'Calibration file: {self.calibration_path}')

        # Very simple spoken routine
        self.stage_defs = [
            {
                'name': 'neutral',
                'prompt': 'Step 1. Relax your right hand.',
            },
            {
                'name': 'open',
                'prompt': 'Step 2. Open your right hand.',
            },
            {
                'name': 'fist',
                'prompt': 'Step 3. Make a fist.',
            },
            {
                'name': 'pinch',
                'prompt': 'Step 4. Touch thumb and index.',
            },
            {
                'name': 'tripod',
                'prompt': 'Step 5. Make tripod grasp. Thumb, index, middle.',
            },
        ]

        self.stage_index = 0
        self.phase = 'idle'
        self.phase_end_time = 0.0
        self.capture_samples = []
        self.calib_data = {}

        self.timer = self.create_timer(0.1, self.timer_callback)

        if self.mode == 'live':
            if not self.load_calibration():
                self.speak('Calibration file not found. Starting calibration.')
                self.mode = 'calibrate'
                self.start_calibration()
            else:
                self.speak('Right hand live control started.')
        else:
            self.start_calibration()

    def detect_speaker(self):
        if not self.use_speech:
            return None

        if shutil.which('spd-say'):
            return 'spd-say'
        if shutil.which('espeak-ng'):
            return 'espeak-ng'
        if shutil.which('espeak'):
            return 'espeak'
        return None

    def speak(self, text):
        self.get_logger().info(text)

        if not self.use_speech or self.speaker_cmd is None:
            return

        try:
            if self.speaker_cmd == 'spd-say':
                subprocess.Popen(['spd-say', text])
            else:
                subprocess.Popen([self.speaker_cmd, text])
        except Exception as e:
            self.get_logger().warn(f'Voice output failed: {e}')

    def start_calibration(self):
        self.mode = 'calibrate'
        self.stage_index = 0
        self.phase = 'announce'
        self.phase_end_time = 0.0
        self.capture_samples = []
        self.calib_data = {}
        self.have_filter_state = False
        self.speak('Right hand calibration started.')

    def save_calibration(self):
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': 1,
            'created_unix_sec': time.time(),
            'stages': self.calib_data,
        }

        with open(self.calibration_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        self.get_logger().info(f'Saved calibration to {self.calibration_path}')

    def load_calibration(self):
        if not self.calibration_path.exists():
            self.get_logger().warn(f'Calibration file not found: {self.calibration_path}')
            return False

        try:
            with open(self.calibration_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            self.calib = payload['stages']

            required = ['neutral', 'open', 'fist', 'pinch', 'tripod']
            for key in required:
                if key not in self.calib:
                    raise RuntimeError(f'Missing calibration stage: {key}')

            self.get_logger().info('Calibration loaded successfully.')
            return True

        except Exception as e:
            self.get_logger().error(f'Failed to load calibration: {e}')
            return False

    def extract_features(self, msg):
        """
        Features used for the 3-finger robot hand:
        - index curl
        - middle curl
        - thumb pinch distance
        - thumb opposition distance
        """
        if len(msg.landmarks) < 21:
            return None

        lm = msg.landmarks

        index_curl = finger_curl(msg, 5, 6, 7, 8)
        middle_curl = finger_curl(msg, 9, 10, 11, 12)

        thumb_tip = lm[4]
        index_tip = lm[8]
        thumb_pinch_dist = dist_points(thumb_tip, index_tip)

        index_base = lm[5]
        middle_base = lm[9]
        palm_mid_x = 0.5 * (index_base.x + middle_base.x)
        palm_mid_y = 0.5 * (index_base.y + middle_base.y)
        palm_mid_z = 0.5 * (index_base.z + middle_base.z)

        thumb_opposition_dist = dist_xyz(
            thumb_tip.x, thumb_tip.y, thumb_tip.z,
            palm_mid_x, palm_mid_y, palm_mid_z
        )

        return {
            'index_curl': index_curl,
            'middle_curl': middle_curl,
            'thumb_pinch_dist': thumb_pinch_dist,
            'thumb_opposition_dist': thumb_opposition_dist,
        }

    def low_pass(self, key, new_value):
        if not self.have_filter_state:
            self.filtered[key] = new_value
            return new_value

        value = (1.0 - self.alpha) * self.filtered[key] + self.alpha * new_value
        self.filtered[key] = value
        return value

    def publish_robot_hand(self, thumb_opp, thumb_pinch, index_close, middle_close):
        thumb_opp = self.low_pass('thumb_opp', thumb_opp)
        thumb_pinch = self.low_pass('thumb_pinch', thumb_pinch)
        index_close = self.low_pass('index', index_close)
        middle_close = self.low_pass('middle', middle_close)
        self.have_filter_state = True

        # Robot hand mapping
        thumb_0 = self.thumb0_sign * (0.15 + 0.75 * thumb_opp)
        thumb_1 = -0.10 - 0.40 * thumb_pinch
        thumb_2 = -0.15 - 1.55 * thumb_pinch

        index_0 = 0.10 + 1.35 * index_close
        index_1 = 0.05 + 1.70 * index_close

        middle_0 = 0.10 + 1.35 * middle_close
        middle_1 = 0.05 + 1.70 * middle_close

        # Safe clamps
        thumb_0 = clamp(thumb_0, -1.0, 1.0)
        thumb_1 = clamp(thumb_1, -1.0, 0.7)
        thumb_2 = clamp(thumb_2, -2.0, 0.0)

        index_0 = clamp(index_0, -0.19, 1.83)
        index_1 = clamp(index_1, 0.0, 2.09)

        middle_0 = clamp(middle_0, -0.19, 1.83)
        middle_1 = clamp(middle_1, 0.0, 2.09)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [
            'right_hand_thumb_0_joint',
            'right_hand_thumb_1_joint',
            'right_hand_thumb_2_joint',
            'right_hand_index_0_joint',
            'right_hand_index_1_joint',
            'right_hand_middle_0_joint',
            'right_hand_middle_1_joint',
        ]
        js.position = [
            thumb_0,
            thumb_1,
            thumb_2,
            index_0,
            index_1,
            middle_0,
            middle_1,
        ]
        self.pub.publish(js)

    def run_live_mapping(self, features):
        if self.calib is None:
            return

        index_close = normalize_between(
            features['index_curl'],
            self.calib['open']['index_curl'],
            self.calib['fist']['index_curl']
        )

        middle_close = normalize_between(
            features['middle_curl'],
            self.calib['open']['middle_curl'],
            self.calib['fist']['middle_curl']
        )

        thumb_pinch = normalize_between(
            features['thumb_pinch_dist'],
            self.calib['open']['thumb_pinch_dist'],
            self.calib['pinch']['thumb_pinch_dist']
        )

        thumb_opp = normalize_between(
            features['thumb_opposition_dist'],
            self.calib['open']['thumb_opposition_dist'],
            self.calib['tripod']['thumb_opposition_dist']
        )

        self.publish_robot_hand(
            thumb_opp=thumb_opp,
            thumb_pinch=thumb_pinch,
            index_close=index_close,
            middle_close=middle_close
        )

        self.msg_count += 1
        if self.msg_count % 20 == 0:
            self.get_logger().info(
                f'LIVE | thumb_opp={thumb_opp:.3f}, thumb_pinch={thumb_pinch:.3f}, '
                f'index={index_close:.3f}, middle={middle_close:.3f}'
            )

    def hand_callback(self, msg):
        features = self.extract_features(msg)
        if features is None:
            return

        self.latest_features = features

        if self.mode == 'calibrate':
            if self.phase == 'capture':
                self.capture_samples.append(features.copy())

        elif self.mode == 'live':
            self.run_live_mapping(features)

    def timer_callback(self):
        if self.mode != 'calibrate':
            return

        now = time.monotonic()

        if self.stage_index >= len(self.stage_defs):
            return

        stage = self.stage_defs[self.stage_index]

        if self.phase == 'announce':
            self.speak(stage['prompt'])
            self.speak(f'Get ready. Capture starts in {int(self.announce_delay_sec)} seconds.')
            self.phase = 'wait'
            self.phase_end_time = now + self.announce_delay_sec
            return

        if self.phase == 'wait':
            if now >= self.phase_end_time:
                if self.latest_features is None:
                    self.speak('Hand not visible. Repeat this step.')
                    self.phase = 'announce'
                    return

                self.capture_samples = []
                self.speak('Capture now. Hold still.')
                self.phase = 'capture'
                self.phase_end_time = now + self.capture_sec
            return

        if self.phase == 'capture':
            if now >= self.phase_end_time:
                if len(self.capture_samples) < 10:
                    self.speak('Not enough samples. Repeat this step.')
                    self.phase = 'announce'
                    return

                averaged = average_feature_dicts(self.capture_samples)
                self.calib_data[stage['name']] = averaged
                self.speak(f'{stage["name"]} captured.')

                self.get_logger().info(
                    f'Captured {stage["name"]}: {json.dumps(averaged, indent=2)}'
                )

                self.stage_index += 1
                if self.stage_index < len(self.stage_defs):
                    self.phase = 'announce'
                else:
                    self.save_calibration()
                    self.speak('Calibration complete.')

                    if self.auto_switch_to_live:
                        if self.load_calibration():
                            self.mode = 'live'
                            self.have_filter_state = False
                            self.speak('Live control enabled.')
                        else:
                            self.speak('Calibration saved, but loading failed.')
                            self.phase = 'done'
                    else:
                        self.phase = 'done'
                        self.speak('Calibration saved. Restart in live mode.')
            return


def main(args=None):
    rclpy.init(args=args)
    node = RightHandMapper()
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