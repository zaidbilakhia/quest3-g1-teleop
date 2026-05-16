from setuptools import find_packages, setup

package_name = 'quest_head_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zaid',
    maintainer_email='zaid@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'head_pose_mapper = quest_head_teleop.head_pose_mapper:main',
            'hand_feature_debug = quest_head_teleop.hand_feature_debug:main',
            'right_hand_live_mapper = quest_head_teleop.right_hand_live_mapper:main',
            'right_arm_live_mapper = quest_head_teleop.both_arm_live_mapper:main',
            (
                'hand_axis_calibration_recorder = '
                'quest_head_teleop.hand_axis_calibration_recorder:main'
            ),
            (
                'hand_calibration_csv_replay = '
                'quest_head_teleop.hand_calibration_csv_replay:main'
            ),
            (
                'wrist_orientation_calibration_recorder = '
                'quest_head_teleop.wrist_orientation_calibration_recorder:main'
            ),
            (
                'combined_hand_wrist_calibration_recorder = '
                'quest_head_teleop.combined_hand_wrist_calibration_recorder:main'
            ),
            (
                'both_arm_live_mapper = '
                'quest_head_teleop.both_arm_live_mapper:both_arm_main'
            ),
        ],
    },
)
