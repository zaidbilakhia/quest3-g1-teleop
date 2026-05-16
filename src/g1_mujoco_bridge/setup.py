from setuptools import setup

package_name = 'g1_mujoco_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zaid',
    maintainer_email='zaid@example.com',
    description='ROS 2 bridge between MuJoCo and Unitree G1',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'g1_bridge = g1_mujoco_bridge.g1_ros2_bridge:main',
            'g1_ros2_bridge = g1_mujoco_bridge.g1_ros2_bridge:main',
        ],
    },
)
