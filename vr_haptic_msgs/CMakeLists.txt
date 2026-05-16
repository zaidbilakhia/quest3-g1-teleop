cmake_minimum_required(VERSION 3.16)
project(vr_haptic_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HandGesture.msg"
  "msg/HapticReadings.msg"
  "msg/ManoLandmarks.msg"
  "msg/Taxel.msg"
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
