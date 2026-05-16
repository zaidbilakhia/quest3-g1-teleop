# VR Haptic Messages

ROS 2 Jazzy message package for haptic data in VR workflows.

## Messages
- `Taxel.msg`: Represents a single taxel (tactile pixel) with an intensity and force
- `HapticReadings.msg`: Contains an array of `Taxel` messages to represent points on the hand
- `ManoLandmarks.msg`: Represents hand landmarks (imported from [Mano](https://mano.is.tue.mpg.de/))
- `HandGesture.msg`: Represents hand gestures (imported from [Mano](https://mano.is.tue.mpg.de/))

## Build (ROS 2 Jazzy)

From a sourced ROS 2 workspace:

```bash
colcon build --packages-select vr_haptic_msgs
```