# amr_apriltag — Vision & Obstacle Detection

`amr_apriltag` is the perception layer of the OSU-AMR tiered architecture. It is responsible for detecting physical objects and obstacles in the operating environment using AprilTags.

This package captures video feeds (via local USB or network IP cameras), detects standard AprilTag markers, and publishes their calculated positions as ROS 2 tf2 transforms. These transforms are then consumed by `amr_central/course_manager` to dynamically block or unblock routes on the map if an obstacle is placed in the robots' path.

---

## 🏗️ System Role

```
[Camera Feed (USB/IP)] 
        ↓ 
[webcam_node / ip_webcam_node] → (Raw Images)
        ↓
[apriltag_node] → (Calculates Pose)
        ↓
[TF2 Transforms (/tf)]
        ↓
[amr_central / Course Manager] → (Determines Blocked Routes)
```

---

## 📁 File Reference

| File | What it does |
|---|---|
| `apriltag_node.py` | The core processing node. Subscribes to raw image feeds, detects AprilTags, calculates their 3D pose, and broadcasts them to the ROS 2 `/tf` tree. |
| `webcam_node.py` | Captures images from a locally connected USB camera and publishes them as ROS 2 Image messages. |
| `ip_webcam_node.py` | Captures an image stream from a networked IP camera (used for overhead lab cameras) and publishes them. |
| `manual_calibrator.py` | Utility script to determine intrinsic calibration parameters (camera matrix, distortion coefficients) for accurate 3D pose estimation. |
| `detection.launch.py` | Standard launch file. Starts camera nodes, detection node, and RViz for visual debugging. |
| `detection_headless.launch.py` | Starts camera and detection nodes without a graphical interface. For background/server operation. |
| `apriltag_rviz.rviz` | Pre-configured RViz layout for monitoring camera feeds, tag overlays, and TF frames. |

---

## 📁 Directory Structure

```
amr_apriltag/
├── apriltag_detection/
│   ├── apriltag_detection/
│   │   ├── apriltag_node.py           # Core detection and TF broadcaster
│   │   ├── webcam_node.py             # USB camera driver
│   │   ├── ip_webcam_node.py          # IP camera driver
│   │   └── manual_calibrator.py       # Lens calibration utility
│   ├── launch/
│   │   ├── detection.launch.py        # Launch with RViz GUI
│   │   └── detection_headless.launch.py  # Launch for background execution
│   ├── rviz/
│   │   └── apriltag_rviz.rviz         # Pre-configured visualizer layout
│   ├── test/                          # Unit and linting tests
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── setup.cfg
│   └── setup.py
└── test/
```

---

## 🚀 How to Run

### 1. Standard Mode (With Visualization)

If you want to view the camera feed and verify that tags are being detected and mapped correctly in 3D space:

```bash
source /opt/ros/humble/setup.bash
source ~/AMR/install/setup.bash
ros2 launch apriltag_detection detection.launch.py
```

### 2. Headless Mode (Lab Operation)

If you are running the full fleet and don't want to spend CPU resources rendering a graphical interface:

```bash
source /opt/ros/humble/setup.bash
source ~/AMR/install/setup.bash
ros2 launch apriltag_detection detection_headless.launch.py
```

---

## ⚙️ Calibration & Configuration

> ⚠️ **IMPORTANT: Camera Calibration**
> AprilTag detection relies on knowing the physical distortion properties of the camera lens. If you change the camera hardware or adjust the focus/zoom of an overhead lab camera, you **must** recalibrate using `manual_calibrator.py`. Using incorrect calibration matrices will result in skewed TF transforms, causing the `course_manager` to block the wrong routes.

---

## 🔗 Related Repositories

| Repo | Role |
|---|---|
| `amr_central` | Consumes TF data generated here to detect blocked paths |
| `amr_msgs` | Provides standard message definitions for the architecture |
| `amr_docs` | Full system documentation and lab manual |
