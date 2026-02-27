#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2
import math
import rclpy
import threading
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from dt_apriltags import Detector
import tf2_ros
from tf_transformations import (
    quaternion_from_matrix,
    rotation_matrix,
)
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Constants
APRIL_TAG_LOOKUP_SUB_PATH = "test_data/april_tag_lookup.yaml"
ROBOT_TAG_COLOR = (172, 16, 48)
MAP_TAG_COLOR = (255, 165, 0)
OBSTACLE_TAG_COLOR = (0, 0, 255)
TAG_GRACE_PERIOD = 0.5

class ApriltagNode(Node):
    def __init__(self):
        super().__init__('apriltag_node')
        self.get_logger().info("Starting AprilTag Node with Distortion Correction")

        # === Tools ===
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # === Orientation Fix: DIFFERENT for USB vs IP ===
        # USB webcam is upside-down → needs 180° flip around X
        self.R_flip_usb = rotation_matrix(math.pi, [1.0, 0.0, 0.0])[:3, :3]

        # IP camera has correct orientation → NO FLIP
        self.R_flip_ip = np.eye(3)

        # === State ===
        self.tag_pose_cache = {}
        self.tag_last_seen = {}
        self.smoothing_alpha = 0.2
        self.last_visible_tags_cam1 = set()
        self.last_visible_tags_cam2 = set()

        # --- Anti-Latency Frame Handling ---
        self.latest_frame_cam1 = None
        self.latest_frame_cam2 = None
        self.lock_cam1 = threading.Lock()
        self.lock_cam2 = threading.Lock()
        
        # === Load Parameters and Data ===
        self._declare_and_get_params()
        self.load_in_tag_data()

        # === AprilTag Detector ===
        self.detector = self._initialize_detector()

        # === QoS Profile for low-latency image handling ===
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # === Subscriptions (only store frames) ===
        self.subscription_cam1 = self.create_subscription(
            Image, self.usb_camera_topic, self.image_callback_cam1, qos)
        self.subscription_cam2 = self.create_subscription(
            Image, self.ip_camera_topic, self.image_callback_cam2, qos)
        
        # --- Dedicated Processing Timer ---
        # Runs at 15Hz to process the latest available frame
        processing_rate = 15.0
        self.create_timer(1.0 / processing_rate, self.process_latest_frames)

        self.get_logger().info("AprilTag node has started successfully.")

    def load_in_tag_data(self):
        try:
            # Adjust package name 'amr_central' if your lookup file is elsewhere
            config_path = os.path.join(get_package_share_directory("amr_central"), APRIL_TAG_LOOKUP_SUB_PATH)
            with open(config_path) as config_file:
                self.tag_lookup = yaml.safe_load(config_file)
                self.registered_tags = self.tag_lookup["registered_tags"]
                self.robot_tags = self.tag_lookup["robot_tags"]
                self.map_tags = self.tag_lookup["map_tags"]
                self.obstacle_tags = self.tag_lookup["obstacle_tags"]
        except Exception as e:
            self.get_logger().error(f"Failed to load tag lookup file: {e}. Using empty lists.")
            self.registered_tags = []
            self.robot_tags = []
            self.map_tags = []
            self.obstacle_tags = []

    def _declare_and_get_params(self):
        self.declare_parameter('usb_camera_topic', '/camera/image_raw')
        self.declare_parameter('ip_camera_topic', '/ip_camera/image_raw')
        self.declare_parameter('tag_size', 0.0796)
        self.declare_parameter('show_processed_video', True)
        
        # USB Camera Intrinsics (Default)
        self.declare_parameter('usb_cam.fx', 921.62)
        self.declare_parameter('usb_cam.fy', 923.23)
        self.declare_parameter('usb_cam.cx', 614.94)
        self.declare_parameter('usb_cam.cy', 360.22)
        
        # IP Camera Intrinsics (Default)
        self.declare_parameter('ip_cam.fx', 609.75)
        self.declare_parameter('ip_cam.fy', 609.75)
        self.declare_parameter('ip_cam.cx', 597.88)
        self.declare_parameter('ip_cam.cy', 324.10)
        
        # IP Camera Distortion Coefficients
        self.declare_parameter('ip_cam.k1', 0.0)
        self.declare_parameter('ip_cam.k2', 0.0)
        self.declare_parameter('ip_cam.p1', 0.0)
        self.declare_parameter('ip_cam.p2', 0.0)
        self.declare_parameter('ip_cam.k3', 0.0)

        self.refresh_parameters()
        self.add_on_set_parameters_callback(self.parameter_callback)

    def refresh_parameters(self):
        self.usb_camera_topic = self.get_parameter('usb_camera_topic').get_parameter_value().string_value
        self.ip_camera_topic = self.get_parameter('ip_camera_topic').get_parameter_value().string_value
        self.tag_size = self.get_parameter('tag_size').get_parameter_value().double_value
        
        self.usb_camera_params = [
            self.get_parameter('usb_cam.fx').get_parameter_value().double_value,
            self.get_parameter('usb_cam.fy').get_parameter_value().double_value,
            self.get_parameter('usb_cam.cx').get_parameter_value().double_value,
            self.get_parameter('usb_cam.cy').get_parameter_value().double_value,
        ]
        
        # IP Camera Params
        fx = self.get_parameter('ip_cam.fx').get_parameter_value().double_value
        fy = self.get_parameter('ip_cam.fy').get_parameter_value().double_value
        cx = self.get_parameter('ip_cam.cx').get_parameter_value().double_value
        cy = self.get_parameter('ip_cam.cy').get_parameter_value().double_value
        
        self.ip_camera_params = [fx, fy, cx, cy]
        
        # Build Matrix (K) and Distortion Vector (D) for IP Cam
        self.K_ip = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.D_ip = np.array([
            self.get_parameter('ip_cam.k1').get_parameter_value().double_value,
            self.get_parameter('ip_cam.k2').get_parameter_value().double_value,
            self.get_parameter('ip_cam.p1').get_parameter_value().double_value,
            self.get_parameter('ip_cam.p2').get_parameter_value().double_value,
            self.get_parameter('ip_cam.k3').get_parameter_value().double_value
        ])

    def _initialize_detector(self):
        try:
            return Detector(families='tag36h11',
                            quad_decimate=1.0,
                            nthreads=4,
                            quad_sigma=0.0,
                            refine_edges=1)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize detector: {e}")
            rclpy.shutdown()

    def image_callback_cam1(self, msg: Image):
        with self.lock_cam1:
            self.latest_frame_cam1 = msg

    def image_callback_cam2(self, msg: Image):
        with self.lock_cam2:
            self.latest_frame_cam2 = msg

    def process_latest_frames(self):
        frame_cam1, frame_cam2 = None, None
        with self.lock_cam1:
            if self.latest_frame_cam1:
                frame_cam1 = self.latest_frame_cam1
                self.latest_frame_cam1 = None
        with self.lock_cam2:
            if self.latest_frame_cam2:
                frame_cam2 = self.latest_frame_cam2
                self.latest_frame_cam2 = None

        if frame_cam1:
            # USB Camera (No distortion fix needed usually)
            self.process_image(frame_cam1, "cam1_usb", self.last_visible_tags_cam1, self.usb_camera_params, "usb_camera_link", do_undistort=False)
        if frame_cam2:
            # IP Camera (Needs Distortion Correction!)
            self.process_image(frame_cam2, "cam2_ip", self.last_visible_tags_cam2, self.ip_camera_params, "ip_camera_link", do_undistort=True)

    def process_image(self, msg: Image, window_name: str, last_visible_tags: set, camera_params: list, cam_frame: str, do_undistort: bool):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Default to using original params
            current_camera_params = camera_params 

            # --- CRITICAL FIX: Undistort and Update Matrix ---
            if do_undistort:
                h, w = image.shape[:2]
                
                # 1. Get the NEW Optimal Camera Matrix
                # alpha=0: Zoom in to valid pixels only (fixes black edges)
                new_K, roi = cv2.getOptimalNewCameraMatrix(self.K_ip, self.D_ip, (w, h), 0, (w, h))
                
                # 2. Undistort using the NEW matrix
                image = cv2.undistort(image, self.K_ip, self.D_ip, None, new_K)
                
                # 3. Extract the NEW fx, fy, cx, cy for the AprilTag Detector
                # The detector needs these new values to know where pixels moved to!
                new_fx = new_K[0, 0]
                new_fy = new_K[1, 1]
                new_cx = new_K[0, 2]
                new_cy = new_K[1, 2]
                current_camera_params = [new_fx, new_fy, new_cx, new_cy]
            # -------------------------------------------------
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"Image conversion error for {window_name}: {e}")
            return

        current_visible_tags = set()
        image_timestamp = msg.header.stamp

        try:
            # Detect tags using the CORRECT params (either original or new)
            tags = self.detector.detect(gray, True, current_camera_params, self.tag_size)
            
            for tag in tags:
                current_visible_tags.add(tag.tag_id)
                self._process_tag(tag, image, cam_frame, image_timestamp)
        except Exception as e:
            self.get_logger().error(f"Detection error for {window_name}: {e}")

        disappeared_tags = last_visible_tags - current_visible_tags
        now = self.get_clock().now().nanoseconds / 1e9
        for tag_id in disappeared_tags:
            last_seen = self.tag_last_seen.get(tag_id, None)
            if last_seen is None or (now - last_seen) > TAG_GRACE_PERIOD:
                self._publish_stale_tf(cam_frame, tag_id)

        last_visible_tags.clear()
        last_visible_tags.update(current_visible_tags)

        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            cv2.imshow(window_name, image)
            cv2.waitKey(1)

    def _process_tag(self, tag, image, cam_frame, stamp):
        if tag.pose_R is None or tag.pose_t is None or tag.tag_id not in self.registered_tags:
            return

        ### === CAMERA-SPECIFIC ORIENTATION FIX ===
        if cam_frame == "usb_camera_link":
            R = self.R_flip_usb @ tag.pose_R
            t = self.R_flip_usb @ tag.pose_t.reshape(3, 1)
        else:
            R = self.R_flip_ip @ tag.pose_R
            t = self.R_flip_ip @ tag.pose_t.reshape(3, 1)

        tag_key = f"tag_{tag.tag_id}"

        smoothed_t = self.smoothing_alpha * t + (1 - self.smoothing_alpha) * self.tag_pose_cache.get(tag_key, t)
        self.tag_pose_cache[tag_key] = smoothed_t
        self.tag_last_seen[tag.tag_id] = self.get_clock().now().nanoseconds / 1e9

        self._publish_tf(cam_frame, f"tag_{tag.tag_id}", R, smoothed_t, stamp)
        if tag.tag_id in self.obstacle_tags:
            self._publish_tf(cam_frame, f"tag_{tag.tag_id}", R, smoothed_t, stamp)

        color = (0, 255, 0)
        if tag.tag_id in self.map_tags: color = MAP_TAG_COLOR
        elif tag.tag_id in self.robot_tags: color = ROBOT_TAG_COLOR
        elif tag.tag_id in self.obstacle_tags: color = OBSTACLE_TAG_COLOR
        self._draw_tag(image, tag, color)

    def _publish_tf(self, parent_frame, child_frame, R, t, stamp):
        try:
            T = np.identity(4)
            T[:3, :3] = R
            quat = quaternion_from_matrix(T)
            t_msg = TransformStamped()
            t_msg.header.stamp  = self.get_clock().now().to_msg() 
            t_msg.header.frame_id = parent_frame
            t_msg.child_frame_id = child_frame
            t_msg.transform.translation.x, t_msg.transform.translation.y, t_msg.transform.translation.z = t.flatten()
            t_msg.transform.rotation.x, t_msg.transform.rotation.y, t_msg.transform.rotation.z, t_msg.transform.rotation.w = quat
            self.tf_broadcaster.sendTransform(t_msg)
        except Exception as e:
            self.get_logger().error(f"TF publish error {child_frame}: {e}")

    def _publish_stale_tf(self, parent_frame, tag_id):
        try:
            t_msg = TransformStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = parent_frame
            t_msg.child_frame_id = f"tag_{tag_id}"
            t_msg.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish stale TF for tag {tag_id}: {e}")

    def _draw_tag(self, frame, tag, draw_color_bgr):
        try:
            corners = tag.corners.astype(int)
            center = tuple(tag.center.astype(int))
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), draw_color_bgr, 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {tag.tag_id}", (corners[0][0], corners[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color_bgr, 2)
        except Exception as e:
            self.get_logger().error(f"Draw error for tag {tag.tag_id}: {e}")

    def parameter_callback(self, params):
        for param in params:
            if param.name == "show_processed_video":
                self.get_logger().info(f"Live display toggled to: {param.value}")
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ApriltagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()