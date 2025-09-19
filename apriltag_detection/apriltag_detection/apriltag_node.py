#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from dt_apriltags import Detector
import tf2_ros
from tf_transformations import quaternion_from_matrix, rotation_matrix, quaternion_multiply, quaternion_from_euler, quaternion_matrix
from ament_index_python.packages import get_package_share_directory

# Constants
APRIL_TAG_LOOKUP_SUB_PATH = "test_data/april_tag_lookup.yaml"
ROBOT_TAG_COLOR = (172, 16, 48)
MAP_TAG_COLOR = (255, 165, 0)
OBSTACLE_TAG_COLOR = (0, 39, 76)
FRAME_STALE_TIME = 5  # seconds
FRAME_DEFAULT_VECTOR = [0, 0, -1]


# Helper class for low-pass filtering
class low_pass_filtered_value():
    def __init__(self, starting_value, cutoff_frequency, init_time):
        self.value = starting_value
        self.cutoff_frequency = cutoff_frequency
        self.previous_time = init_time

    def update(self, measurement, time):
        dt = time - self.previous_time
        if dt <= 0:
            return self.value
        alpha = (2 * math.pi * dt * self.cutoff_frequency) / (2 * math.pi * dt * self.cutoff_frequency + 1)
        self.value = alpha * measurement + (1.0 - alpha) * self.value
        self.previous_time = time
        return self.value

    def set_cutoff_frequency(self, frequency):
        self.cutoff_frequency = frequency


class ApriltagNode(Node):
    def __init__(self):
        super().__init__('apriltag_node')
        self.get_logger().info("Starting AprilTag Node with dual camera support")
<<<<<<< Updated upstream

        # === Parameters ===
        self._declare_and_get_params()

        #load in data
        self.load_in_tag_data()

        # === AprilTag Detector ===
        self.detector = self._initialize_detector()
=======
>>>>>>> Stashed changes

        # === Tools ===
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

# Choose ONE parent frame for the map tree (USB is fine)
        self.map_parent_frame = 'usb_camera_link'  # make this a ROS param later if you want

        self.R_flip = rotation_matrix(math.pi, [1.0, 0.0, 0.0])[:3, :3]

        # === State ===
        self.tag_pose_cache = {}
        self.smoothing_alpha = 0.2
        self.last_visible_tags_cam1 = set()
        self.last_visible_tags_cam2 = set()
<<<<<<< Updated upstream

        self.frame_tags = dict()

        # === Subscriptions ===
        # Use the reliably fetched parameters
        self.get_logger().info(f"Subscribing to USB camera at '{self.usb_camera_topic}'")
        self.subscription_cam1 = self.create_subscription(
            Image,
            self.usb_camera_topic,
            self.image_callback_cam1,
            10
        )
        
        self.get_logger().info(f"Subscribing to IP camera at '{self.ip_camera_topic}'")
        self.subscription_cam2 = self.create_subscription(
            Image,
            self.ip_camera_topic,
            self.image_callback_cam2,
            10
        )

        #init low pass filter
        self.map_x_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, self.get_time_in_seconds())
        self.map_y_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, self.get_time_in_seconds())
        self.map_z_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, self.get_time_in_seconds())
        self.map_qx_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, self.get_time_in_seconds())
        self.map_qy_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, self.get_time_in_seconds())
        self.map_qz_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, self.get_time_in_seconds())
        self.map_qw_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, self.get_time_in_seconds())


        #timer for param refresh
        self.create_timer(5.0, self.refresh_parameters)

        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            self.get_logger().info("Video output enabled.")
=======
        self.frame_tags = {}

        # === Load Parameters and Data ===
        self._declare_and_get_params()
        self.load_in_tag_data()

        # === AprilTag Detector ===
        self.detector = self._initialize_detector()

        # === Subscriptions ===
        self.subscription_cam1 = self.create_subscription(Image, self.usb_camera_topic, self.image_callback_cam1, 10)
        self.subscription_cam2 = self.create_subscription(Image, self.ip_camera_topic, self.image_callback_cam2, 10)

        # === Low-pass filters ===
        now = self.get_time_in_seconds()
        self.map_x_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, now)
        self.map_y_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, now)
        self.map_z_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, now)
        self.map_qx_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, now)
        self.map_qy_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, now)
        self.map_qz_filter = low_pass_filtered_value(0, self.map_filtering_lowpass_threshold, now)
        self.map_qw_filter = low_pass_filtered_value(1, self.map_filtering_lowpass_threshold, now)

        # === Timer ===
        self.create_timer(5.0, self.refresh_parameters)
        self.get_logger().info("AprilTag node has started successfully.")
>>>>>>> Stashed changes

    def load_in_tag_data(self):
        try:
            config_path = os.path.join(get_package_share_directory("amr_central"), APRIL_TAG_LOOKUP_SUB_PATH)
            with open(config_path) as config_file:
                self.tag_lookup = yaml.safe_load(config_file)
                self.registered_tags = self.tag_lookup["registered_tags"]
                self.robot_tags = self.tag_lookup["robot_tags"]
                self.map_tags = self.tag_lookup["map_tags"]
                self.obstacle_tags = self.tag_lookup["obstacle_tags"]
<<<<<<< Updated upstream
                self.corner_tag_distance = self.tag_lookup["corner_tag_distance"]

        except:
            self.get_logger().error("Failed to load tag lookup file. Quitting!")
            quit()

    def _declare_and_get_params(self):
        # Declare all parameters with their default values first
=======
        except Exception as e:
            self.get_logger().error(f"Failed to load tag lookup file: {e}. Quitting!")
            rclpy.shutdown()

    def _declare_and_get_params(self):
>>>>>>> Stashed changes
        self.declare_parameter('usb_camera_topic', '/camera/image_raw')
        self.declare_parameter('ip_camera_topic', '/ip_camera/image_raw')
        self.declare_parameter('tag_size', 0.079375)
        self.declare_parameter('show_processed_video', True)
        self.declare_parameter('apply_map_frame_filtering', True)
        self.declare_parameter('map_filtering_lowpass_threshold', 0.5)
<<<<<<< Updated upstream

        # --- CAMERA 1 (USB) PARAMETERS ---
=======
>>>>>>> Stashed changes
        self.declare_parameter('usb_cam.fx', 921.62)
        self.declare_parameter('usb_cam.fy', 923.23)
        self.declare_parameter('usb_cam.cx', 614.94)
        self.declare_parameter('usb_cam.cy', 360.22)
<<<<<<< Updated upstream
        
        # --- CAMERA 2 (IP) PARAMETERS ---
        self.declare_parameter('ip_cam.fx', 461.839808) # Placeholder - YOU MUST CALIBRATE
        self.declare_parameter('ip_cam.fy', 458.096570) # Placeholder - YOU MUST CALIBRATE
        self.declare_parameter('ip_cam.cx', 310.772398) # Placeholder - YOU MUST CALIBRATE
        self.declare_parameter('ip_cam.cy', 220.487791) # Placeholder - YOU MUST CALIBRATE
        

        # Now, get the final values of the parameters
        self.usb_camera_topic = self.get_parameter('usb_camera_topic').get_parameter_value().string_value
        self.ip_camera_topic = self.get_parameter('ip_camera_topic').get_parameter_value().string_value
        self.tag_size = self.get_parameter('tag_size').get_parameter_value().double_value
        
=======
        # Updated camera intrinsic parameters from calibration
        self.declare_parameter('ip_cam.fx', 760.496)
        self.declare_parameter('ip_cam.fy', 757.610)
        self.declare_parameter('ip_cam.cx', 622.506)
        self.declare_parameter('ip_cam.cy', 326.603)
        self.refresh_parameters()
        self.add_on_set_parameters_callback(self.parameter_callback)
        
    def refresh_parameters(self):
        self.usb_camera_topic = self.get_parameter('usb_camera_topic').get_parameter_value().string_value
        self.ip_camera_topic = self.get_parameter('ip_camera_topic').get_parameter_value().string_value
        self.tag_size = self.get_parameter('tag_size').get_parameter_value().double_value
>>>>>>> Stashed changes
        self.usb_camera_params = [
            self.get_parameter('usb_cam.fx').get_parameter_value().double_value,
            self.get_parameter('usb_cam.fy').get_parameter_value().double_value,
            self.get_parameter('usb_cam.cx').get_parameter_value().double_value,
            self.get_parameter('usb_cam.cy').get_parameter_value().double_value,
        ]
        self.ip_camera_params = [
            self.get_parameter('ip_cam.fx').get_parameter_value().double_value,
            self.get_parameter('ip_cam.fy').get_parameter_value().double_value,
            self.get_parameter('ip_cam.cx').get_parameter_value().double_value,
            self.get_parameter('ip_cam.cy').get_parameter_value().double_value,
        ]
<<<<<<< Updated upstream

        self.apply_map_frame_filtering = self.get_parameter("apply_map_frame_filtering").value
        self.map_filtering_lowpass_threshold = self.get_parameter("map_filtering_lowpass_threshold").value

        self.add_on_set_parameters_callback(self.parameter_callback)

    def _initialize_detector(self):
        family = 'tag36h11'
=======
        self.apply_map_frame_filtering = self.get_parameter("apply_map_frame_filtering").value
        self.map_filtering_lowpass_threshold = self.get_parameter("map_filtering_lowpass_threshold").value

    def _initialize_detector(self):
>>>>>>> Stashed changes
        try:
            return Detector(families='tag36h11')
        except Exception as e:
            self.get_logger().error(f"Failed to initialize detector: {e}")
            rclpy.shutdown()

    def image_callback_cam1(self, msg: Image):
        self.process_image(msg, "cam1_usb", self.last_visible_tags_cam1, self.usb_camera_params, is_primary_cam=True)

<<<<<<< Updated upstream
    # --- Callback for Camera 1 (USB) ---
    def image_callback_cam1(self, msg: Image):
        self.process_image(msg, "cam1_detections", self.last_visible_tags_cam1, self.usb_camera_params)

    # --- Callback for Camera 2 (IP) ---
    def image_callback_cam2(self, msg: Image):
        self.process_image(msg, "cam2_detections", self.last_visible_tags_cam2, self.ip_camera_params)

    # --- Generic Image Processing Function ---
    def process_image(self, msg: Image, window_name: str, last_visible_tags: set, camera_params: list):
=======
    def image_callback_cam2(self, msg: Image):
        self.process_image(msg, "cam2_ip", self.last_visible_tags_cam2, self.ip_camera_params, is_primary_cam=False)

    def process_image(self, msg: Image, window_name: str, last_visible_tags: set, camera_params: list, is_primary_cam: bool):
>>>>>>> Stashed changes
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"Image conversion error for {window_name}: {e}")
            return
        
        current_visible_tags = set()
        post_detection_time = self.get_time_in_seconds()
        
        try:
            tags = self.detector.detect(gray, True, camera_params, self.tag_size)
<<<<<<< Updated upstream
            post_detection_time = self.get_time_in_seconds()

            if tags:
                self.get_logger().info(f"Found {len(tags)} tags in '{window_name}'")

            for tag in tags:
                tag_id = tag.tag_id
                current_visible_tags.add(tag_id)
                self._process_tag(msg.header, tag, image, post_detection_time)

            # Only process map frame from the primary camera to avoid conflicts
            if window_name == "cam1_detections":
                 self.process_map_frame(post_detection_time, msg.header)

        except KeyError as e:
            self.get_logger().error(f"Detection error for {window_name}: {e}")

        disappeared_tags = last_visible_tags - current_visible_tags
        for tag_id in disappeared_tags:
            self._publish_stale_tf(msg.header, tag_id)
        last_visible_tags.clear()
        last_visible_tags.update(current_visible_tags)


=======
            for tag in tags:
                current_visible_tags.add(tag.tag_id)
                self._process_tag(msg.header, tag, image, post_detection_time, is_primary_cam)
                self.process_map_frame(post_detection_time, msg.header)
        except Exception as e:
            self.get_logger().error(f"Detection error for {window_name}: {e}")
            
        disappeared_tags = last_visible_tags - current_visible_tags
        for tag_id in disappeared_tags:
            self._publish_stale_tf(msg.header, tag_id)
        
        last_visible_tags.clear()
        last_visible_tags.update(current_visible_tags)
        
>>>>>>> Stashed changes
        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            cv2.imshow(window_name, image)
            cv2.waitKey(1)

<<<<<<< Updated upstream
    def process_map_frame(self, time, header):

        #remove staled tags
        for tag_key in list(self.frame_tags.keys()):
            if(self.frame_tags[tag_key][1] + FRAME_STALE_TIME < time):
                self.frame_tags.pop(tag_key)
                self.get_logger().info(f"Tag {tag_key} has staled. Current time: {time}, tag time: {self.frame_tags[tag_key][1]}")

        tag_set = self.frame_tags

        #make sure there are enough staled keys to begin
        if(len(tag_set.keys()) < 3):
            return

        if(len(tag_set.keys()) > 4):
            #too many tags
            self.get_logger().warn("To many tags to complete frame detection! How?")
            return

        #get rid of the farthest tag if there are four
        if(len(tag_set.keys()) == 4):
            #track the tag with the worst sum of discrepancy from the others
            smallest_discrepancy = None
            smallest_discrepancy_index = 0

            for tag_key in tag_set.keys():
                #the tag's point
                p4  = (float(tag_set[tag_key][0][0]), float(tag_set[tag_key][0][1]), float(tag_set[tag_key][0][2]))

                # get the other three transforms
                other_keys = list(tag_set.keys())
                other_keys.remove(tag_key)
                p1 = (float(tag_set[other_keys[0]][0][0]), float(tag_set[other_keys[0]][0][1]), float(tag_set[other_keys[0]][0][2]))
                p2 = (float(tag_set[other_keys[1]][0][0]), float(tag_set[other_keys[1]][0][1]), float(tag_set[other_keys[1]][0][2]))
                p3 = (float(tag_set[other_keys[2]][0][0]), float(tag_set[other_keys[2]][0][1]), float(tag_set[other_keys[2]][0][2]))

                dist = self.get_4th_point_distance(p1, p2, p3, p4)
                if(smallest_discrepancy is None) or (dist < smallest_discrepancy):
                    smallest_discrepancy = dist
                    smallest_discrepancy_index = tag_key

            #all points colinear - do not pub
            if(smallest_discrepancy is None):
                self.get_logger().warn(f"Colinear detections, not updating map frame.")
                return

            #keep the best combination (remove the points with the shortest perpendiculr distance)
            tag_set.pop(smallest_discrepancy_index)
        
        tag_keys = list(tag_set.keys())
        #get the best plane
        k1, k2 ,k3 = self.find_plane_coefficients(
            (float(tag_set[tag_keys[0]][0][0]), float(tag_set[tag_keys[0]][0][1]), float(tag_set[tag_keys[0]][0][2])),
            (float(tag_set[tag_keys[1]][0][0]), float(tag_set[tag_keys[1]][0][1]), float(tag_set[tag_keys[1]][0][2])),
            (float(tag_set[tag_keys[2]][0][0]), float(tag_set[tag_keys[2]][0][1]), float(tag_set[tag_keys[2]][0][2])))

        if(k1 is None):
            #detections are colinear
            self.get_logger().warn(f"Colinear detections, not updating map frame.")
            return

        #determine the center point of the plane - find cp of two points on the corner from each other
        center = np.array((0,0,0))
        if(0 in tag_set.keys() and 2 in tag_set.keys()):
            center = (np.array((float(tag_set[0][0][0]), float(tag_set[0][0][1]), float(tag_set[0][0][2]))) + np.array((float(tag_set[2][0][0]), float(tag_set[2][0][1]), float(tag_set[2][0][2])))) / 2
        elif (1 in tag_set.keys() and 3 in tag_set.keys()):
            center = (np.array((float(tag_set[1][0][0]), float(tag_set[1][0][1]), float(tag_set[1][0][2]))) + np.array((float(tag_set[3][0][0]), float(tag_set[3][0][1]), float(tag_set[3][0][2])))) / 2
        else: # Fallback if diagonal tags aren't present
             p1 = np.array((float(tag_set[tag_keys[0]][0][0]), float(tag_set[tag_keys[0]][0][1]), float(tag_set[tag_keys[0]][0][2])))
             p2 = np.array((float(tag_set[tag_keys[1]][0][0]), float(tag_set[tag_keys[1]][0][1]), float(tag_set[tag_keys[1]][0][2])))
             p3 = np.array((float(tag_set[tag_keys[2]][0][0]), float(tag_set[tag_keys[2]][0][1]), float(tag_set[tag_keys[2]][0][2])))
             center = (p1+p2+p3)/3

        #determine the plane's angle relative to the camera
        quaternion = self.get_vector_quaternion([k1, k2, k3])

        if(self.apply_map_frame_filtering):
            #filter the map frame
            time = self.get_time_in_seconds()
            x_filtered = self.map_x_filter.update(center[0], time)
            y_filtered = self.map_y_filter.update(center[1], time)
            z_filtered = self.map_z_filter.update(center[2], time)
            qx_filtered = self.map_qx_filter.update(quaternion[1], time)
            qy_filtered = self.map_qy_filter.update(quaternion[2], time)
            qz_filtered = self.map_qz_filter.update(quaternion[3], time)
            qw_filtered = self.map_qw_filter.update(quaternion[0], time)

            self.publish_frame_tf(header, [qw_filtered, qx_filtered, qy_filtered, qz_filtered], [x_filtered, y_filtered, z_filtered])

        else:
            self.publish_frame_tf(header, quaternion, [center[0], center[1], center[2]])

    def get_vector_quaternion(self, plane):

        #normalize the plane's normal vector
        norm_plane = np.array(plane) /  np.linalg.norm(plane)


        dp = np.dot(norm_plane, np.array(FRAME_DEFAULT_VECTOR))

        #vector to rotate around
        rot_vector = np.cross(np.array(FRAME_DEFAULT_VECTOR), norm_plane)

        #compose quat
        quat = np.array([1+dp , rot_vector[0], rot_vector[1], rot_vector[2]])

        #normalize the quaternion
        quat = quat /np.linalg.norm(quat)

        return quat


    def find_plane_coefficients(self, p1, p2, p3):
        #whos a good llm

        # Convert points to numpy arrays
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

        # Create matrix A with the coordinates of the three points
        A = np.vstack([p1, p2, p3])  # Shape (3, 3)

        #colinear points
        if(np.linalg.matrix_rank(A) != 3):
            return None, None, None

        # Create vector b which is [1, 1, 1]
        b = np.ones(3)

        # Solve for [k1, k2, k3] in A @ [k1, k2, k3] = b
        try:
            k = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None, None, None

        return k[0], k[1], k[2]

    def get_4th_point_distance(self, p1, p2, p3, p4):

        #solve the plane for the first three points
        k1, k2, k3 = self.find_plane_coefficients(p1, p2, p3)

        if(k1 is None):
            #colinear points
            return None

        #get the parallel plane that the fourth point lies on
        k4 = abs(k1 * p4[0] + k2 * p4[1] + k3 * p4[2])

        #get the vector length between the two planes
        dist = k4 * pow((k1**2 + k2**2 + k3**2),.5)
=======
    def _process_tag(self, header, tag, image, time, is_primary_cam):
        if tag.pose_R is None or tag.pose_t is None or tag.tag_id not in self.registered_tags:
            return
            
        R = self.R_flip @ tag.pose_R
        t = self.R_flip @ tag.pose_t.reshape(3, 1)
        tag_key = f"{header.frame_id}_tag_{tag.tag_id}"
        
        smoothed_t = self.smoothing_alpha * t + (1 - self.smoothing_alpha) * self.tag_pose_cache.get(tag_key, t)
        self.tag_pose_cache[tag_key] = smoothed_t
        
        # ---- Transform tag pose into the common parent frame ----
        parent = self.map_parent_frame
        cam = header.frame_id
        t_in_parent = smoothed_t
        R_in_parent = R
>>>>>>> Stashed changes

        if cam != parent:
            try:
        # TF: parent <- cam (at current time)
                tfmsg = self.tf_buffer.lookup_transform(parent, cam, rclpy.time.Time())
        # translation + quaternion of camera in parent frame
                tx = tfmsg.transform.translation.x
                ty = tfmsg.transform.translation.y
                tz = tfmsg.transform.translation.z
                qx = tfmsg.transform.rotation.x
                qy = tfmsg.transform.rotation.y
                qz = tfmsg.transform.rotation.z
                qw = tfmsg.transform.rotation.w

        # Build 4x4 transform parent_T_cam
                Tpc = quaternion_matrix([qx, qy, qz, qw])
                Tpc[:3, 3] = [tx, ty, tz]

        # Tag pose in cam: [R | t]
                Tc_tag = np.eye(4)
                Tc_tag[:3, :3] = R
                Tc_tag[:3, 3] = smoothed_t.flatten()

        # Convert to parent: parent_T_tag = parent_T_cam * cam_T_tag
                Tp_tag = Tpc.dot(Tc_tag)

                R_in_parent = Tp_tag[:3, :3]
                t_in_parent = Tp_tag[:3, 3].reshape(3, 1)

            except Exception as e:
                self.get_logger().warn(f"TF lookup {parent} <- {cam} failed: {e}. Using cam frame for this tag.")

# Store tag in the common frame for plane fitting
        if tag.tag_id in self.map_tags:
            corner_id = self.tag_lookup.get(f"tag_{tag.tag_id}", {}).get("corner")
            if corner_id is not None:
                self.frame_tags[corner_id] = (t_in_parent, time)

# Still publish tag relative to its camera frame (for debugging)
        self._publish_tf(header, tag.tag_id, R, smoothed_t)

        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            color = (0, 255, 0)
            if tag.tag_id in self.map_tags:
                color = MAP_TAG_COLOR
            elif tag.tag_id in self.robot_tags:
                color = ROBOT_TAG_COLOR
            elif tag.tag_id in self.obstacle_tags:
                color = OBSTACLE_TAG_COLOR
            self._draw_tag(image, tag, color)

    def process_map_frame(self, time, header):
    # Remove stale tags
        for tag_key in list(self.frame_tags.keys()):
            if (self.frame_tags[tag_key][1] + FRAME_STALE_TIME) < time:
                self.frame_tags.pop(tag_key)

<<<<<<< Updated upstream
    def _process_tag(self, header, tag, image, time):
        if tag.pose_R is None or tag.pose_t is None:
            return

        tag_id = tag.tag_id

        R = self.R_flip @ tag.pose_R
        t = self.R_flip @ tag.pose_t.reshape(3, 1)

        # Smoothing
        tag_key = f"tag_{tag.tag_id}"
        alpha = self.smoothing_alpha

        if tag_key in self.tag_pose_cache:
            prev_t = self.tag_pose_cache[tag_key]
            smoothed_t = alpha * t + (1 - alpha) * prev_t
=======
    # Need at least 3 tags to define a plane
        if len(self.frame_tags) < 3:
            return

    # Gather all tag positions (already transformed into map_parent_frame)
        points = np.array([p[0].flatten() for p in self.frame_tags.values()])
        center = np.mean(points, axis=0)

    # --- Plane fitting via SVD ---
    # Subtract mean → SVD → normal is last singular vector
        _, _, vh = np.linalg.svd(points - center)
        normal = vh[2, :]  # plane normal

    # Convert normal into quaternion aligning FRAME_DEFAULT_VECTOR (0,0,-1) to plane
        quaternion = self.get_vector_quaternion(normal)

    # Optional filtering (smooths orientation)
        if self.apply_map_frame_filtering:
            now = self.get_time_in_seconds()
            qw, qx, qy, qz = (
                self.map_qw_filter.update(quaternion[0], now),
                self.map_qx_filter.update(quaternion[1], now),
                self.map_qy_filter.update(quaternion[2], now),
                self.map_qz_filter.update(quaternion[3], now),
            )
            norm = np.linalg.norm([qw, qx, qy, qz])
            final_quat = [qw / norm, qx / norm, qy / norm, qz / norm]
>>>>>>> Stashed changes
        else:
            final_quat = quaternion

    # --- Origin choice ---
    # Right now: hardcode (0,0,0) for tile 1
        origin = np.array([0.0, 0.0, 0.0])

    # Publish "map" frame relative to the chosen parent
        self.publish_frame_tf(header, final_quat, origin)


    def find_plane_coefficients(self, p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        A = np.vstack([p1, p2, p3])
        if np.linalg.matrix_rank(A) < 3:
            return None
        try:
            return np.linalg.solve(A, np.ones(3))
        except np.linalg.LinAlgError:
            return None

    def get_vector_quaternion(self, plane_normal):
    # Normalize
        norm_plane = plane_normal / np.linalg.norm(plane_normal)

    # Ensure consistent orientation (flip if pointing away from default vector)
        if np.dot(norm_plane, FRAME_DEFAULT_VECTOR) < 0:
            norm_plane = -norm_plane

    # Compute rotation quaternion from FRAME_DEFAULT_VECTOR → norm_plane
        dp = np.dot(norm_plane, FRAME_DEFAULT_VECTOR)
        rot_vector = np.cross(FRAME_DEFAULT_VECTOR, norm_plane)

<<<<<<< Updated upstream
            if tag_id in self.map_tags:      # map
                color = MAP_TAG_COLOR
            elif tag_id in self.robot_tags:    # AMRs
                color = ROBOT_TAG_COLOR
            elif tag_id in self.obstacle_tags:   #obstalces
                color = OBSTACLE_TAG_COLOR
=======
        if np.linalg.norm(rot_vector) < 1e-6:
        # Vectors aligned or opposite
            return np.array([1.0, 0.0, 0.0, 0.0]) if dp > 0 else np.array([0.0, 1.0, 0.0, 0.0])
>>>>>>> Stashed changes

        quat = np.array([1 + dp, *rot_vector])
        return quat / np.linalg.norm(quat)

    def _publish_tf(self, header, tag_id, R, t):
        try:
            T = np.identity(4)
            T[:3, :3] = R
            quat = quaternion_from_matrix(T)
            t_msg = TransformStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = header.frame_id
            t_msg.child_frame_id = f"{header.frame_id}_tag_{tag_id}"
            t_msg.transform.translation.x, t_msg.transform.translation.y, t_msg.transform.translation.z = t.flatten()
            t_msg.transform.rotation.x, t_msg.transform.rotation.y, t_msg.transform.rotation.z, t_msg.transform.rotation.w = quat
            self.tf_broadcaster.sendTransform(t_msg)
        except Exception as e:
            self.get_logger().error(f"TF publish error for tag {tag_id}: {e}")

    def publish_frame_tf(self, header, quat, center):
        try:
            t_msg = TransformStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = self.map_parent_frame

            t_msg.child_frame_id = "map"

            t_msg.transform.translation.x = float(center[0])
            t_msg.transform.translation.y = float(center[1])
            t_msg.transform.translation.z = float(center[2])
            t_msg.transform.rotation.w = quat[0]
            t_msg.transform.rotation.x = quat[1]
            t_msg.transform.rotation.y = quat[2]
            t_msg.transform.rotation.z = quat[3]

            self.tf_broadcaster.sendTransform(t_msg)
        except Exception as e:
            self.get_logger().error(f"TF publish error for map: {e}")


    def _publish_stale_tf(self, header, tag_id):
        try:
            past_time = Time.from_msg(header.stamp) - Duration(seconds=1.0)
            t_msg = TransformStamped()
            t_msg.header.stamp = past_time.to_msg()
            t_msg.header.frame_id = header.frame_id
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
<<<<<<< Updated upstream
                pt1, pt2 = tuple(corners[i]), tuple(corners[(i + 1) % 4])
                cv2.line(frame, pt1, pt2, draw_color_bgr, 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {tag.tag_id}", (corners[0][0] + 5, corners[0][1] - 10),
=======
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), draw_color_bgr, 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {tag.tag_id}", (corners[0][0], corners[0][1] - 15),
>>>>>>> Stashed changes
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color_bgr, 2)
        except Exception as e:
            self.get_logger().error(f"Draw error for tag {tag.tag_id}: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down node and closing windows.")
        if hasattr(self, 'subscription_cam1'):
            self.destroy_subscription(self.subscription_cam1)
        if hasattr(self, 'subscription_cam2'):
            self.destroy_subscription(self.subscription_cam2)

        cv2.destroyAllWindows()
        super().destroy_node()

    def parameter_callback(self, params):
        for param in params:
            if param.name == "show_processed_video":
                self.get_logger().info(f"Live display toggled to: {param.value}")
        return SetParametersResult(successful=True)

    def get_time_in_seconds(self):
<<<<<<< Updated upstream
        #returns the time in second
        tuple_time = self.get_clock().now().seconds_nanoseconds()

        return tuple_time[0] + tuple_time[1] * .000000001

    def refresh_parameters(self):
        self.apply_map_frame_filtering = self.get_parameter("apply_map_frame_filtering").value
        self.map_filtering_lowpass_threshold = self.get_parameter("map_filtering_lowpass_threshold").value
=======
        now = self.get_clock().now()
        return now.seconds_nanoseconds()[0] + now.seconds_nanoseconds()[1] / 1e9

>>>>>>> Stashed changes

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
