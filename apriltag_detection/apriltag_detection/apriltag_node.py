#! /usr/bin/env python3

import os
import yaml

from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import cv2
from cv_bridge import CvBridge
from dt_apriltags import Detector
import numpy as np
import tf2_ros
from tf_transformations import quaternion_from_matrix, rotation_matrix
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.duration import Duration

from ament_index_python.packages import get_package_share_directory

APRIL_TAG_LOOKUP_SUB_PATH = "test_data/april_tag_lookup.yaml"

ROBOT_TAG_COLOR = (172,16,48)
MAP_TAG_COLOR = (255,165,0)
OBSTACLE_TAG_COLOR = (0,39, 76)

FRAME_STALE_TIME = 5 #seconds

FRAME_DEFAULT_VECTOR = [0,0,-1]

class low_pass_filtered_value():

    #initialize the value
    def __init__ (self, starting_value, cutoff_frequency, init_time):
        self.value = starting_value
        self.cutoff_frequency = cutoff_frequency
        self.previous_time = init_time

    def update(self, measurement, time):
        #i know this is wrong but it also works

        self.value = self.value * pow(10, -(time - self.previous_time) * self.cutoff_frequency) + measurement * (1 - pow(10, -(time - self.previous_time) * self.cutoff_frequency))
        self.previous_time = time

        return self.value


    def set_cutoff_frequency(self, frequency):
        self.cutoff_frequency = frequency


class apriltag_node(Node):
    def __init__(self):
        super().__init__('apriltag_node')
        self.get_logger().info("Starting AprilTag Node with TF broadcast + smoothing + color (ID-based) + TF cleanup")

        # === Parameters ===
        self._declare_and_get_params()

        #load in data
        self.load_in_tag_data()

        # === AprilTag Detector ===
        self.detector = self._initialize_detector() # Changed from self.detectors

        # === Tools ===
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.R_flip = rotation_matrix(math.pi, [1.0, 0.0, 0.0])[:3, :3]

        # === State ===
        self.tag_pose_cache = {}
        self.smoothing_alpha = 0.2
        self.last_visible_tags = set()

        self.frame_tags = dict()

        # === Subscription ===
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
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

        self.get_logger().info(f"Subscribed to {self.image_topic}")
        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            self.get_logger().info("Video output enabled.")

    def load_in_tag_data(self):
        try:
            config_path = os.path.join(get_package_share_directory("amr_central"), APRIL_TAG_LOOKUP_SUB_PATH)

            with open(config_path) as config_file:
                self.tag_lookup = yaml.safe_load(config_file)
                self.registered_tags = self.tag_lookup["registered_tags"]
                self.robot_tags = self.tag_lookup["robot_tags"]
                self.map_tags = self.tag_lookup["map_tags"]
                self.obstacle_tags = self.tag_lookup["obstacle_tags"]
                self.corner_tag_distance = self.tag_lookup["corner_tag_distance"]
            
        except:
            self.get_logger().error("Failed to load tag lookup file. Quitting!")
            quit()

    def _declare_and_get_params(self):
        def declare_param(name, default): return self.declare_parameter(name, default).value

        self.image_topic = declare_param('subscribe_topic', '/camera/image_raw')

        self.tag_size = declare_param('tag_size', 0.079375)
        #self.camera_params = [
        #    declare_param('camera_params.fx', 940.62),
        #    declare_param('camera_params.fy', 946.23),
        #    declare_param('camera_params.cx', 633.94),
        #    declare_param('camera_params.cy', 374.22),
        #]

        self.camera_params = [
            declare_param('camera_params.fx', 921.62),
            declare_param('camera_params.fy', 923.23),
            declare_param('camera_params.cx', 614.94),
            declare_param('camera_params.cy', 360.22),
        ]


        self.declare_parameter('show_processed_video', True)
        self.declare_parameter('apply_map_frame_filtering', True)
        self.declare_parameter('map_filtering_lowpass_threshold', 0.5)

        self.apply_map_frame_filtering = self.get_parameter("apply_map_frame_filtering").value
        self.map_filtering_lowpass_threshold = self.get_parameter("map_filtering_lowpass_threshold").value
        
        self.add_on_set_parameters_callback(self.parameter_callback)

    def _initialize_detector(self): # Renamed and modified
        family = 'tag36h11'
        try:
            detector = Detector(families=family)
            self.get_logger().info(f"Initialized detector for family: {family}")
            return detector
        except Exception as e:
            self.get_logger().error(f"Failed to initialize detector for family {family}: {e}")
            raise

    def image_callback(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        current_visible_tags = set()

        try:
            # Detect tags using the single detector
            tags = self.detector.detect(gray, True, self.camera_params, self.tag_size)

            post_detection_time = self.get_time_in_seconds()

            for tag in tags:
                tag_id = tag.tag_id
                current_visible_tags.add(tag_id)
                self._process_tag(msg.header, tag, image, post_detection_time) # Removed family argument

            #handle the map frame
            self.process_map_frame(post_detection_time, msg.header)
        except KeyError as e:
            self.get_logger().error(f"Detection error: {e}") # Modified error log

        # Handle tag disappearance
        disappeared_tags = self.last_visible_tags - current_visible_tags
        for tag_id in disappeared_tags:
            self._publish_stale_tf(msg.header, tag_id)
        self.last_visible_tags = current_visible_tags

        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            cv2.imshow("AprilTag Detection", image)
            cv2.waitKey(1)

    def process_map_frame(self, time, header):

        #remove staled tags
        for tag in self.frame_tags.keys():
            if(self.frame_tags[tag][1] + FRAME_STALE_TIME < time):
                self.frame_tags.pop(tag)
                self.get_logger().info(f"Tag {tag} has staled. Current time: {time}, tag time: {self.frame_tags[tag][1]}")

        tag_set = self.frame_tags

        #make sure there are enough staled keys to begin
        if(len(self.frame_tags.keys()) < 3):
            return
        
        if(len(self.frame_tags.keys()) > 4):
            #too many tags
            self.get_logger().warn("To many tags to complete frame detection! How?")

            return

        #get rid of the farthest tag if there are four
        if(len(self.frame_tags.keys()) == 4):
            #track the tag with the worst sum of discrepancy from the others
            smallest_discrepancy = None
            smallest_discrepancy_index = 0

            for tag in tag_set.keys():
                #the tag's point
                p4  = (float(tag_set[tag][0][0]), float(tag_set[tag][0][1]), float(tag_set[tag][0][2]))

                # get the other three transforms
                p1 = (float(tag_set[(tag + 1) % 4][0][0]), float(tag_set[(tag + 1) % 4][0][1]), float(tag_set[(tag + 1) % 4][0][2]))
                p2 = (float(tag_set[(tag + 2) % 4][0][0]), float(tag_set[(tag + 2) % 4][0][1]), float(tag_set[(tag + 2) % 4][0][2]))
                p3 = (float(tag_set[(tag + 3) % 4][0][0]), float(tag_set[(tag + 3) % 4][0][1]), float(tag_set[(tag + 3) % 4][0][2]))

                dist = self.get_4th_point_distance(p1, p2, p3, p4)
                if(smallest_discrepancy is None) or (dist < smallest_discrepancy):
                    smallest_discrepancy = dist
                    smallest_discrepancy_index = tag

            #all points colinear - do not pub
            if(smallest_discrepancy is None):
                self.get_logger().warn(f"Colinear detections, not updating map frame. {(float(tag_set[tag_set.keys()[0]][0][0]), float(tag_set[tag_set.keys()[0]][0][1]), float(tag_set[tag_set.keys()[0]][0][2]))}, {(float(tag_set[tag_set.keys()[1]][0][0]), float(tag_set[tag_set.keys()[1]][0][1]), float(tag_set[tag_set.keys()[1]][0][2]))}, {(float(tag_set[tag_set.keys()[2]][0][0]), float(tag_set[tag_set.keys()[2]][0][1]), float(tag_set[tag_set.keys()[2]][0][2]))}")
                return
            
            #keep the best combination (remove the points with the shortest perpendiculr distance)
            tag_set.pop(smallest_discrepancy_index)

        #get the best plane
        k1, k2 ,k3 = self.find_plane_coefficients(
            (float(tag_set[list(tag_set.keys())[0]][0][0]), float(tag_set[list(tag_set.keys())[0]][0][1]), float(tag_set[list(tag_set.keys())[0]][0][2])),
            (float(tag_set[list(tag_set.keys())[1]][0][0]), float(tag_set[list(tag_set.keys())[1]][0][1]), float(tag_set[list(tag_set.keys())[1]][0][2])),
            (float(tag_set[list(tag_set.keys())[2]][0][0]), float(tag_set[list(tag_set.keys())[2]][0][1]), float(tag_set[list(tag_set.keys())[2]][0][2])))
        
        if(k1 is None):
            #detections are colinear
            self.get_logger().warn(f"Colinear detections, not updating map frame. {(float(tag_set[tag_set.keys()[0]][0][0]), float(tag_set[tag_set.keys()[0]][0][1]), float(tag_set[tag_set.keys()[0]][0][2]))}, {(float(tag_set[tag_set.keys()[1]][0][0]), float(tag_set[tag_set.keys()[1]][0][1]), float(tag_set[tag_set.keys()[1]][0][2]))}, {(float(tag_set[tag_set.keys()[2]][0][0]), float(tag_set[tag_set.keys()[2]][0][1]), float(tag_set[tag_set.keys()[2]][0][2]))}")
            return

        #determine the center point of the plane - find cp of two points on the corner from each other
        center = np.array((0,0,0))
        if(0 in tag_set.keys() and 2 in tag_set.keys()):
            center = (np.array((float(tag_set[0][0][0]), float(tag_set[0][0][1]), float(tag_set[0][0][2]))) + np.array((float(tag_set[2][0][0]), float(tag_set[2][0][1]), float(tag_set[2][0][2])))) / 2
        else:
            center = (np.array((float(tag_set[1][0][0]), float(tag_set[1][0][1]), float(tag_set[1][0][2]))) + np.array((float(tag_set[3][0][0]), float(tag_set[3][0][1]), float(tag_set[3][0][2])))) / 2

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
        k = np.linalg.solve(A, b)
        
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

        return dist



    def _process_tag(self, header, tag, image, time): # Removed family argument
        if tag.pose_R is None or tag.pose_t is None:
            return

        tag_id = tag.tag_id

        R = self.R_flip @ tag.pose_R
        t = self.R_flip @ tag.pose_t.reshape(3, 1)

        # Smoothing
        tag_key = f"{tag_id}" # Already uses tag_id, so no change needed here
        alpha = self.smoothing_alpha

        if tag_key in self.tag_pose_cache:
            prev_t = self.tag_pose_cache[tag_key]
            smoothed_t = alpha * t + (1 - alpha) * prev_t
        else:
            smoothed_t = t

        self.tag_pose_cache[tag_key] = smoothed_t

        #don't show unregistered tags
        if not (tag_id in self.registered_tags):
            return


        if(tag_id in self.map_tags):

            #append to the frame detection
            self.frame_tags[self.tag_lookup[f"tag_{tag_id}"]["corner"]] = (t, time)


        # TF Publish
        self._publish_tf(header, tag_id, R, smoothed_t)

        # Drawing with color based on tag_id
        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            color = (0, 255, 0)  # Default Green

            if tag_id in self.map_tags:      # map
                color = MAP_TAG_COLOR  
            elif tag_id in self.robot_tags:    # AMRs
                color = ROBOT_TAG_COLOR 
            elif tag_id in self.obstacle_tags:   #obstalces
                color = OBSTACLE_TAG_COLOR

            self._draw_tag(image, tag, color)

    def _publish_tf(self, header, tag_id, R, t):
        try:
            T = np.identity(4)
            T[:3, :3] = R
            quat = quaternion_from_matrix(T)

            t_msg = TransformStamped()
            t_msg.header.stamp = header.stamp
            t_msg.header.frame_id = header.frame_id
            t_msg.child_frame_id = f"tag_{tag_id}"

            t_msg.transform.translation.x = float(t[0])
            t_msg.transform.translation.y = float(t[1])
            t_msg.transform.translation.z = float(t[2])
            t_msg.transform.rotation.x = quat[0]
            t_msg.transform.rotation.y = quat[1]
            t_msg.transform.rotation.z = quat[2]
            t_msg.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(t_msg)
        except Exception as e:
            self.get_logger().error(f"TF publish error for tag {tag_id}: {e}")

    def publish_frame_tf(self, header ,quat, center):
        try:
            t_msg = TransformStamped()
            t_msg.header.stamp = header.stamp
            t_msg.header.frame_id = header.frame_id
            t_msg.child_frame_id = "map"

            t_msg.transform.translation.x = float(center[0])
            t_msg.transform.translation.y = float(center[1])
            t_msg.transform.translation.z = float(center[2])
            t_msg.transform.rotation.x = quat[1]
            t_msg.transform.rotation.y = quat[2]
            t_msg.transform.rotation.z = quat[3]
            t_msg.transform.rotation.w = quat[0]

            self.tf_broadcaster.sendTransform(t_msg)
        except Exception as e:
            self.get_logger().error(f"TF publish error for map: {e}")

    def _publish_stale_tf(self, header, tag_id):
        try:
            past_time = Time.from_msg(header.stamp) - Duration(seconds=1)
            t_msg = TransformStamped()
            t_msg.header.stamp = past_time.to_msg()
            t_msg.header.frame_id = header.frame_id
            t_msg.child_frame_id = f"tag_{tag_id}"

            t_msg.transform.translation.x = 0.0
            t_msg.transform.translation.y = 0.0
            t_msg.transform.translation.z = 0.0
            t_msg.transform.rotation.x = 0.0
            t_msg.transform.rotation.y = 0.0
            t_msg.transform.rotation.z = 0.0
            t_msg.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish stale TF for tag {tag_id}: {e}")

    def _draw_tag(self, frame, tag, draw_color_bgr):
        try:
            corners = tag.corners.astype(int)
            center = tuple(tag.center.astype(int))
            for i in range(4):
                pt1, pt2 = tuple(corners[i]), tuple(corners[(i + 1) % 4])
                cv2.line(frame, pt1, pt2, draw_color_bgr, 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1) # Center circle is still red, can be changed if needed
            cv2.putText(frame, f"ID: {tag.tag_id}", (corners[0][0] + 5, corners[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color_bgr, 2)
        except Exception as e:
            self.get_logger().error(f"Draw error for tag {tag.tag_id}: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down node and closing windows.")
        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()

    def parameter_callback(self, params):
        for param in params:
            if param.name == "show_processed_video":
                self.get_logger().info(f"Live display toggled to: {param.value}")
        return rclpy.parameter.SetParametersResult(successful=True)

    def get_time_in_seconds(self):
        #returns the time in second
        tuple_time = self.get_clock().now().seconds_nanoseconds()
            
        return tuple_time[0] + tuple_time[1] * .000000001
    
    def refresh_parameters(self):
        self.apply_map_frame_filtering = self.get_parameter("apply_map_frame_filtering").value
        self.map_filtering_lowpass_threshold = self.get_parameter("map_filtering_lowpass_threshold").value

def main(args=None):
    rclpy.init(args=args)
    node = apriltag_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()