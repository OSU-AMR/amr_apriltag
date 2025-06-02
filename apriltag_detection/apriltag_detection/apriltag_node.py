import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import cv2
from cv_bridge import CvBridge
from dt_apriltags import Detector
import numpy as np
import tf2_ros
from tf_transformations import quaternion_from_matrix, rotation_matrix
import math
from rclpy.time import Time
from rclpy.duration import Duration

class apriltag_node(Node):
    def __init__(self):
        super().__init__('apriltag_node')
        self.get_logger().info("Starting AprilTag Node with TF broadcast + smoothing + color (ID-based) + TF cleanup")

        # === Parameters ===
        self._declare_and_get_params()

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

        # === Subscription ===
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(f"Subscribed to {self.image_topic}")
        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            self.get_logger().info("Video output enabled.")

    def _declare_and_get_params(self):
        def declare_param(name, default): return self.declare_parameter(name, default).value

        self.image_topic = declare_param('subscribe_topic', '/camera/image_raw')
        self.tag_size = declare_param('tag_size', 0.079375)
        self.camera_params = [
            declare_param('camera_params.fx', 940.62),
            declare_param('camera_params.fy', 946.23),
            declare_param('camera_params.cx', 633.94),
            declare_param('camera_params.cy', 374.22),
        ]
        self.declare_parameter('show_processed_video', True)
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
            for tag in tags:
                tag_id = tag.tag_id
                current_visible_tags.add(tag_id)
                self._process_tag(msg.header, tag, image) # Removed family argument
        except Exception as e:
            self.get_logger().error(f"Detection error: {e}") # Modified error log

        # Handle tag disappearance
        disappeared_tags = self.last_visible_tags - current_visible_tags
        for tag_id in disappeared_tags:
            self._publish_stale_tf(msg.header, tag_id)
        self.last_visible_tags = current_visible_tags

        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            cv2.imshow("AprilTag Detection", image)
            cv2.waitKey(1)

    def _process_tag(self, header, tag, image): # Removed family argument
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

        # TF Publish
        self._publish_tf(header, tag_id, R, smoothed_t)

        # Drawing with color based on tag_id
        if self.get_parameter("show_processed_video").get_parameter_value().bool_value:
            color = (0, 255, 0)  # Default Green

            if tag_id == 100:      # Floor
                color = (255, 0, 0)  # Blue
            elif tag_id == 3:    # AMRs
                color = (0, 0, 255)  # Red
            elif tag_id == 50:   # Obstacles
                color = (0, 255, 255) # Yellow
            # else, color remains default Green for other tag36h11 IDs

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