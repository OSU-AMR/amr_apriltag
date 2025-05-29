import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from dt_apriltags import Detector
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker

class CourseManagerNode(Node):
    def __init__(self):
        super().__init__('course_manager_node')

        # Initialize webcam
        self.cap = cv2.VideoCapture(4)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open webcam.")
            rclpy.shutdown()
            return

        # Initialize AprilTag detectors for different families
        self.floor_detector = Detector(families='tag36h11')
        self.obstacle_detector = Detector(families='tagStandard41h12')
        self.amr_detector = Detector(families='tagStandard52h13')

        # Camera parameters
        #self.camera_params = [940.62, 633.94, 946.23, 374.22]
        self.camera_params = [1351.3, 1368.6, 973.12,  562.49]
        self.tag_size = 0.096

        # ROS2 setup
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.marker_publisher = self.create_publisher(Marker, 'apriltag_marker', 10)
        self.obstacle_report_publisher = self.create_publisher(String, 'obstacle_reports', 10)
        self.timer = self.create_timer(0.033, self.process_image)

        self.collision_distance = 0.3  # Minimum distance for collision (meters)
        self.collision_alert_publisher = self.create_publisher(String, 'collision_alerts', 10)

    def process_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Warning: Failed to read frame from webcam.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags from all families
        floor_tags = self.floor_detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size)
        obstacle_tags = self.obstacle_detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size)
        amr_tags = self.amr_detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size)

        for tag in floor_tags:
            self.draw_tag(frame, tag, (0, 255, 0))
            self.publish_tf(tag)
            self.publish_marker(tag, "floor")

        for tag in obstacle_tags:
            self.draw_tag(frame, tag, (0, 0, 255))
            self.publish_tf(tag)
            self.publish_marker(tag, "obstacle")
            self.report_obstacle(tag)  # Report obstacle

        for tag in amr_tags:
            self.draw_tag(frame, tag, (255, 0, 0))
            self.publish_tf(tag)
            self.publish_marker(tag, "amr")

        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_frame'
        self.image_publisher.publish(ros_image)

        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)

        self.check_for_collisions(amr_tags, obstacle_tags)  # Check for collisions

    def draw_tag(self, frame, tag, color):
        corners = tag.corners.astype(int)
        center_x, center_y = np.mean(corners, axis=0).astype(int)

        cv2.putText(frame, str(tag.tag_id), (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, 2)

    def publish_tf(self, tag):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "camera_link"
        t.child_frame_id = f"tag_{tag.tag_id}"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = tag.pose_t.flatten()

        T = np.eye(4)
        T[:3, :3] = tag.pose_R
        quaternion = quaternion_from_matrix(T)

        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quaternion
        self.tf_broadcaster.sendTransform(t)

    def publish_marker(self, tag, marker_type):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "camera_link"
        marker.ns = "apriltags"
        marker.id = tag.tag_id
        marker.action = Marker.ADD

        marker.pose.position.x = tag.pose_t[0][0]
        marker.pose.position.y = tag.pose_t[1][0]
        marker.pose.position.z = tag.pose_t[2][0]

        if marker_type == "obstacle":
            marker.type = Marker.CUBE
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0  # Red
            marker.text = f"Obstacle Pos: ({tag.pose_t[0][0]:.2f}, {tag.pose_t[1][0]:.2f}, {tag.pose_t[2][0]:.2f})\n"
        elif marker_type == "amr":
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = marker.scale.y = marker.scale.z = 0.15
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 1.0  # Blue
            marker.text = f"AMR Pos: ({tag.pose_t[0][0]:.2f}, {tag.pose_t[1][0]:.2f}, {tag.pose_t[2][0]:.2f})\n"
        else:
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = marker.scale.y = marker.scale.z = 0.1
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 1.0, 0.0, 1.0  # Yellow
            marker.text = f"Floor Pos: ({tag.pose_t[0][0]:.2f}, {tag.pose_t[1][0]:.2f}, {tag.pose_t[2][0]:.2f})\n"

        self.marker_publisher.publish(marker)

    def report_obstacle(self, tag):
        """Publishes obstacle information."""

        obstacle_msg = String()
        obstacle_msg.data = f"Obstacle {tag.tag_id} detected at x={tag.pose_t[0][0]:.2f}, y={tag.pose_t[1][0]:.2f}, z={tag.pose_t[2][0]:.2f}"
        self.obstacle_report_publisher.publish(obstacle_msg)
        self.get_logger().info(obstacle_msg.data)

    def check_for_collisions(self, amr_tags, obstacle_tags):
        """Checks for collisions between AMRs and obstacles."""

        for i in range(len(amr_tags)):
            amr1 = amr_tags[i]
            for j in range(i + 1, len(amr_tags)):  # Avoid comparing the same AMRs twice
                amr2 = amr_tags[j]
                if self.is_collision(amr1, amr2):
                    self.report_collision(amr1, amr2)

            for obstacle in obstacle_tags:
                if self.is_collision(amr1, obstacle):
                    self.report_collision(amr1, obstacle)

    def is_collision(self, tag1, tag2):
        """Checks if two tags are close enough to be considered a collision."""

        pos1 = np.array(tag1.pose_t.flatten())
        pos2 = np.array(tag2.pose_t.flatten())
        distance = np.linalg.norm(pos1 - pos2)
        return distance < self.collision_distance

    def report_collision(self, tag1, tag2):
        """Publishes a collision alert message."""

        alert_msg = String()
        alert_msg.data = f"⚠️ Potential Collision: {self.get_tag_type(tag1)} {tag1.tag_id} and {self.get_tag_type(tag2)} {tag2.tag_id}"
        self.collision_alert_publisher.publish(alert_msg)
        self.get_logger().warn(alert_msg.data)

    def get_tag_type(self, tag):
        """Determines the type of a tag (AMR or Obstacle)."""

        #  This is a simplified way to determine the tag type.
        #  You might need a more robust method depending on your tag families.
        if tag.tag_family == 'tagStandard52h13':  # Adjust if needed
            return "AMR"
        elif tag.tag_family == 'tagStandard41h12':  # Adjust if needed
            return "Obstacle"
        else:
            return "Unknown"

def main(args=None):
    rclpy.init(args=args)
    node = CourseManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
