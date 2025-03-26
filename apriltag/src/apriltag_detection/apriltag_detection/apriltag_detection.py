import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from dt_apriltags import Detector
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker
from std_msgs.msg import String

class WebcamAprilTagNode(Node):
    def __init__(self):
        super().__init__('webcam_apriltag_node')
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open webcam.")
            exit()

        # Initialize AprilTag detector
        self.at_detector = Detector(families='tag36h11')

        # Camera parameters 
        self.camera_matrix = np.array([
            [940.62252582 , 0, 633.94],
            [0, 946.23605109 ,374.2256],
            [0, 0, 1]
        ])

        self.dist_coeffs = np.array([0.04200936 ,-0.09287515 , 0.00434402 , 0.00175762 , 0.04323584])
        
        self.camera_params = [940.62252582 , 633.94,
                             946.23605109, 374.2256]
        self.tag_size = 0.100  # Tag size in meters

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create publisher for the image
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.marker_publisher = self.create_publisher(Marker, 'apriltag_marker', 10)
        self.timer = self.create_timer(0.060, self.process_image)  # 30 Hz
    
    def process_image(self):
        ret, frame = self.cap.read()
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.at_detector.detect(gray, estimate_tag_pose=True,
                                           camera_params=self.camera_params,
                                           tag_size=self.tag_size)

            for tag in tags:
                # Extract corners
                corners = tag.corners.astype(int)

                # Draw the ID of the AprilTag and bounding box
                tag_id = str(tag.tag_id)
                center_x = int((corners[0][0] + corners[2][0]) / 2)
                center_y = int((corners[0][1] + corners[2][1]) / 2)
                cv2.putText(frame, tag_id, (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw lines connecting the corners
                for i in range(4):
                    start_point = tuple(corners[i])
                    end_point = tuple(corners[(i + 1) % 4])
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                # Publish the transform
                self.publish_tf(tag)

                # Publish the marker for RViz
                self.publish_marker(tag)

            # Publish the image
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera_frame'
            self.image_publisher.publish(ros_image)

            # Show the video with detections
            cv2.imshow('Webcam', frame)
            cv2.waitKey(1)
        else:
            self.get_logger().error("Error: Could not read frame from webcam.")

    def publish_tf(self, tag):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "camera_link"  # Change if needed
        t.child_frame_id = f"tag_{tag.tag_id}"

        t.transform.translation.x = tag.pose_t[0][0]
        t.transform.translation.y = tag.pose_t[1][0]
        t.transform.translation.z = tag.pose_t[2][0]

        R = tag.pose_R

        # Correct Quaternion Calculation:
        T = np.eye(4)
        T[:3, :3] = R
        quaternion = quaternion_from_matrix(T)

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t)
    
    def publish_marker(self, tag):
        # Create a marker for the AprilTag
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "camera_link"  # You can change this to whatever base frame you want
        marker.ns = "apriltags"
        marker.id = tag.tag_id
        marker.type = Marker.TEXT_VIEW_FACING  # This marker type is for displaying text
        marker.action = Marker.ADD

        # Set the pose of the marker (translation and rotation)
        marker.pose.position.x = tag.pose_t[0][0]
        marker.pose.position.y = tag.pose_t[1][0]
        marker.pose.position.z = tag.pose_t[2][0]

        # Set the orientation (rotation as quaternion)
        R = tag.pose_R
        T = np.eye(4)
        T[:3, :3] = R
        quaternion = quaternion_from_matrix(T)
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        # Set scale for the axes (text size)
        marker.scale.x = 0.2  # Size of the text
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        # Set color (you can change this)
        marker.color.r = 1.0  # Red
        marker.color.g = 1.0  # Green
        marker.color.b = 0.0  # Blue
        marker.color.a = 1.0  # Alpha (opacity)

        # Add text with position and orientation
        text = f"Pos: ({tag.pose_t[0][0]:.2f}, {tag.pose_t[1][0]:.2f}, {tag.pose_t[2][0]:.2f})\n"
       
        marker.text = text

        # Publish the marker
        self.marker_publisher.publish(marker)
    
def main(args=None):
    rclpy.init(args=args)
    node = WebcamAprilTagNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

