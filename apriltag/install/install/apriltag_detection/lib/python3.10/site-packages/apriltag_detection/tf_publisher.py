import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from dt_apriltags import Detector
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_matrix

class AprilTagPose(Node):
    def __init__(self):
        super().__init__('apriltag_pose')

        self.at_detector = Detector(families='tag36h11')

        # Camera parameters (YOUR VALUES)
        self.camera_matrix = np.array([
            [1.33509253e+03, 0, 9.34526997e+02],
            [0, 1.34225732e+03, 5.83016486e+02],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([-0.0487205  , 0.15446917 , 0.00098029 ,-0.00573289 ,-0.20121614])
        self.camera_params = [639, 639,
                              320, 240]

        self.tag_size = 0.101  # Tag size in meters

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.image_subscriber = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray, estimate_tag_pose=True,
                                        camera_params=self.camera_params,
                                        tag_size=self.tag_size)

        for tag in tags:
           # print(f"Tag ID: {tag.tag_id}, Translation: {tag.pose_t}, Rotation: {tag.pose_R}")  # Debugging
            self.publish_tf(tag)

    def publish_tf(self, tag):
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = "camera_link"  # Or your desired frame
        t2.child_frame_id = f"tag_{tag.tag_id}"

        t2.transform.translation.x = tag.pose_t[0][0]
        t2.transform.translation.y = tag.pose_t[1][0]
        t2.transform.translation.z = tag.pose_t[2][0]

        R = tag.pose_R

        # Correct Quaternion Calculation:
        T = np.eye(4)
        T[:3, :3] = R
        quaternion = quaternion_from_matrix(T)

        t2.transform.rotation.x = quaternion[0]
        t2.transform.rotation.y = quaternion[1]
        t2.transform.rotation.z = quaternion[2]
        t2.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t2)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
