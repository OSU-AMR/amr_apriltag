#! /usr/bin/env python3
# amr_apriltag/apriltag_detection/ip_webcam_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class IpWebcamNode(Node):
    def __init__(self):
        super().__init__('ip_webcam_node')
        
        # --- Parameters ---
        self.declare_parameter('camera_url', 'http://10.254.239.1/video_feed')
        self.declare_parameter('topic_name', '/ip_camera/image_raw')
        self.declare_parameter('frequency', 30.0) # Hz
        # --- ADD THIS PARAMETER TO CONTROL THE DISPLAY ---
        self.declare_parameter('show_video', False)
        
        # Get parameters
        camera_url = self.get_parameter('camera_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        frequency = self.get_parameter('frequency').get_parameter_value().double_value
        # --- GET THE NEW PARAMETER ---
        self.show_video = self.get_parameter('show_video').get_parameter_value().bool_value
        
        # --- Node Initialization ---
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.timer = self.create_timer(1.0/frequency, self.timer_callback)
        self.bridge = CvBridge()
        
        # --- Camera Capture ---
        self.cap = cv2.VideoCapture(camera_url)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open video stream at URL: {camera_url}')
            raise SystemExit # Or handle more gracefully
            
        self.get_logger().info(f'Successfully opened camera stream and publishing to {topic_name}')
        if self.show_video:
            self.get_logger().info('Raw video display is enabled.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV image to ROS2 Image message and publish
            ros_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image_msg.header.stamp = self.get_clock().now().to_msg() # Good practice to add a timestamp
            self.publisher_.publish(ros_image_msg)

            # --- ADD THIS BLOCK TO DISPLAY THE VIDEO ---
            if self.show_video:
                cv2.imshow("IP Camera Feed", frame)
                cv2.waitKey(1) # This is crucial for the window to update

        else:
            self.get_logger().warn('Failed to grab frame from IP camera.')

    def destroy_node(self):
        # --- Proper cleanup of resources ---
        self.get_logger().info("Releasing camera and destroying node.")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    ip_webcam_node = IpWebcamNode()
    try:
        rclpy.spin(ip_webcam_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup is now handled in destroy_node
        ip_webcam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()