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
        self.declare_parameter('show_video', True) # Defaulting to True for convenience
        
        # --- ADD DISPLAY SIZE PARAMETERS ---
        self.declare_parameter('display_width', 1280)
        self.declare_parameter('display_height', 720)
        
        # Get parameters
        camera_url = self.get_parameter('camera_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.show_video = self.get_parameter('show_video').get_parameter_value().bool_value
        
        # --- GET DISPLAY SIZE PARAMETERS ---
        self.display_width = self.get_parameter('display_width').get_parameter_value().integer_value
        self.display_height = self.get_parameter('display_height').get_parameter_value().integer_value

        # --- Node Initialization ---
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.timer = self.create_timer(1.0/frequency, self.timer_callback)
        self.bridge = CvBridge()
        
        # --- Camera Capture ---
        self.cap = cv2.VideoCapture(camera_url)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open video stream at URL: {camera_url}')
            raise SystemExit 
            
        self.get_logger().info(f'Successfully opened camera stream and publishing to {topic_name}')
        if self.show_video:
            self.get_logger().info(f'Raw video display enabled at {self.display_width}x{self.display_height}.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Publish the full-resolution frame
            ros_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image_msg.header.stamp = self.get_clock().now().to_msg() 
            self.publisher_.publish(ros_image_msg)

            # --- Display the video if enabled ---
            if self.show_video:
                # --- RESIZE THE FRAME FOR DISPLAY ---
                display_frame = cv2.resize(frame, (self.display_width, self.display_height))
                
                # --- SHOW THE RESIZED FRAME ---
                cv2.imshow("IP Camera Feed", display_frame)
                cv2.waitKey(1)

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
        ip_webcam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()