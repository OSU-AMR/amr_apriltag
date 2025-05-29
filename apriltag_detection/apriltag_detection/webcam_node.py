# Filename: webcam_node_1080p_mjpeg.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import traceback
import sys
import time # Import time for potential delays

class WebcamNode(Node):
    def __init__(self):
        super().__init__('webcam_node')
        self.get_logger().info("Initializing Webcam Node - Requesting 1080p MJPEG...")

        # Parameters
        self.declare_parameter('display_enabled', False)

        # --- Modified Settings ---
        start_camera_index = 0
        topic_name = 'camera/image_raw'
        self.frame_id = 'camera_link'
        desired_width = 1280
        desired_height = 720
        desired_fps = 25.0 # Use float 
        # --- End Modified Settings ---

        self.cap = self.find_and_configure_camera(
            start_camera_index,
            desired_width,
            desired_height,
            desired_fps
        )

        if self.cap is None:
            self.get_logger().error("FATAL: No suitable webcam found or configured. Exiting.")
            # rclpy might not be fully initialized here for shutdown yet
            sys.exit(1) # Exit directly if camera setup fails critically

        self.get_logger().info(f"Publishing to: {topic_name}")
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, topic_name, 10)

        # --- Adjust Timer based on FPS ---
        timer_period = 1.0 / desired_fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Timer set to capture at ~{desired_fps:.2f} FPS (period: {timer_period:.4f}s)")
        # --- End Adjust Timer ---


    def find_and_configure_camera(self, start_index, req_width, req_height, req_fps):
        self.get_logger().info(f"Searching for working camera from index {start_index}...")
        for index in range(start_index, start_index + 9): # Check next 10 indices
            self.get_logger().info(f"Attempting index {index}...")
            # Try specifying the backend explicitly if default doesn't work well
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2) # Or cv2.CAP_DSHOW on Windows, etc.
            #cap = cv2.VideoCapture(index)

            if not cap.isOpened():
                self.get_logger().warn(f"Index {index}: Failed to open.")
                cap.release()
                continue

            self.get_logger().info(f"Index {index}: Opened successfully. Attempting configuration...")

            # --- Explicitly request MJPEG format ---
            fourcc_code = cv2.VideoWriter_fourcc(*'MJPG')
            set_fourcc = cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
            if not set_fourcc:
                self.get_logger().warn(f"Index {index}: Failed to set FOURCC to MJPG.")
                # continue # Optionally skip if MJPG is strictly required
            # --- End Set FOURCC ---

            # --- Set Resolution and FPS ---
            set_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(req_width))
            set_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(req_height))
            set_fps = cap.set(cv2.CAP_PROP_FPS, float(req_fps))
            # --- End Set Resolution and FPS ---

            # Some drivers require a short delay or a first read for settings to apply
            time.sleep(0.2) # Small delay

            # --- Verify actual settings ---
            actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc_str = "".join([chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            self.get_logger().info(f"Index {index}: Requested: MJPG {req_width}x{req_height} @ {req_fps} FPS")
            self.get_logger().info(f"Index {index}: Actual:   {actual_fourcc_str} ({actual_fourcc_int}) {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
            # --- End Verify Settings ---

            # Check if the core settings were applied (resolution is most critical usually)
            # Allow some tolerance for FPS as it might not be exact
            if actual_width == req_width and actual_height == req_height:
                # Check if MJPG was successfully set if it was attempted
                if set_fourcc and actual_fourcc_int != fourcc_code:
                     self.get_logger().warn(f"Index {index}: FOURCC set request succeeded but reported value differs. Proceeding anyway.")
                elif not set_fourcc and actual_fourcc_int != fourcc_code:
                     self.get_logger().warn(f"Index {index}: Failed to set FOURCC to MJPG, but resolution is correct. Proceeding with format '{actual_fourcc_str}'.")

                self.get_logger().info(f"Index {index}: Found suitable camera and configured successfully.")

                # Perform a test read
                ret, frame = cap.read()
                if ret and frame is not None:
                     self.get_logger().info(f"Index {index}: Initial frame read successful.")
                     return cap
                else:
                     self.get_logger().error(f"Index {index}: Configuration seemed okay, but failed to read initial frame.")
                     cap.release()
                     continue
            else:
                self.get_logger().warn(f"Index {index}: Failed to set desired resolution ({actual_width}x{actual_height} obtained). Releasing.")
                cap.release()
                continue # Try next index

        self.get_logger().error("Failed to find and configure any suitable camera.")
        return None # No suitable camera found/configured

    def timer_callback(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
             self.get_logger().error("Timer callback called but camera is not available.")
             # Consider stopping the timer or attempting recovery
             return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # This can happen occasionally, might not be fatal
            self.get_logger().warn("Failed to read frame from camera.")
            return

        try:
            # OpenCV reads in BGR format by default
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = self.frame_id
            self.image_publisher.publish(ros_image)

            # Check parameter for display
            display_enabled = self.get_parameter('display_enabled').get_parameter_value().bool_value
            if display_enabled:
                # Displaying 1080p might be slow, consider resizing for display only
                # display_frame = cv2.resize(frame, (640, 360)) # Example resize
                # cv2.imshow("Webcam Feed (Resized)", display_frame)
                cv2.imshow("Webcam Feed", frame) # Show full res
                key = cv2.waitKey(1) & 0xFF
                # Optional: Add way to quit cleanly from display window
                # if key == ord('q'):
                #     self.destroy_node()
                #     rclpy.try_shutdown()


        except cv2.error as cv_err:
             self.get_logger().error(f"OpenCV Error during processing/display: {cv_err}")
             # This might indicate a deeper issue with the frame or resources
        except self.bridge.CvBridgeError as bridge_err:
            self.get_logger().error(f"CvBridge Error: {bridge_err}")
        except Exception as e:
            self.get_logger().error(f"Frame processing/publishing failed: {e}")
            self.get_logger().error(traceback.format_exc()) # Log full traceback for unexpected errors

    def destroy_node(self):
        self.get_logger().info("Cleaning up Webcam Node...")
        # Stop the timer first to prevent callbacks during cleanup
        if hasattr(self, 'timer') and self.timer:
            self.timer.cancel()
        # Release camera capture
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.get_logger().info("Releasing video capture device...")
            self.cap.release()
        # Close OpenCV display windows
        cv2.destroyAllWindows()
        self.get_logger().info("Cleanup complete.")
        super().destroy_node() # Call parent cleanup

def main(args=None):
    print("Starting webcam node (1080p MJPEG Attempt)...")
    rclpy.init(args=args)
    node = None
    try:
        node = WebcamNode()
        # Check if node initialization failed (e.g., no camera)
        if node.cap is None:
             print("Node initialization failed (likely camera issue). Shutting down.")
        else:
             rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        # Ensure cleanup happens even if node creation failed partially
        if node is not None:
            node.destroy_node()
        else:
             # If node object wasn't even created, ensure CV windows are closed
             cv2.destroyAllWindows()
        # Ensure rclpy shuts down
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS shutdown complete.")

if __name__ == '__main__':
    main()