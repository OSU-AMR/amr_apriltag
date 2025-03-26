import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from apriltag_msgs.msg import AprilTagDetectionArray

class ApriltagPosePublisher(Node):
    def __init__(self):
        super().__init__('apriltag_pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, '/apriltag_poses', 10)
        self.subscription = self.create_subscription(
            AprilTagDetectionArray,
            '/apriltag_marker',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Iterate over all the detected AprilTags in the array
        for detection in msg.detections:
            tag_id = detection.id[0]  # Get tag ID
            tag_pose = detection.pose.pose  # Get pose (position + orientation)

            # Create a PoseStamped message to publish the pose
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'camera_link'  # or the appropriate frame
            
            pose_msg.pose.position.x = tag_pose.position.x
            pose_msg.pose.position.y = tag_pose.position.y
            pose_msg.pose.position.z = tag_pose.position.z
            
            pose_msg.pose.orientation.x = tag_pose.orientation.x
            pose_msg.pose.orientation.y = tag_pose.orientation.y
            pose_msg.pose.orientation.z = tag_pose.orientation.z
            pose_msg.pose.orientation.w = tag_pose.orientation.w

            # Publish the pose of the tag
            self.publisher_.publish(pose_msg)
            self.get_logger().info(f'Publishing Pose for AprilTag {tag_id}')

def main(args=None):
    rclpy.init(args=args)
    node = ApriltagPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

