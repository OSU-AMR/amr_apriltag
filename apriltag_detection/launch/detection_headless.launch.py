import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='apriltag_detection',
            executable='apriltag_node',
            name='apriltag_node'
        ),
        
         Node(
            package='apriltag_detection',
            executable='webcam_node',
            name='webcam_node',
            parameters=[{"show_processed_video": False}]
        ),
        
    ])

