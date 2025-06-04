import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='amr_apriltag',
            executable='apriltag_node.py',
            name='apriltag_node'
        ),
        
         Node(
            package='amr_apriltag',
            executable='webcam_node.py',
            name='webcam_node',
            parameters=[{"show_processed_video": False}]
        ),
        
    ])

