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
            parameters=[{"display_enabled": False}]
        ),
         Node(
            package='amr_apriltag',
            executable='ip_webcam_node.py',
            name='ip_webcam_node',
            parameters=[
                
                {'ip_camera_url': 'http://10.254.239.1/video_feed'},{'show_video': False}
            ]
        ),
        
    ])

