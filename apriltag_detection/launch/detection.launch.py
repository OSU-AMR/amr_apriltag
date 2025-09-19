import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='amr_apriltag',
            executable='apriltag_node',
            name='apriltag_node'
        ),
        
         Node(
            package='amr_apriltag',
            executable='webcam_node',
            name='webcam_node'
        ),
        
        ExecuteProcess(
            cmd=['rviz2', '-d', 'install/apriltag_detection/share/apriltag_detection/rviz/apriltag_rviz.rviz'],
            output='screen'
        )
    ])

