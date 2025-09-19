import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Static transform: map → usb_camera_link
    static_tf_map_usb = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '2.7', '-1.125', '2.1',                   # translation (x y z)
            '0.188683', '-0.188532', '-0.681218', '0.681760',  # quaternion (x y z w)
            'map',
            'usb_camera_link'
        ]
    )

    # Static transform: map → ip_camera_link
    static_tf_map_ip = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '2.6', '-1.12', '2.14',                   # translation (x y z)
            '0.084629', '0.177775', '0.032499', '0.979887',   # quaternion (x y z w)
            'map',
            'ip_camera_link'
        ]
    )

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
<<<<<<< Updated upstream
        ),
         Node(
            package='amr_apriltag',
            executable='ip_webcam_node.py',
            name='ip_webcam_node',
            parameters=[
                
                {'ip_camera_url': 'http://10.254.239.1/video_feed'},{'show_video': False}
            ]
=======
>>>>>>> Stashed changes
        ),
        Node(
            package='amr_apriltag',
            executable='ip_webcam_node.py',
            name='ip_webcam_node',
            parameters=[
                {'ip_camera_url': 'http://10.254.239.1:5000/video_feed'},
                {'show_video': False}
            ]
        ),
        static_tf_map_usb,
        static_tf_map_ip
    ])
