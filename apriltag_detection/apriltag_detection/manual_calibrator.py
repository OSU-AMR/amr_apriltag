import cv2
import numpy as np
import yaml
import urllib.request
import time

# --- Configuration ---
# URL of your IP camera stream
CAMERA_URL = 'http://10.254.239.1:5000/video_feed' 
# The size of your checkerboard (number of inner corners)
CHECKERBOARD_SIZE = (7, 10) # (points_per_row, points_per_col)
# The size of a square on your checkerboard in meters
SQUARE_SIZE = 0.024 # meters

# --- Script ---

def main():
    print("--- Manual Camera Calibration ---")
    print(f"Connecting to stream at {CAMERA_URL}...")
    
    # Create a VideoCapture object
    stream = urllib.request.urlopen(CAMERA_URL)
    bytes_buffer = bytes()

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1,2)
    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images_captured = 0
    
    print("\nInstructions:")
    print("  - Show the checkerboard to the camera from different angles and distances.")
    print("  - Press 'c' to capture an image. You need at least 15 good images.")
    print("  - Press 'q' to quit and perform the calibration.")

    while True:
        bytes_buffer += stream.read(1024)
        a = bytes_buffer.find(b'\xff\xd8')
        b = bytes_buffer.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_buffer[a:b+2]
            bytes_buffer = bytes_buffer[b+2:]
            
            if jpg:
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Display the resulting frame
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Images Captured: {images_captured}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('IP Camera Feed - Press "c" to capture, "q" to calibrate', display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

                    # If found, add object points, image points (after refining them)
                    if ret == True:
                        images_captured += 1
                        print(f"Checkerboard found! Image {images_captured} captured.")
                        objpoints.append(objp)
                        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                        imgpoints.append(corners2)
                    else:
                        print("Checkerboard not found in the current frame. Try a different angle.")

                elif key == ord('q'):
                    if images_captured < 15:
                        print("\nNot enough images captured! Please capture at least 15 images.")
                    else:
                        break

    cv2.destroyAllWindows()

    print("\nPerforming calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n--- CALIBRATION COMPLETE ---")
    print("Copy the following parameters into your ROS2 launch file or params file.")

    # Create a dictionary to hold the calibration data in a ROS-friendly format
    calibration_data = {
        'image_width': gray.shape[1],
        'image_height': gray.shape[0],
        'camera_name': 'ip_camera',
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': mtx.flatten().tolist()
        },
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {
            'rows': 1,
            'cols': 5,
            'data': dist.flatten().tolist()
        },
        'rectification_matrix': { # Not used by apriltag node, but good practice
            'rows': 3,
            'cols': 3,
            'data': np.eye(3).flatten().tolist()
        },
        'projection_matrix': { # Not used by apriltag node, but good practice
            'rows': 3,
            'cols': 4,
            'data': np.hstack([mtx, np.zeros((3, 1))]).flatten().tolist()
        }
    }

    # Print as YAML
    print("\n--- YAML Output ---")
    print(yaml.dump(calibration_data, default_flow_style=False))
    
    fx = mtx[0,0]
    fy = mtx[1,1]
    cx = mtx[0,2]
    cy = mtx[1,2]
    k1, k2, p1, p2, k3 = dist[0]
    
    print("\n--- Individual Parameters for apriltag_node.py ---")
    print(f"ip_cam.fx: {fx}")
    print(f"ip_cam.fy: {fy}")
    print(f"ip_cam.cx: {cx}")
    print(f"ip_cam.cy: {cy}")
    print(f"ip_cam.k1: {k1}")
    print(f"ip_cam.k2: {k2}")
    print(f"ip_cam.p1: {p1}")
    print(f"ip_cam.p2: {p2}")
    print(f"ip_cam.k3: {k3}")


if __name__ == '__main__':
    main()