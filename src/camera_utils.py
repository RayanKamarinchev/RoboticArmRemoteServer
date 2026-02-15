import cv2
import cv2.aruco as aruco
import numpy as np
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_PARAMETERS_DIR = os.path.join(BASE_DIR, "cam_parameters")
CAMERA_MATRIX_DIR = os.path.join(CAM_PARAMETERS_DIR, "camera_matrix.npy")
DIST_COEFFS_DIR = os.path.join(CAM_PARAMETERS_DIR, "dist_coeffs.npy")

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def get_marker_positions(marker_size, marker_spacing, rows=5, cols=4):
    grid = np.arange(0, 20).reshape((rows,cols))
    marker_grid = [[[(marker_size+marker_spacing)*x, (marker_size+marker_spacing)*y, 0] for x in range(0,cols)] for y in range(0,rows)]
    marker_positions = {grid[y][x]: marker_grid[y][x] for y in range(grid.shape[0]) for x in range(grid.shape[1])}
    return marker_positions

def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def undistort_image(image):
    cameraMatrix = np.load(CAMERA_MATRIX_DIR)
    distCoeffs   = np.load(DIST_COEFFS_DIR)
    
    undistorted = cv2.undistort(image, cameraMatrix, distCoeffs)
    img = cv2.rotate(undistorted, cv2.ROTATE_90_CLOCKWISE)
    return img


def get_camera_position(img, marker_positions, marker_size):
    img_copy = img.copy()
    camera_matrix, dist_coeffs = get_camera_matrix_and_dist_coeffs()

    marker_corners, img_points = get_all_markers(img, marker_positions, marker_size)
    
    if marker_corners is None or img_points is None:
        print("No markers detected or matched.")
        return img_copy, None, None, None, None, None

    for i, point in enumerate(img_points):
        if i%2==0:
            cv2.circle(img_copy, tuple(point.astype(int)), 5, (255,0,0), -1)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=marker_corners,
        imagePoints=img_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    proj, _ = cv2.projectPoints(marker_corners, rvec, tvec, camera_matrix, dist_coeffs)
    for p in proj.reshape(-1, 2):
        cv2.circle(img_copy, tuple(p.astype(int)), 3, (0, 0, 255), -1)

    cv2.drawFrameAxes(img_copy, camera_matrix, dist_coeffs, rvec, tvec, 0.2)

    error = np.mean(np.linalg.norm(img_points - proj.reshape(-1, 2), axis=1))
    print("Reprojection error:", error)

    #swapping boards x and y
    transform = np.array([
        [0, 1,  0],
        [1, 0,  0],
        [0, 0, -1]
    ])
    
    R, _ = cv2.Rodrigues(rvec)
    camera_position = transform @ (-R.T @ tvec)
    camera_position = camera_position.flatten()
    
    board_y = np.array([0, 1, 0])

    board_y_cam = R @ board_y

    v_xy = board_y_cam[:2]

    v_xy /= np.linalg.norm(v_xy)

    cam_angle = -np.arctan2(v_xy[1], v_xy[0])
    # print(np.degrees(cam_angle))

    return img_copy, camera_position, cam_angle, R, rvec, tvec

def get_all_markers(img, marker_positions, marker_size=0.036):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(ids) == 0:
        return None, None

    marker_corners_3d = np.array([
        [0, 0, 0],                  # top-left
        [marker_size, 0, 0],        # top-right
        [marker_size, marker_size, 0],  # bottom-right
        [0, marker_size, 0]         # bottom-left
    ], dtype=np.float32)

    marker_corners = []
    image_points = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in marker_positions:
            continue

        origin = np.array(marker_positions[marker_id], dtype=np.float32)
        obj_points = marker_corners_3d + origin

        marker_corners.append(obj_points)
        image_points.append(corners[i][0].astype(np.float32))

    if len(marker_corners) == 0:
        return None, None

    marker_corners = np.vstack(marker_corners)
    image_points = np.vstack(image_points)

    return marker_corners, image_points

def get_depth_map(img_left, img_right):
    markerPositions = get_marker_positions()
    objPoints, imgPointsLeft = get_all_markers(img_left, markerPositions)
    objPoints, imgPointsRigth = get_all_markers(img_right, markerPositions)
    
    cameraMatrix = np.load(CAMERA_MATRIX_DIR)
    distCoeffs   = np.load(DIST_COEFFS_DIR)
    
    h, w = img_left.shape[:2]
    imageSize = (w, h)
    
    rms, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objPoints,
        imgPointsLeft,
        imgPointsRigth,
        cameraMatrix, distCoeffs,
        cameraMatrix, distCoeffs,
        imageSize,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    print("RMS: ", rms)
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        imageSize,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )
    
    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_16SC2
    )
    
    rectL = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
    
    for y in range(0, rectL.shape[0], 40):
        cv2.line(rectL, (0,y), (rectL.shape[1],y), (0,255,0), 1)
    
    cv2.imwrite("rectified_img.jpg", rectL)
    print("RMS:", rms)
    print("T:", T.ravel())


def get_camera_matrix_and_dist_coeffs():
    camera_matrix = np.load(CAMERA_MATRIX_DIR)
    dist_coeffs = np.load(DIST_COEFFS_DIR)
    return camera_matrix, dist_coeffs