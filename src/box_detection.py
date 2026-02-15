import cv2
import numpy as np
from ultralytics import YOLO
import os
import cv2.aruco as aruco
from dataclasses import dataclass, asdict

BOX_CODE_SIZE = 0.03
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model/best.pt')

@dataclass
class Box:
    id: int
    grab_point: list
    width: float
    length: float
    height: float
    
    def to_dict(self):
        d = asdict(self)
        if isinstance(d["grab_point"], np.ndarray):
            d["grab_point"] = d["grab_point"].tolist()
        return d

def rescale_masks(masks, img_shape):
    scale = img_shape[0] / masks.shape[1]
    new_masks = []
    
    for mask in masks:
        mask = (mask * 255).astype(np.uint8)

        mask = cv2.resize(
            mask,
            (int(mask.shape[1] * scale), int(mask.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST
        )
        new_masks.append(mask)
    return new_masks

def get_polygons_from_masks(masks, epsilon=0.012):
    polygons = []
    
    for mask in masks:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        contour = max(contours, key=cv2.contourArea)

        eps = epsilon * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, eps, True)
        polygon = polygon.squeeze()

        polygons.append(polygon)

    return polygons

def draw_masks_and_polygons(img, masks, polygons):
    overlay = img.copy()

    color = np.array([255, 0, 0], dtype=np.uint8)
    for mask in masks:
        overlay[mask > 0] = (
            0.5 * overlay[mask > 0] + 0.5 * color
        ).astype(np.uint8)
    for polygon in polygons:
        cv2.polylines(overlay, [polygon.astype(np.int32)], True, (0,255,0), 2)

    cv2.imwrite("result.png", overlay)
    return overlay

def detect_box_codes(img, boxes):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    
    box_data = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.imwrite("qr.jpg", img[y1:y2, x1:x2])
        
        corners, ids, _ = detector.detectMarkers(img[y1:y2, x1:x2])
        if ids is None or len(ids) == 0:
            box_data.append(None)
            continue
        
        corners_in_image = [[x + x1, y + y1] for (x,y) in corners[0][0]]
        box_data.append({"corners": corners_in_image, "id": ids[0][0]})
    
    return box_data

def get_polygon_centroid(polygon):
    return np.mean(polygon, axis=0)

def get_height_from_box_code(box_corners, camera_matrix, dist_coeffs, camera_position, R):
    box_code_2d = undistort_points(box_corners, camera_matrix, dist_coeffs)
    rays = get_world_rays_from_img_points(box_code_2d, camera_matrix, R)

    d0 = rays[0]
    d1 = rays[1]
    #distance between the projections of the 2 qr code rays on the z plane
    K_factor = np.linalg.norm(
        d1 / d1[2] - d0 / d0[2]
    )

    h = camera_position[2] - BOX_CODE_SIZE / K_factor
    return h
    
    
def undistort_points(points, camera_matrix, dist_coeffs):
    points_2d = np.array(points, dtype=np.float32).reshape(-1,1,2)
    undistorted = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs, P=camera_matrix)
    undistorted = undistorted.reshape(-1,2)
    return undistorted

def get_world_rays_from_img_points(points, camera_matrix, R):
    rays = []
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    for p in points:
        p_h = np.array([p[0], p[1], 1.0])
        ray_cam = camera_matrix_inv @ p_h
        ray_cam /= np.linalg.norm(ray_cam)
        ray = R.T @ ray_cam
        rays.append(ray)

    rays = np.array(rays)
    return rays

def clean_mask(mask, kernel_size=3):
    mask = (mask > 0).astype(np.uint8) * 255
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    return mask_clean

def get_cuboid_info(top_side_world_points):
    print("Top side world points:", top_side_world_points)
    len1 = np.linalg.norm(top_side_world_points[0] - top_side_world_points[1])
    len2 = np.linalg.norm(top_side_world_points[1] - top_side_world_points[2])
    width, length, width_vec, length_vec = (None, None, None, None)
    if len1 > len2:
        length = len1
        width = len2
        length_vec = top_side_world_points[0] - top_side_world_points[1]
        width_vec = top_side_world_points[2] - top_side_world_points[1]
    else:
        length = len2
        width = len1
        length_vec = top_side_world_points[2] - top_side_world_points[1]
        width_vec = top_side_world_points[0] - top_side_world_points[1]

    print("Cuboid width vector:", width_vec)
    print("Cuboid length vector:", length_vec)
    
    print("Cuboid width:", width)
    print("Cuboid length:", length)
    grab_point = top_side_world_points[1] + 0.5 * width_vec + 0.5 * length_vec - [0,0,top_side_world_points[1][2] * 0.7]
    print("Cuboid grab point:", grab_point)
    
    return grab_point, width, length

def undistort_img(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted, new_camera_matrix

def image_to_world_undistorted1(u, v, Z_known, K, rvec, tvec):
    transform = np.array([
        [0, 1,  0],
        [1, 0,  0],
        [0, 0, -1]
    ])
    
    # pts = np.array([[[u, v]]], dtype=np.float32)
    # pts_norm = cv2.undistortPoints(pts, K, dist_coeffs)
    # ray_c = np.array([[pts_norm[0,0,0]],
    #                 [pts_norm[0,0,1]],
    #                 [1.0]])
    ray_c = np.linalg.inv(K) @ np.array([[u], [v], [1.0]])

    R, _ = cv2.Rodrigues(rvec)

    ray_board = R.T @ ray_c
    C_board = -R.T @ tvec

    s = (Z_known - C_board[2, 0]) / ray_board[2, 0]
    P_board = C_board + s * ray_board

    P_board = transform @ P_board
    
    return P_board.flatten()

def get_box_coordinates(img, camera_position, R, camera_matrix, dist_coeffs, rvec, tvec):
    model = YOLO(MODEL_DIR)
    img, new_camera_matrix = undistort_img(img, camera_matrix, dist_coeffs)
    result = model.predict(source=img)[0]
    
    if(result.masks is None):
        return []

    masks = result.masks.data.cpu().numpy()
    masks = rescale_masks(masks, img.shape)
    new_masks = []
    for mask in masks:
        mask = clean_mask(mask, kernel_size=25)
        new_masks.append(mask)
        

    polygons = get_polygons_from_masks(new_masks, epsilon=0.017)
    overlay = draw_masks_and_polygons(img, new_masks, polygons)
    
    boxes = result.boxes.data.cpu().numpy()
    boxes_codes_info = detect_box_codes(img, boxes)
    print(len(boxes_codes_info), "Box codes detected")

    h, w = img.shape[:2]
    camera_center = np.array([w/2, h/2]) #TODO cam angle not 90
    boxes_info = []
    for i, polygon in enumerate(polygons):
        #we know that the furthest point from the camera center is the top of the box, and so are its 2 adjacent points
        furthest_point = np.argmax([np.linalg.norm(p - camera_center) for p in polygon])
        top_side_points = [polygon[(furthest_point-1)%len(polygon)], polygon[furthest_point], polygon[(furthest_point+1)%len(polygon)]]
        
        for (x, y) in top_side_points:
            cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imwrite("result.png", overlay)
        box_code_info = boxes_codes_info[i]
        print(box_code_info)
        if box_code_info is None:
            continue
        
        cuboid_height = get_height_from_box_code(box_code_info["corners"], new_camera_matrix, dist_coeffs, camera_position, R)
        print("Cuboid height:", cuboid_height)
        print("Box id:", box_code_info["id"])

        top_side_world_points = [image_to_world_undistorted1(p[0], p[1], -cuboid_height, new_camera_matrix, rvec, tvec) for p in top_side_points]
            
        grab_point, width, length = get_cuboid_info(top_side_world_points)
        boxes_info.append(Box(box_code_info["id"], grab_point, width, length, cuboid_height))
        
    return boxes_info, overlay
    