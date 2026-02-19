from box_detection import get_box_coordinates
from camera_utils import get_camera_position, get_marker_positions, get_camera_matrix_and_dist_coeffs
import cv2
MARKER_SIZE=0.036
MARKER_SPACING=0.005
img = cv2.imread("./uploads/latest.jpg")
camera_matrix, dist_coeffs = get_camera_matrix_and_dist_coeffs()
_, camera_position, coordinate_systems_angle, R, rvec, tvec = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
get_box_coordinates(img, camera_position, R, camera_matrix, dist_coeffs, rvec, tvec)