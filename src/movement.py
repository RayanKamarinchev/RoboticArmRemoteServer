import math
import os
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

class Angle:
    def __init__(self, *, rad=None, deg=None):
        if rad is not None:
            self._rad = float(rad)
        elif deg is not None:
            self._rad = np.radians(deg)
        else:
            raise ValueError("Provide radians or degrees")

    def __float__(self):
        return self._rad
    
    def __array__(self, dtype=None):
        return np.array(self._rad, dtype=dtype)

    @property
    def rad(self):
        return self._rad

    @property
    def deg(self):
        return np.degrees(self._rad)

    def __add__(self, other):
        return Angle(rad=self._rad + self._to_rad(other))

    def __sub__(self, other):
        return Angle(rad=self._rad - self._to_rad(other))

    def __mul__(self, scalar):
        return Angle(rad=self._rad * scalar)

    def __truediv__(self, scalar):
        return Angle(rad=self._rad / scalar)

    def __eq__(self, other):
        return np.isclose(self._rad, self._to_rad(other))

    def __repr__(self):
        return f"Angle({self.deg:.2f}°)"

    def _to_rad(self, value):
        if isinstance(value, Angle):
            return value.rad
        return float(value)
    
@dataclass
class Angles:
    alpha: Angle
    beta: Angle
    gamma: Angle
    theta: Angle
    psi: Angle
    
    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

# Robot arm segment lengths in cm
a = 0.12
b = 0.127
c = 0.13
e = 0.07#68-70
camera_offset_len = 0.016
baseElevation = 0.132 # 30-32
# Initial joint angles in degrees
delta = np.radians(75) #around 78
offsets = [Angle(deg=-3), Angle(deg=205), Angle(deg=65), Angle(deg=155)]

def get_initial_angles():
    alpha = Angle(deg=100) + offsets[0]
    beta = offsets[1] - Angle(deg=100)
    gamma = Angle(deg=6) + offsets[2]
    theta = Angle(deg=0)
    psi = Angle(deg=0)
    return Angles(alpha, beta, gamma, theta, psi)

def world_to_servo_angles(angles: Angles):
    #[theta, alpha, beta, psi, gamma]
    servo_angles = [Angle(deg=30)-angles.theta, angles.alpha-offsets[0], offsets[1]-angles.beta, offsets[3]-angles.psi, angles.gamma-offsets[2]]
    return [np.round(angle.deg).astype(int) for angle in servo_angles]

def servo_to_world_angle(servo_angles: np.ndarray, idx):
    index_to_name_map = ["theta", "alpha", "beta", "psi", "gamma"]
    
    world_angles = [Angle(deg=30)-Angle(deg=servo_angles[0]),
                        Angle(deg=servo_angles[1])+offsets[0],
                        offsets[1]-Angle(deg=servo_angles[2]),
                        offsets[3]-Angle(deg=servo_angles[3]),
                        Angle(deg=servo_angles[4])+offsets[2]]
    return (index_to_name_map[idx], world_angles[idx])


def get_arm_vectors(alpha, beta, gamma, psi): 
    l2_angle = alpha + beta - np.pi
    #arm1
    l1 = a * np.array([np.cos(alpha), 0, np.sin(alpha)])
    #arm2
    l2 = b * np.array([np.cos(l2_angle),0, np.sin(l2_angle)])
    #combined base arm
    lb = l1 + l2
    #angle above the xy plane (first z then x)
    phi = np.arctan2(lb[2], lb[0])
    #head arm angle relative to x
    angle_sum = alpha + beta + gamma
    #head elevation from base arm
    epsilon = angle_sum + np.pi/2 - phi
    #head arm components
    lh_x = c * np.array([np.cos(angle_sum), 0, np.sin(angle_sum)])
    lh_y = c * np.array([np.sin(epsilon) * np.cos(phi), np.cos(epsilon), np.sin(epsilon) * np.sin(phi)])
    lh = np.cos(psi) * lh_x + np.sin(psi) * lh_y
    
    return lb, lh

def get_gripper_coords_and_cam_rotation_from_arm(angles):
    if isinstance(angles, Angles):
        alpha, beta, gamma, theta, psi = (angles.alpha, angles.beta, angles.gamma, angles.theta, angles.psi)
    else:
        alpha, beta, gamma, theta, psi = angles
    psi=0
    
    lb, lh = get_arm_vectors(alpha, beta, gamma, psi)
    
    arm_vector = lb + lh
    #calculating the rotation angle around the z axis
    position = rotate_vec(arm_vector, theta)
    #calculate camera angles and rotation
    base_arm_rotated = rotate_vec(lb, theta)
    head_rotated = position-base_arm_rotated
    azimuth = np.arctan2(head_rotated[1], head_rotated[0])
    radius = np.sqrt(head_rotated[0]**2 + head_rotated[1]**2)
    elevation = np.arctan2(head_rotated[2], radius)
    rotation = psi
    
    position[2] += baseElevation
    return (position, [azimuth, elevation, rotation])

# def rotate_vec(vec, theta):
#     vec_already_rotated = np.arctan2(vec[1], vec[0])
#     full_rotation_angle = vec_already_rotated + theta
#     radius = vec[0]/np.cos(vec_already_rotated)
#     return np.array([radius*np.cos(full_rotation_angle), radius*np.sin(full_rotation_angle), vec[2]])
def rotate_vec(vec, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return R @ vec


def get_rotation_matrix(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([
        [-c, s, 0],
        [ s, c, 0],
        [0, 0, 1]
    ])
    
def get_translation(world_pos, arm_pos, alpha):
    R = get_rotation_matrix(alpha)
    return world_pos - R @ arm_pos

def transform_arm_to_world_coords(p1, alpha, t):
    R = get_rotation_matrix(alpha)
    return R @ p1 + t

def transform_world_to_arm_coords(p2, alpha, t):
    R = get_rotation_matrix(alpha)
    return R.T @ (p2 - t)

def get_move_angles(target_coords, translation, rotation_angle, starting_angles = get_initial_angles(), is_in_world_frame = True):
    x, y, z = target_coords
    starting_angles = [starting_angles.alpha.rad, starting_angles.beta.rad, starting_angles.gamma.rad, starting_angles.theta.rad, starting_angles.psi.rad]
    print("Target: ", target_coords)
    print("Starting angles: ", starting_angles)
    print("Initial angles: ", get_initial_angles())
    def objective(vars):
        position_pred, camera_angles = get_gripper_coords_and_cam_rotation_from_arm(vars)
        #convert from arm coordinate system to from board(in space) coordinate system
        if(is_in_world_frame):
            position_pred = transform_arm_to_world_coords(position_pred, rotation_angle, translation)
        
        position_diff = np.linalg.norm(position_pred-np.array([x,y,z]))
        # position_diff = np.linalg.norm(position_in_space-np.array([x,-y,z]))
        #TODO camera difference
        # penalty = np.linalg.norm(vars - initial_angles)
        penalty = np.abs(vars[2]-np.pi/2)*3 + np.abs(vars[3]) + np.abs(vars[4])*3
        # penalty = 5*vars[0]-np.round(vars[0])
        return position_diff + 1e-4 * penalty

    bounds = [
        (offsets[0].rad, offsets[0].rad+np.pi),       # alpha
        (offsets[1].rad-np.pi, offsets[1].rad),       # beta
        (offsets[2].rad, offsets[2].rad+np.pi),       # gamma
        (np.radians(-150), np.radians(-150+180)),   # theta
        # (np.radians(-30), np.radians(-30+180))      # psi
        (0, 0)      # psi
    ]

    result = minimize(
        objective,
        starting_angles,
        bounds=bounds,
    )
    
    alpha, beta, gamma, theta, psi  = result.x
    angles_output = Angles(Angle(rad=alpha), Angle(rad=beta), Angle(rad=gamma), Angle(rad=theta), Angle(rad=psi))
    print("Angles: ", angles_output)
    # if result.success or result.fun < 1e-6:
    #     alpha, beta, gamma, theta, psi  = result.x
    # else:
    #     print("Optimization failed:", result.message)

    return angles_output

def conv_camera_coords_to_gripper_coords(camera_coords, angles, coordinate_systems_angle):
    alpha, beta, gamma, theta, psi = (angles.alpha, angles.beta, angles.gamma, angles.theta, angles.psi)
    
    _, arm_head = get_arm_vectors(alpha, beta, gamma, psi)
    _, camera_vector_direction = get_arm_vectors(alpha, beta, gamma + delta, psi)
    camera_vector_normalized = camera_vector_direction * e / c
    camera_offset = np.cross(arm_head, camera_vector_normalized)
    # print(camera_vector_normalized, "cam vec")
    # print(arm_head, "arm")
    caemra_offset_normalized = camera_offset / np.linalg.norm(camera_offset) * camera_offset_len
    # print(caemra_offset_normalized, "Camera offset")
    # translation_vec = rotate_vec(-caemra_offset_normalized-camera_vector_normalized+arm_head, -coordinate_systems_angle)
    disposition_vec_in_arm_system = -caemra_offset_normalized-camera_vector_normalized+arm_head
    
    print("Angle is this" , coordinate_systems_angle)
    co, si = np.cos(coordinate_systems_angle), np.sin(coordinate_systems_angle)
    mat = np.array([
        [-co, si, 0],
        [si,  co, 0],
        [0,  0, 1]
    ])
    print("Disposition vec in arm system", disposition_vec_in_arm_system)
    disposition_vec = mat @ disposition_vec_in_arm_system
    print("Disposition vec ", disposition_vec)
    gripper_position = camera_coords + disposition_vec
    return gripper_position
