from flask import Flask, render_template, request, jsonify
import cv2
import json
import os
import numpy as np
import base64
from flask.json.provider import DefaultJSONProvider

from src.box_detection import get_box_coordinates, Box
from src.camera_utils import decode_image, get_camera_position, get_marker_positions, get_camera_matrix_and_dist_coeffs
from src.movement import (get_move_angles, get_initial_angles,
                          conv_camera_coords_to_gripper_coords, get_gripper_coords_and_cam_rotation_from_arm,
                          transform_arm_to_world_coords, transform_world_to_arm_coords,
                          get_translation, world_to_servo_angles, servo_to_world_angle)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

app = Flask(__name__)

MARKER_SIZE=0.036
MARKER_SPACING=0.005
BASELINE=0.02

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LATEST_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "latest.jpg")
DATA_FILE = 'program_data/program_data.json'
serial_commands = []
latest_img = None
detected_boxes = []
counter = 0
world_angles = get_initial_angles()
initial_gripper_position_in_arm, _ = get_gripper_coords_and_cam_rotation_from_arm(world_angles)
current_gripper_position_in_world = None
translation = None
system_angle = None

#Domain Logic
def get_position_command(target_coords):
    global translation, system_angle, world_angles, current_gripper_position_in_world
    target_coords = np.array(target_coords)
    current_gripper_position_in_world = target_coords
    
    world_angles = get_move_angles(target_coords, translation, system_angle, world_angles, True)
    servo_angles = world_to_servo_angles(world_angles)
    return ["move", *servo_angles]

def get_box_distance(box: Box):
    return np.linalg.norm(box.grab_point - current_gripper_position_in_world)

def get_nearest_box(boxes):
    return min(boxes, key=get_box_distance)

def prepare_instructions(img):
    global world_angles, serial_commands, current_gripper_position_in_world, detected_boxes, translation, system_angle
    
    _, camera_position, coordinate_systems_angle, R, rvec, tvec = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
    if(camera_position is None):
        return jsonify({"error": "No aruco board"}), 400

    print("Coordinate systems angle: ", np.degrees(coordinate_systems_angle))
    current_gripper_position_in_world = conv_camera_coords_to_gripper_coords(camera_position, world_angles, coordinate_systems_angle)
    arm_angle = np.arctan2(initial_gripper_position_in_arm[1], initial_gripper_position_in_arm[0])
    print("Arm angle: ", np.degrees(arm_angle))
    system_angle = coordinate_systems_angle-arm_angle
    
    translation = get_translation(current_gripper_position_in_world, initial_gripper_position_in_arm, system_angle)
    
    camera_matrix, dist_coeffs = get_camera_matrix_and_dist_coeffs()
    
    detected_boxes, boxes_overlay = get_box_coordinates(img, camera_position, R, camera_matrix, dist_coeffs, rvec, tvec)
    
    data = load_data()
    instructions = data["instructions"][::-1]
    groups = data["groups"]
    serial_commands = []
        
    for instruction in instructions:
        instruction_type = instruction["type"]
        instruction_params = instruction["params"]
        serial_command = None
        if instruction_type == "wait":
            serial_command = ["wait", instruction_params["time"]]
            
        elif instruction_type == "grip":
            serial_command = ["grip", 1]
            
        elif instruction_type == "ungrip":
            serial_command = ["grip", 0]
            
        elif instruction_type == "initial":
            world_angles = get_initial_angles()
            serial_command = ["initial"]
            
        elif instruction_type == "go_box":
            box = next((box for box in detected_boxes if box.id == instruction_params["box_id"]), None)
            if box is not None:
                serial_command = get_position_command(box.grab_point)
                
        elif instruction_type == "go_group_box":
            group = next((group for group in groups if group["id"] == instruction_params["group_id"]), None)
            if group is None:
                continue
            boxes = [box for box in detected_boxes if box.id in group["boxes"]]
            if boxes is None or len(boxes) == 0:
                continue
            box = get_nearest_box(boxes)
            serial_command = get_position_command(box.grab_point)
        elif instruction_type == "go_nearest_box":
            if len(detected_boxes) == 0:
                continue
            box = get_nearest_box(detected_boxes)
            if box is not None:
                serial_command = get_position_command(box.grab_point)
        elif instruction_type == "go_pos":
            serial_command = get_position_command([instruction_params['x']/100, instruction_params['y']/100, instruction_params['z']/100])
        elif instruction_type == "go_group_location":
            group = next((group for group in groups if group["id"] == instruction_params["group_id"]), None)
            if group is None:
                continue
            serial_command = get_position_command([group["location"]['x']/100, group["location"]['y']/100, group["location"]['z']/100,])
            
        if serial_command is not None:
            serial_commands.append(serial_command)
            
    serial_commands.append(["initial"])
    
    return boxes_overlay
            

@app.route('/get_position', methods=['POST'])
def receive_image():
    global latest_img
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    img = decode_image(file_bytes)
    cv2.imwrite(LATEST_IMAGE_PATH, img)
    
    boxes_overlay = prepare_instructions(img)
    
    _, buffer = cv2.imencode(".jpg", boxes_overlay)

    image_bytes = buffer.tobytes()
    
    latest_img = base64.b64encode(image_bytes).decode('utf-8')
    
    return jsonify({"message": "OK"}), 200

@app.route('/get_movements', methods=['GET'])
def receive_data():
    global serial_commands
    
    if latest_img is None:
        return jsonify({"error": "No camera image yet."}), 400
    
    print("Sending serial commands:", serial_commands)
    return jsonify(serial_commands), 200

@app.route('/api/get_cam_data', methods=['GET'])
def get_cam_data():
    global serial_commands, latest_img, detected_boxes
    
    if latest_img is None:
        return jsonify({"error": "No camera image yet."}), 400
    
    return jsonify({"success": True, "image": latest_img, "boxes": [box.to_dict() for box in detected_boxes]}), 200

# User Interface

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {'groups': [], 'instructions': []}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        data = load_data()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/groups', methods=['GET'])
def get_groups():
    try:
        data = load_data()
        return jsonify({'success': True, 'groups': data.get('groups', [])})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/groups', methods=['POST'])
def add_group():
    try:
        data = load_data()
        new_group = request.json
        
        if 'id' not in new_group:
            new_group['id'] = str(len(data['groups']) + 1)
        
        data['groups'].append(new_group)
        save_data(data)
        
        return jsonify({'success': True, 'group': new_group})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/groups/<group_id>', methods=['PUT'])
def update_group(group_id):
    try:
        data = load_data()
        updated_group = request.json
        
        for i, group in enumerate(data['groups']):
            if group['id'] == group_id:
                data['groups'][i] = updated_group
                save_data(data)
                return jsonify({'success': True, 'group': updated_group})
        
        return jsonify({'success': False, 'error': 'Group not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/groups/<group_id>', methods=['DELETE'])
def delete_group(group_id):
    try:
        data = load_data()
        data['groups'] = [g for g in data['groups'] if g['id'] != group_id]
        save_data(data)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/instructions', methods=['GET'])
def get_instructions():
    try:
        data = load_data()
        return jsonify({'success': True, 'instructions': data.get('instructions', [])})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/instructions', methods=['POST'])
def save_instructions():
    try:
        data = load_data()
        data['instructions'] = request.json.get('instructions', [])
        save_data(data)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.json = NumpyJSONProvider(app)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)