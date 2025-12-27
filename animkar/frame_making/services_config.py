import os
import json
import pandas as pd
from django.conf import settings
from pathlib import Path

# --- Constants ---
FRAME_MAKING_DIR = Path(settings.BASE_DIR) / 'frame_making'
DATA_DIR = FRAME_MAKING_DIR / 'data'
IMAGES_DIR = FRAME_MAKING_DIR / 'images'
def get_output_folder(project_id):
    """Get the output folder for a specific project"""
    return Path(settings.MEDIA_ROOT) / 'projects' / str(project_id) / 'videos'
TOP_PADDING = 150
FINAL_CANVAS_SIZE = (1920, 1080)
FPS = 30
ANIMATION_EFFECTS = ['zoom_out_in', 'pan_left_to_right', 'pan_right_to_left']

# --- Load JSON Data ---
def load_json(filename):
    path = DATA_DIR / filename
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

M_eye = load_json("eyes_for_M.json")
L_eye = load_json("eyes_for_L.json")
R_eye = load_json("eyes_for_R.json")
M_mouth = load_json("mouth_for_M.json")
L_mouth = load_json("mouth_for_L.json")
R_mouth = load_json("mouth_for_R.json")
phonemes = load_json("phonemes_json.json")

L_body = load_json("body_for_L.json")
M_body = load_json("body_for_M.json")
R_body = load_json("body_for_R.json")

big_side_placement = load_json("Big_side.json")
small_side_placement = load_json("Small_side.json")
big_center_placement = load_json("Big_center.json")

media_json = load_json("media.json")

# --- Mappings ---
eye_data_map = {'M': M_eye, 'L': L_eye, 'R': R_eye}
mouth_data_map = {'M': M_mouth, 'L': L_mouth, 'R': R_mouth}
head_on_body_map = {'L': L_body, 'M': M_body, 'R': R_body}

mode_to_character_placement_map = {
    'big_side': big_side_placement,
    'small_side': small_side_placement,
    'big_center': big_center_placement,
    'no_avatar': small_side_placement
}

MODE_TO_MEDIA_KEY = {
    'big_side': 'media_BS',
    'small_side': 'media_SS',
    'big_side_vertical': 'media_BSV'
}
