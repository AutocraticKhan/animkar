import random
from PIL import Image
from .services_config import ANIMATION_EFFECTS, FINAL_CANVAS_SIZE

# Global dictionary to store animation state per media file
animation_state_map = {}

def ease_in_out_cubic(t):
    """
    Easing function for smooth acceleration and deceleration.
    t: progress from 0.0 to 1.0
    Returns: eased value from 0.0 to 1.0
    """
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def get_animation_state(media_path, total_frames):
    """
    Get or create animation state for a specific media file.
    Returns a dictionary with animation parameters.
    """
    if media_path not in animation_state_map:
        # Randomly select an animation effect
        effect_type = random.choice(ANIMATION_EFFECTS)

        animation_state_map[media_path] = {
            'effect_type': effect_type,
            'total_frames': total_frames,
            'current_frame': 0
        }
    return animation_state_map[media_path]

def apply_animation_effect(background_img, animation_state):
    """
    Apply ultra-smooth zoom or pan animation effect to the background image.
    Each full cycle (out→in or left→right→left) takes ~140 frames (70 per direction).
    """
    effect_type = animation_state['effect_type']
    current_frame = animation_state['current_frame']

    # Define cycle lengths
    frames_per_half_cycle = 70  # 70 frames for one direction
    total_cycle_frames = frames_per_half_cycle * 2  # full ping-pong = 140 frames

    # Create a looping progress (0 → 1 → 0)
    cycle_progress = (current_frame % total_cycle_frames) / frames_per_half_cycle
    if cycle_progress > 1:
        cycle_progress = 2 - cycle_progress  # reverses back smoothly
    eased = ease_in_out_cubic(cycle_progress)

    canvas_width, canvas_height = FINAL_CANVAS_SIZE

    if effect_type == 'zoom_out_in':
        # Zoom between 1.0 and 0.85 slowly and smoothly
        scale_factor = 1.0 - (eased * 0.15)
        new_width = int(background_img.width * scale_factor)
        new_height = int(background_img.height * scale_factor)
        scaled_img = background_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        result = Image.new("RGBA", FINAL_CANVAS_SIZE, (0, 0, 0, 255))
        paste_x = (canvas_width - new_width) // 2
        paste_y = (canvas_height - new_height) // 2
        result.paste(scaled_img, (paste_x, paste_y))
        return result

    elif effect_type == 'pan_left_to_right':
        # Pan slowly back and forth across ~7.5% width range
        max_movement = int(canvas_width * 0.075)
        offset_x = int(eased * max_movement)
        offset_x = max(0, min(offset_x, background_img.width - canvas_width))
        cropped = background_img.crop((offset_x, 0, offset_x + canvas_width, canvas_height))
        return cropped

    elif effect_type == 'pan_right_to_left':
        # Reverse direction, same smoothness
        max_movement = int(canvas_width * 0.075)
        offset_x = int(max_movement - (eased * max_movement))
        offset_x = max(0, min(offset_x, background_img.width - canvas_width))
        cropped = background_img.crop((offset_x, 0, offset_x + canvas_width, canvas_height))
        return cropped

    # Default fallback (no animation)
    return background_img
