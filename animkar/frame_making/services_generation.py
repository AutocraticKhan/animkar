import os
import traceback
import pandas as pd
from PIL import Image
from .services_config import (
    TOP_PADDING, FINAL_CANVAS_SIZE, IMAGES_DIR,
    eye_data_map, mouth_data_map, head_on_body_map,
    mode_to_character_placement_map, MODE_TO_MEDIA_KEY,
    phonemes, media_json
)
from .services_animation import get_animation_state, apply_animation_effect

def optimize_image(img, target_w, target_h):
    """
    Optimize image by resizing to fit inside target dimensions while maintaining aspect ratio,
    then pad with transparent background to exactly match target size.
    """
    img = img.convert("RGBA")
    w, h = img.size

    # Calculate scaling factor to fit inside target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize maintaining aspect ratio
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create transparent background
    new_img = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))

    # Center the resized image
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    new_img.paste(resized, (offset_x, offset_y))

    return new_img

def generate_composite_image(frame_annotation, blink_counter, current_body_image_path, body_filename, avatar_visibility_map, no_avatar_segments, all_no_avatar_frames, output_folder):
    """
    Generates a frame using a multi-layer system.
    frame_annotation: A FrameAnnotation model instance.
    output_folder: Path to save the generated frames
    """
    frame_index = frame_annotation.frame_number
    try:
        # =========================================================================
        # LAYER 0: CREATE THE FINAL CANVAS & DETERMINE LOGIC PATH
        # =========================================================================
        final_canvas = Image.new("RGBA", FINAL_CANVAS_SIZE, (0, 0, 0, 0))
        mode = frame_annotation.mode.strip().lower()

        # Determine if the character should be rendered based on the mode and visibility map.
        if mode == 'no_avatar':
            should_render_character = avatar_visibility_map.get(frame_index, False)
        else:
            should_render_character = True  # All other modes always show the character

        # =========================================================================
        # LAYER 0.5: HANDLE BACKGROUND WITH ANIMATION
        # =========================================================================
        if mode == 'no_avatar':
            media_input = str(frame_annotation.media).strip() if frame_annotation.media else ""
            background_input = str(frame_annotation.background).strip() if frame_annotation.background else ""
            background_applied = False
            
            if media_input and media_input != 'None':
                # Map media path. In your original code it was images/ + media_input
                media_image_path = IMAGES_DIR / media_input
                if media_image_path.exists():
                    # Find the current no_avatar block
                    no_avatar_frames_list = all_no_avatar_frames

                    if frame_index in no_avatar_frames_list:
                        # Find the start of the current continuous block
                        current_index = no_avatar_frames_list.index(frame_index)
                        current_block_start = frame_index
                        
                        # Go backwards to find block start
                        temp_idx = current_index
                        while temp_idx > 0 and no_avatar_frames_list[temp_idx - 1] == no_avatar_frames_list[temp_idx] - 1:
                            temp_idx -= 1
                            current_block_start = no_avatar_frames_list[temp_idx]

                        # Find block end
                        current_block_end = frame_index
                        temp_idx = current_index
                        while temp_idx < len(no_avatar_frames_list) - 1 and no_avatar_frames_list[temp_idx + 1] == no_avatar_frames_list[temp_idx] + 1:
                            temp_idx += 1
                            current_block_end = no_avatar_frames_list[temp_idx]

                        total_block_frames = current_block_end - current_block_start + 1
                        segment_key = str(media_image_path)

                        animation_state = get_animation_state(segment_key, total_block_frames)
                        animation_state['current_frame'] += 1

                        # Load the background image
                        background_img = Image.open(media_image_path).convert("RGBA")

                        # Ensure image is at least canvas size (scale up if needed)
                        if background_img.width < FINAL_CANVAS_SIZE[0] or background_img.height < FINAL_CANVAS_SIZE[1]:
                            scale_factor = max(
                                FINAL_CANVAS_SIZE[0] / background_img.width,
                                FINAL_CANVAS_SIZE[1] / background_img.height
                            )
                            new_width = int(background_img.width * scale_factor * 1.15)  # Add 15% for movement
                            new_height = int(background_img.height * scale_factor * 1.15)
                            background_img = background_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        # Apply animation effect
                        animated_bg = apply_animation_effect(background_img, animation_state)
                        final_canvas.paste(animated_bg, (0, 0))
                        background_applied = True
                    else:
                        # Fallback: no animation
                        background_img = Image.open(media_image_path).convert("RGBA")
                        if background_img.size != FINAL_CANVAS_SIZE:
                            background_img = background_img.resize(FINAL_CANVAS_SIZE, Image.Resampling.LANCZOS)
                        final_canvas.paste(background_img, (0, 0))
                        background_applied = True
                else:
                    print(f"Warning for Frame {frame_index}: In 'no_avatar' mode, media file not found at '{media_image_path}'. Falling back to Background column.")

            # Fallback to Background column if media failed or empty
            if not background_applied and background_input:
                if background_input.startswith('#'):
                    try:
                        background_img = Image.new("RGBA", FINAL_CANVAS_SIZE, background_input)
                        final_canvas.paste(background_img, (0, 0))
                        background_applied = True
                    except ValueError:
                        print(f"Warning for Frame {frame_index}: Invalid hex color '{background_input}'.")
                elif background_input.lower().endswith(('.png', '.jpg', '.jpeg')):
                    background_image_path = IMAGES_DIR / background_input
                    if background_image_path.exists():
                        background_img = Image.open(background_image_path).convert("RGBA")
                        if background_img.size != FINAL_CANVAS_SIZE:
                            background_img = background_img.resize(FINAL_CANVAS_SIZE, Image.Resampling.LANCZOS)
                        final_canvas.paste(background_img, (0, 0))
                        background_applied = True
                else:
                    # Keyword background
                    background_image_path = IMAGES_DIR / "background" / background_input.lower() / "character_1" / "wall.png"
                    if background_image_path.exists():
                        background_img = Image.open(background_image_path).convert("RGBA")
                        if background_img.size != FINAL_CANVAS_SIZE:
                            background_img = background_img.resize(FINAL_CANVAS_SIZE, Image.Resampling.LANCZOS)
                        final_canvas.paste(background_img, (0, 0))
                        background_applied = True
                    else:
                        # Default fallback
                        default_path = IMAGES_DIR / "background" / "default" / "character_1" / "wall.png"
                        if default_path.exists():
                            background_img = Image.open(default_path).convert("RGBA")
                            if background_img.size != FINAL_CANVAS_SIZE:
                                background_img = background_img.resize(FINAL_CANVAS_SIZE, Image.Resampling.LANCZOS)
                            final_canvas.paste(background_img, (0, 0))
                            background_applied = True
        
        else: # Not no_avatar
            background_input = str(frame_annotation.background).strip() if frame_annotation.background else ""
            if background_input:
                background_img = None
                if background_input.startswith('#'):
                    try:
                        background_img = Image.new("RGBA", FINAL_CANVAS_SIZE, background_input)
                    except ValueError: pass
                elif background_input.lower().endswith(('.png', '.jpg', '.jpeg')):
                    background_image_path = IMAGES_DIR / background_input
                    if background_image_path.exists():
                        background_img = Image.open(background_image_path).convert("RGBA")
                else:
                    background_image_path = IMAGES_DIR / "background" / background_input.lower() / "character_1" / "wall.png"
                    if background_image_path.exists():
                        background_img = Image.open(background_image_path).convert("RGBA")

                if background_img:
                    if background_img.size != FINAL_CANVAS_SIZE:
                        background_img = background_img.resize(FINAL_CANVAS_SIZE, Image.Resampling.LANCZOS)
                    final_canvas.paste(background_img, (0, 0))

        # =========================================================================
        # RENDER CHARACTER AND/OR MEDIA OVERLAY IF APPLICABLE
        # =========================================================================
        is_vertical = False
        if should_render_character:
            OVERLAY_MODES = ['big_side', 'small_side', 'big_side_vertical']
            media_val = frame_annotation.media
            if mode in OVERLAY_MODES and media_val and str(media_val).strip() != 'None':
                overlay_image_path_2 = IMAGES_DIR / str(media_val).strip()
                if overlay_image_path_2.exists():
                    overlay_img_2 = Image.open(overlay_image_path_2).convert("RGBA")
                    w, h = overlay_img_2.size
                    is_vertical = h > w
                    media_key = 'media_BSV' if is_vertical else MODE_TO_MEDIA_KEY.get(mode)
                    overlay_data = media_json.get(media_key)
                    if overlay_data:
                        size2 = overlay_data.get("size")
                        cord2_from_json = overlay_data.get("location")
                        optimized_overlay = optimize_image(overlay_img_2, size2[0], size2[1])
                        canvas_width, canvas_height = FINAL_CANVAS_SIZE
                        media_paste_x = (canvas_width // 2) + cord2_from_json[0] - (size2[0] // 2)
                        media_paste_y = (canvas_height // 2) + cord2_from_json[1] - (size2[1] // 2)
                        final_canvas.paste(optimized_overlay, (media_paste_x, media_paste_y), mask=optimized_overlay)

            # =========================================================================
            # LAYER 2: ASSEMBLE AND POSITION THE CHARACTER
            # =========================================================================
            head_direction = frame_annotation.head_direction
            current_eye_json = eye_data_map.get(head_direction)
            current_mouth_json = mouth_data_map.get(head_direction)
            current_head_on_body_json = head_on_body_map.get(head_direction)

            base_head_path = IMAGES_DIR / 'head' / 'character_1' / f'{head_direction}.png'
            head_img = Image.open(base_head_path).convert("RGBA")

            emo = frame_annotation.emotion
            eyedir = frame_annotation.eye_direction
            emo_v2 = 'happy' if emo in ['happy', 'content', 'sarcasm'] else 'sad'
            
            if frame_annotation.intensity:
                emo_asset = f"{emo}_2"
            else:
                emo_asset = emo

            if frame_annotation.blink:
                blink_frames_sequence = [f'{emo_asset}_{eyedir}.png', '02.png', '03.png', '04.png', f'{emo_asset}_{eyedir}.png']
                blinking_frame_file = blink_frames_sequence[min(blink_counter - 1, 4)]
                overlay_eye_image_path = IMAGES_DIR / 'eyes' / 'character_1' / 'side_eyes_blinking' / emo / blinking_frame_file
            else:
                overlay_eye_image_path = IMAGES_DIR / 'eyes' / 'character_1' / 'side_eyes' / emo / f'{emo_asset}_{eyedir}.png'
            
            eye_data = current_eye_json.get(f"{emo}.png")
            if eye_data:
                size = eye_data.get("size")
                cord = eye_data.get("location")
                overlay_eye_img = Image.open(overlay_eye_image_path).transpose(Image.FLIP_LEFT_RIGHT).convert("RGBA")
                resized_eye_overlay = overlay_eye_img.resize((size[0], size[1]), Image.Resampling.LANCZOS)
                head_img.paste(resized_eye_overlay, (cord[0], cord[1]), mask=resized_eye_overlay)

            mouth = frame_annotation.phoneme
            if mouth and mouth in phonemes:
                phoneme_mouth = phonemes.get(mouth).get(emo_v2)
                overlay_mouth_image_path = IMAGES_DIR / 'mouth' / 'character_1' / emo_v2 / phoneme_mouth
                mouth_data = current_mouth_json.get(phoneme_mouth)
                if mouth_data:
                    mouth_size = mouth_data.get("size")
                    mouth_location = mouth_data.get("location")
                    overlay_mouth_img = Image.open(overlay_mouth_image_path).convert("RGBA")
                    resized_mouth_overlay = overlay_mouth_img.resize((mouth_size[0], mouth_size[1]), Image.Resampling.LANCZOS)
                    head_img.paste(resized_mouth_overlay, (mouth_location[0], mouth_location[1]), mask=resized_mouth_overlay)

            body_img = Image.open(current_body_image_path).convert("RGBA")
            character_asset = Image.new("RGBA", (body_img.width, body_img.height + TOP_PADDING), (0, 0, 0, 0))
            character_asset.paste(body_img, (0, TOP_PADDING))
            body_metadata = current_head_on_body_json.get(body_filename)
            if body_metadata:
                head_size = body_metadata.get("size")
                head_location = body_metadata.get("location")
                resized_head = head_img.resize((head_size[0], head_size[1]), Image.Resampling.LANCZOS)
                character_asset.paste(resized_head, (head_location[0], head_location[1] + TOP_PADDING), mask=resized_head)
            
            placement_json = mode_to_character_placement_map.get(mode)
            if mode == 'small_side' and is_vertical:
                placement_json = mode_to_character_placement_map.get('big_side')
            
            if placement_json is not None:
                placement_data = placement_json.get(body_filename)
                if placement_data:
                    target_size = placement_data.get("size")
                    target_location = placement_data.get("location")
                    resized_character = character_asset.resize((target_size[0], target_size[1]), Image.Resampling.LANCZOS)
                    # Grayscale as in original code
                    grayscale_character = resized_character.convert('LA').convert('RGBA')
                    final_canvas.paste(grayscale_character, (target_location[0], target_location[1]), mask=grayscale_character)

        # =========================================================================
        # SAVE THE FINAL RESULT
        # =========================================================================
        os.makedirs(output_folder, exist_ok=True)
        output_filename = f'frame_{frame_index:04d}.png'
        output_path = output_folder / output_filename
        final_canvas.save(str(output_path), "PNG")

    except Exception as e:
        print(f"An error occurred in Frame {frame_index}: {e}")
        traceback.print_exc()
