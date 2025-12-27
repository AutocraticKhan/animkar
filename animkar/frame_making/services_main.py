import os
import random
from tqdm import tqdm
from .services_config import get_output_folder, IMAGES_DIR, FPS
from .services_generation import generate_composite_image
from .video_creation import create_video_ffmpeg

def process_frames_for_transcription(transcription, frame_annotations):
    """
    Main processing loop for generating frames and video from database records.
    """
    # Get output folder for this project
    output_folder = get_output_folder(transcription.project.id)
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # --- Pre-determine avatar visibility for 'no_avatar' mode ---
    avatar_visibility_map = {}
    no_avatar_segments = []
    
    # Get all frames designated as 'no_avatar' sorted chronologically
    no_avatar_frames = sorted([fa.frame_number for fa in frame_annotations if fa.mode.strip().lower() == 'no_avatar'])

    if no_avatar_frames:
        show_avatar_state = True 
        current_block_duration = random.randint(70, 90)
        frames_in_current_block = 0
        current_segment_start = None

        for i, frame_num in enumerate(no_avatar_frames):
            if frames_in_current_block >= current_block_duration:
                if not show_avatar_state and current_segment_start is not None:
                    no_avatar_segments.append({
                        'start_frame': current_segment_start,
                        'end_frame': no_avatar_frames[i - 1],
                        'total_frames': i - no_avatar_frames.index(current_segment_start)
                    })
                
                show_avatar_state = not show_avatar_state
                if not show_avatar_state:
                    current_segment_start = frame_num
                
                frames_in_current_block = 0
                current_block_duration = random.randint(70, 90)

            avatar_visibility_map[frame_num] = show_avatar_state
            frames_in_current_block += 1
        
        if not show_avatar_state and current_segment_start is not None:
            no_avatar_segments.append({
                'start_frame': current_segment_start,
                'end_frame': no_avatar_frames[-1],
                'total_frames': len(no_avatar_frames) - no_avatar_frames.index(current_segment_start)
            })

    print(f"Processing {len(frame_annotations)} frames...")

    blink_counter = 0
    current_body_image_path = None
    current_body_filename = None
    last_posture = None
    frames_to_hold_image = 0

    for fa in tqdm(frame_annotations, desc="Processing frames"):
        posture = fa.body_posture
        if posture != last_posture or frames_to_hold_image <= 0:
            body_images_dir = IMAGES_DIR / 'body' / 'character_1' / posture
            try:
                available_images = [f for f in os.listdir(body_images_dir) if f.lower().endswith('.png')]
                if not available_images:
                    print(f"Warning: No PNG files in directory: {body_images_dir}. Skipping.")
                    continue
                current_body_filename = random.choice(available_images)
                current_body_image_path = body_images_dir / current_body_filename
                frames_to_hold_image = random.randint(70, 90)
                last_posture = posture
            except FileNotFoundError:
                print(f"Error: Dir not found for posture '{posture}': {body_images_dir}. Skipping frame {fa.frame_number}.")
                continue
        
        frames_to_hold_image -= 1
        if fa.blink:
            blink_counter += 1
        else:
            blink_counter = 0
            
        if current_body_image_path:
            generate_composite_image(
                fa, blink_counter, str(current_body_image_path),
                current_body_filename, avatar_visibility_map, no_avatar_segments, no_avatar_frames, output_folder
            )

    print("All frames processed successfully!")

    # Create the video
    audio_path = transcription.get_audio_file_path()
    output_video_path = output_folder / f"project_{transcription.project.id}_video.mp4"

    create_video_ffmpeg(str(output_folder), str(output_video_path), FPS, audio_path)

    return str(output_video_path)
