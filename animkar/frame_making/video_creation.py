import os
import subprocess
import shutil

def create_video_ffmpeg(frames_folder, output_video, fps, audio_file):
    """
    Generate the video with audio using FFmpeg from generated frames.
    """
    # --- 1️⃣ Ensure frames exist ---
    frames = sorted([
        f for f in os.listdir(frames_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and f.startswith("frame_")
    ])

    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_folder}")

    print(f"Found {len(frames)} frames in '{frames_folder}'")

    # --- 2️⃣ Generate the video with audio using FFmpeg ---
    # %04d matches frame_0000.png, frame_0001.png, etc.
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                                # overwrite output if exists
        "-framerate", str(fps),
        "-i", os.path.join(frames_folder, "frame_%04d.png"),
        "-i", audio_file,                    # input audio
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",                       # encode audio
        "-shortest",                         # stop when shortest input ends (audio/video)
        output_video
    ]

    print(f"🎥 Creating video '{output_video}' with audio '{audio_file}'...")
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Done! Video created successfully: {output_video}")

    # --- 3️⃣ Cleanup frames ---
    for f in frames:
        os.remove(os.path.join(frames_folder, f))
    print(f"🧹 Cleaned up {len(frames)} frames from '{frames_folder}'")
