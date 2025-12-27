from ..models import EmotionAnnotation, BodyPostureAnnotation, ModeAnnotation, CharacterAnnotation, BackgroundAnnotation, FrameAnnotation


def calculate_coverage_status(transcription):
    """
    Calculate coverage status for all annotation types for a given transcription.
    Returns a dictionary with completion status for each annotation type.
    """
    total_words = transcription.word_timestamps.count()

    coverage_status = {}

    # Emotion coverage
    emotion_count = EmotionAnnotation.objects.filter(word_timestamp__transcription=transcription).count()
    coverage_status['emotion_complete'] = emotion_count == total_words

    # Body posture coverage
    body_count = BodyPostureAnnotation.objects.filter(word_timestamp__transcription=transcription).count()
    coverage_status['body_complete'] = body_count == total_words

    # Mode coverage
    mode_count = ModeAnnotation.objects.filter(word_timestamp__transcription=transcription).count()
    coverage_status['mode_complete'] = mode_count == total_words

    # Characters coverage
    characters_count = CharacterAnnotation.objects.filter(word_timestamp__transcription=transcription).count()
    coverage_status['characters_complete'] = characters_count == total_words

    # Background coverage
    background_count = BackgroundAnnotation.objects.filter(word_timestamp__transcription=transcription).count()
    coverage_status['background_complete'] = background_count == total_words

    # Additional completion statuses
    coverage_status['transcription_complete'] = total_words > 0  # Transcription is complete if it has words

    # Check if frames are saved to database
    coverage_status['frames_saved'] = FrameAnnotation.objects.filter(transcription=transcription).exists()

    # Check if any media is added to frames
    coverage_status['media_added'] = FrameAnnotation.objects.filter(transcription=transcription).exclude(media='').exists()

    # Check if video exists
    from frame_making.services_config import get_output_folder
    video_filename = f"project_{transcription.project.id}_transcription_{transcription.id}_video.mp4"
    video_path = get_output_folder(transcription.project.id) / video_filename
    coverage_status['video_exists'] = video_path.exists()

    return coverage_status
