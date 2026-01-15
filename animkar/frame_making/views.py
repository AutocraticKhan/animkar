from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404, render
from audio_transcription.models import AudioTranscription
from annotation.models import FrameAnnotation
from annotation.services.utils import calculate_coverage_status
from .services_main import process_frames_for_transcription
import os

def generate_video_view(request, transcription_id):
    """
    View to trigger video generation for a specific transcription.
    """
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)

    # Fetch all frame annotations for this transcription
    frame_annotations = FrameAnnotation.objects.filter(transcription=transcription).order_by('frame_number')

    if not frame_annotations.exists():
        return JsonResponse({
            "status": "error",
            "message": "No frame annotations found for this transcription. Please run the annotation process first."
        }, status=400)

    try:
        # Set initial video status
        transcription.video_status = 'processing'
        transcription.video_progress_percentage = 0
        transcription.video_progress_message = 'Starting video generation...'
        transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

        def progress_callback(percentage, message):
            """Update video progress in database"""
            print(f"Progress callback: {percentage}% - {message}")
            transcription.video_progress_percentage = percentage
            transcription.video_progress_message = message
            if percentage >= 100:
                transcription.video_status = 'completed'
                print(f"Setting video status to completed")
            try:
                transcription.save(update_fields=['video_progress_percentage', 'video_progress_message', 'video_status'])
                print(f"Saved progress: status={transcription.video_status}, percentage={transcription.video_progress_percentage}")
            except Exception as e:
                print(f"Error saving progress: {e}")

        video_path = process_frames_for_transcription(transcription, frame_annotations, progress_callback)

        # Mark as completed
        transcription.video_status = 'completed'
        transcription.video_progress_percentage = 100
        transcription.video_progress_message = 'Video generation completed successfully!'
        transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

        video_url = f"/media/projects/{transcription.project.id}/videos/project_{transcription.project.id}_transcription_{transcription.id}_video.mp4"
        return render(request, 'frame_making/video_generated.html', {
            'transcription': transcription,
            'video_url': video_url,
            'video_path': video_path
        })
    except Exception as e:
        transcription.video_status = 'failed'
        transcription.video_progress_percentage = 0
        transcription.video_progress_message = f"Video generation failed: {str(e)}"
        transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

        return render(request, 'frame_making/video_generated.html', {
            'error': str(e),
            'transcription': transcription
        })

def download_video_view(request, transcription_id):
    """
    View to download the generated video.
    """
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)
    video_filename = f"project_{transcription.project.id}_transcription_{transcription.id}_video.mp4"
    from .services_config import get_output_folder
    video_path = get_output_folder(transcription.project.id) / video_filename

    if not video_path.exists():
        return HttpResponse("Video not found", status=404)

    with open(video_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type="video/mp4")
        response['Content-Disposition'] = f'attachment; filename="{video_filename}"'
        return response

def video_status_view(request, transcription_id):
    """
    Check if video exists for a transcription.
    """
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)
    video_filename = f"project_{transcription.project.id}_transcription_{transcription.id}_video.mp4"
    from .services_config import get_output_folder
    video_path = get_output_folder(transcription.project.id) / video_filename

    exists = video_path.exists()
    video_url = f"/media/projects/{transcription.project.id}/videos/{video_filename}" if exists else None

    return JsonResponse({
        'exists': exists,
        'video_url': video_url
    })

def video_view(request, transcription_id):
    """
    View to display the generated video with navigation layout.
    """
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)

    # Calculate coverage status for navigation
    coverage_status = calculate_coverage_status(transcription)

    # First check if video file exists on disk (for existing videos)
    video_filename = f"project_{transcription.project.id}_transcription_{transcription.id}_video.mp4"
    from .services_config import get_output_folder
    video_path = get_output_folder(transcription.project.id) / video_filename

    if video_path.exists():
        # Video file exists, show it
        # Update status to completed if not already set
        if transcription.video_status != 'completed':
            transcription.video_status = 'completed'
            transcription.video_progress_percentage = 100
            transcription.video_progress_message = 'Video generation completed successfully!'
            transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

        # Add timestamp to force browser to reload video (prevent caching)
        import time
        video_url = f"/media/projects/{transcription.project.id}/videos/{video_filename}?t={int(time.time())}"
        return render(request, 'frame_making/video_view.html', {
            'transcription': transcription,
            'video_url': video_url,
            'video_path': video_path,
            'coverage_status': coverage_status
        })

    # Video file doesn't exist, check status
    elif transcription.video_status == 'processing':
        # Video is currently being generated, show progress
        return render(request, 'frame_making/video_view.html', {
            'transcription': transcription,
            'video_processing': True,
            'coverage_status': coverage_status
        })

    else:
        # Video not generated yet, check if frames exist for generation
        frame_annotations = FrameAnnotation.objects.filter(transcription=transcription)
        if not frame_annotations.exists():
            return render(request, 'frame_making/video_view.html', {
                'error': 'No frame annotations found. Please run the annotation process first.',
                'transcription': transcription,
                'coverage_status': coverage_status
            })

        # Start video generation asynchronously
        import threading
        def generate_video_async():
            try:
                # Set initial status
                transcription.video_status = 'processing'
                transcription.video_progress_percentage = 0
                transcription.video_progress_message = 'Starting video generation...'
                transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

                def progress_callback(percentage, message):
                    print(f"Progress callback: {percentage}% - {message}")
                    transcription.video_progress_percentage = percentage
                    transcription.video_progress_message = message
                    if percentage >= 100:
                        transcription.video_status = 'completed'
                        print(f"Setting video status to completed")
                    try:
                        transcription.save(update_fields=['video_progress_percentage', 'video_progress_message', 'video_status'])
                        print(f"Saved progress: status={transcription.video_status}, percentage={transcription.video_progress_percentage}")
                    except Exception as e:
                        print(f"Error saving progress: {e}")

                process_frames_for_transcription(transcription, frame_annotations, progress_callback)

                # Mark as completed
                transcription.video_status = 'completed'
                transcription.video_progress_percentage = 100
                transcription.video_progress_message = 'Video generation completed successfully!'
                transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

            except Exception as e:
                transcription.video_status = 'failed'
                transcription.video_progress_percentage = 0
                transcription.video_progress_message = f"Video generation failed: {str(e)}"
                transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])
                print(f"Video generation failed: {e}")

        thread = threading.Thread(target=generate_video_async)
        thread.daemon = True
        thread.start()

        return render(request, 'frame_making/video_view.html', {
            'transcription': transcription,
            'video_processing': True,
            'coverage_status': coverage_status
        })

def generate_video_ajax_view(request, transcription_id):
    """
    AJAX view to trigger video generation.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)

    # Check if frames exist
    frame_annotations = FrameAnnotation.objects.filter(transcription=transcription)
    if not frame_annotations.exists():
        return JsonResponse({
            'success': False,
            'error': 'No frame annotations found. Please run the annotation process first.'
        }, status=400)

    # Check if already processing
    if transcription.video_status == 'processing':
        return JsonResponse({
            'success': False,
            'error': 'Video generation is already in progress.'
        }, status=400)

    try:
        # Reset video status for recreation
        transcription.video_status = 'pending'
        transcription.video_progress_percentage = 0
        transcription.video_progress_message = 'Preparing to recreate video...'
        transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

        # Start video generation asynchronously
        import threading
        def generate_video_async():
            try:
                # Set initial status
                transcription.video_status = 'processing'
                transcription.video_progress_percentage = 0
                transcription.video_progress_message = 'Starting video generation...'
                transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

                def progress_callback(percentage, message):
                    print(f"Progress callback: {percentage}% - {message}")
                    transcription.video_progress_percentage = percentage
                    transcription.video_progress_message = message
                    if percentage >= 100:
                        transcription.video_status = 'completed'
                        print(f"Setting video status to completed")
                    try:
                        transcription.save(update_fields=['video_progress_percentage', 'video_progress_message', 'video_status'])
                        print(f"Saved progress: status={transcription.video_status}, percentage={transcription.video_progress_percentage}")
                    except Exception as e:
                        print(f"Error saving progress: {e}")

                process_frames_for_transcription(transcription, frame_annotations, progress_callback)

                # Mark as completed
                transcription.video_status = 'completed'
                transcription.video_progress_percentage = 100
                transcription.video_progress_message = 'Video generation completed successfully!'
                transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])

            except Exception as e:
                transcription.video_status = 'failed'
                transcription.video_progress_percentage = 0
                transcription.video_progress_message = f"Video generation failed: {str(e)}"
                transcription.save(update_fields=['video_status', 'video_progress_percentage', 'video_progress_message'])
                print(f"Video generation failed: {e}")

        thread = threading.Thread(target=generate_video_async)
        thread.daemon = True
        thread.start()

        return JsonResponse({'success': True, 'message': 'Video generation started'})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
