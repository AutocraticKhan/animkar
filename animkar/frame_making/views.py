from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404, render
from audio_transcription.models import AudioTranscription
from annotation.models import FrameAnnotation
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
        video_path = process_frames_for_transcription(transcription, frame_annotations)
        video_url = f"/media/projects/{transcription.project.id}/videos/project_{transcription.project.id}_video.mp4"
        return render(request, 'frame_making/video_generated.html', {
            'transcription': transcription,
            'video_url': video_url,
            'video_path': video_path
        })
    except Exception as e:
        return render(request, 'frame_making/video_generated.html', {
            'error': str(e),
            'transcription': transcription
        })

def download_video_view(request, transcription_id):
    """
    View to download the generated video.
    """
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)
    video_filename = f"project_{transcription.project.id}_video.mp4"
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
    video_filename = f"project_{transcription.project.id}_video.mp4"
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

    # Check if video exists
    video_filename = f"project_{transcription.project.id}_video.mp4"
    from .services_config import get_output_folder
    video_path = get_output_folder(transcription.project.id) / video_filename

    if not video_path.exists():
        return render(request, 'frame_making/video_view.html', {
            'error': 'Video not found. Please generate the video first.',
            'transcription': transcription
        })

    video_url = f"/media/projects/{transcription.project.id}/videos/{video_filename}"
    return render(request, 'frame_making/video_view.html', {
        'transcription': transcription,
        'video_url': video_url,
        'video_path': video_path
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

    try:
        # Start video generation asynchronously
        import threading
        def generate_video_async():
            try:
                process_frames_for_transcription(transcription, frame_annotations)
            except Exception as e:
                print(f"Video generation failed: {e}")

        thread = threading.Thread(target=generate_video_async)
        thread.daemon = True
        thread.start()

        return JsonResponse({'success': True, 'message': 'Video generation started'})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
