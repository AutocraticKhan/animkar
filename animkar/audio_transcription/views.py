from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils import timezone
from project_manager.models import Project
from .models import AudioTranscription, WordTimestamp
from .transcription_utils import (
    load_or_download_model,
    transcribe_audio_with_word_timestamps,
    MODEL_CACHE_DIR,
    MODEL_NAME
)
import os
import torch
from django.conf import settings

def upload_audio(request, project_id):
    """Handle audio file upload and transcription processing"""
    project = get_object_or_404(Project, pk=project_id)

    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')

        if not audio_file:
            messages.error(request, 'Please select an audio file.')
            return redirect('project_detail', pk=project_id)

        # Validate file extension
        valid_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
        file_ext = os.path.splitext(audio_file.name)[1].lower()

        if file_ext not in valid_extensions:
            messages.error(request, f'Unsupported file format. Supported formats: {", ".join(valid_extensions)}')
            return redirect('project_detail', pk=project_id)

        # Validate file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if audio_file.size > max_size:
            messages.error(request, 'File size too large. Maximum size is 100MB.')
            return redirect('project_detail', pk=project_id)

        try:
            # Create AudioTranscription instance
            transcription = AudioTranscription.objects.create(
                project=project,
                audio_file=audio_file,
                original_filename=audio_file.name,
                file_size=audio_file.size,
                status='processing'
            )

            # Start transcription process
            process_transcription(transcription)

            messages.success(request, 'Audio file uploaded and transcription started. This may take a few minutes.')
            return redirect('project_detail', pk=project_id)

        except Exception as e:
            messages.error(request, f'Error uploading file: {str(e)}')
            return redirect('project_detail', pk=project_id)

    return redirect('project_detail', pk=project_id)

def process_transcription(transcription):
    """Process the transcription in the background"""
    try:
        # Load Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_or_download_model(MODEL_NAME, MODEL_CACHE_DIR, device)

        if model is None:
            transcription.status = 'failed'
            transcription.save()
            return

        # Get audio file path
        audio_path = transcription.get_audio_file_path()

        # Transcribe audio
        result = transcribe_audio_with_word_timestamps(
            audio_path,
            model,
            language=transcription.language,
            confidence_threshold=0.3,
            use_sample_accurate=True
        )

        # Update transcription metadata
        transcription.status = 'completed'
        transcription.duration_seconds = result['audio_duration']
        transcription.total_words = result['total_words']
        transcription.high_confidence_words = result['high_confidence_words']
        transcription.average_confidence = result['average_confidence']
        transcription.processed_at = timezone.now()
        transcription.save()

        # Create WordTimestamp instances
        for word_data in result['word_data']:
            WordTimestamp.objects.create(
                transcription=transcription,
                word=word_data['Word'],
                start_time_seconds=word_data['Start Time (s)'],
                end_time_seconds=word_data['End Time (s)'],
                confidence=word_data['Confidence'],
                duration_seconds=word_data['End Time (s)'] - word_data['Start Time (s)'],
                is_high_confidence=word_data['Confidence'] >= 0.5
            )

    except Exception as e:
        transcription.status = 'failed'
        transcription.save()
        print(f"Transcription failed: {e}")

def transcription_detail(request, transcription_id):
    """Display detailed transcription results"""
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)

    # Get word timestamps
    word_timestamps = transcription.word_timestamps.all()

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'project': transcription.project,
    }

    return render(request, 'audio_transcription/transcription_detail.html', context)

@require_POST
def retry_transcription(request, transcription_id):
    """Retry failed transcription"""
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)

    if transcription.status == 'failed':
        transcription.status = 'pending'
        transcription.save()

        # Process again
        process_transcription(transcription)

        messages.success(request, 'Transcription retry started.')
    else:
        messages.warning(request, 'Transcription is not in failed state.')

    return redirect('transcription_detail', transcription_id=transcription_id)
