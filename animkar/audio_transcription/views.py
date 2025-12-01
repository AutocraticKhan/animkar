from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.urls import reverse, reverse_lazy
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.generic import DeleteView
from django.utils import timezone
from django.core.paginator import Paginator
from project_manager.models import Project
from .models import AudioTranscription, WordTimestamp
from .transcription_utils import (
    load_or_download_model,
    transcribe_audio_with_accurate_timestamps,
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
        # Get audio file path
        audio_path = transcription.get_audio_file_path()

        # Transcribe audio with improved pipeline
        words_data = transcribe_audio_with_accurate_timestamps(
            audio_path=audio_path,
            language=transcription.language,
            silence_thresh=-40,
            min_silence_len=500,
            confidence_threshold=0.3
        )

        # Calculate statistics
        total_words = len(words_data)
        high_confidence_words = len([w for w in words_data if w['confidence'] >= 0.5])
        avg_confidence = sum(w['confidence'] for w in words_data) / total_words if total_words > 0 else 0
        audio_duration = words_data[-1]['end_time_s'] if words_data else 0

        # Update transcription metadata
        transcription.status = 'completed'
        transcription.duration_seconds = audio_duration
        transcription.total_words = total_words
        transcription.high_confidence_words = high_confidence_words
        transcription.average_confidence = avg_confidence
        transcription.model_name = MODEL_NAME
        transcription.processed_at = timezone.now()
        transcription.save()

        # Create WordTimestamp instances with transliteration data
        for word_data in words_data:
            WordTimestamp.objects.create(
                transcription=transcription,
                word=word_data['word'],  # Transliterated word
                transliterated_word=word_data['original_script'] if word_data['transliterated'] else None,
                start_time_seconds=word_data['start_time_s'],
                end_time_seconds=word_data['end_time_s'],
                confidence=word_data['confidence'],
                duration_seconds=word_data['duration_s'],
                is_high_confidence=word_data['confidence'] >= 0.5
            )

    except Exception as e:
        transcription.status = 'failed'
        transcription.save()
        print(f"Transcription failed: {e}")
        import traceback
        traceback.print_exc()

def transcription_detail(request, transcription_id):
    """Display detailed transcription results"""
    transcription = get_object_or_404(AudioTranscription, pk=transcription_id)

    # Get word timestamps with pagination
    word_timestamps = transcription.word_timestamps.all()
    paginator = Paginator(word_timestamps, 50)  # 50 words per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'transcription': transcription,
        'page_obj': page_obj,
        'word_timestamps': word_timestamps,  # Keep full list for CSV export
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



class AudioTranscriptionDeleteView(DeleteView):
    model = AudioTranscription
    template_name = 'audio_transcription/transcription_confirm_delete.html'
    success_url = reverse_lazy('project_list')

    def get_success_url(self):
        return reverse('project_detail', kwargs={'pk': self.object.project.pk})

    def delete(self, request, *args, **kwargs):
        messages.success(self.request, f'Deleted transcription "{self.get_object().original_filename}".')
        return super().delete(request, *args, **kwargs)
