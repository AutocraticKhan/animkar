import os
import json
import requests
import shutil
from datetime import datetime
from pathlib import Path
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.serializers.json import DjangoJSONEncoder
from audio_transcription.models import AudioTranscription, WordTimestamp
from .models import EmotionAnnotation, BodyPostureAnnotation, ModeAnnotation, CharacterAnnotation, BackgroundAnnotation, FrameAnnotation
from .services.utils import calculate_coverage_status
from g2p_en import G2p

def annotate_transcription(request, transcription_id):
    """Display the emotion annotation interface for a transcription"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get all word timestamps for this transcription
    word_timestamps = transcription.word_timestamps.all()
    total_words = word_timestamps.count()

    # Calculate coverage status for all annotation types
    coverage_status = calculate_coverage_status(transcription)

    # Get existing annotations as a simple dict for template access
    existing_annotations_dict = {}
    for ann in EmotionAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.emotion

    # Annotate word_timestamps with their emotions
    for wt in word_timestamps:
        wt.emotion = existing_annotations_dict.get(str(wt.id), 'none')

    # Check for complete coverage (for current page)
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'emotion_choices': EmotionAnnotation.EMOTION_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
        'coverage_status': coverage_status,
    }

    return render(request, 'annotation/annotate_transcription.html', context)

@require_POST
@csrf_exempt
def save_annotations(request, transcription_id):
    """Save emotion annotations for words"""
    try:
        data = json.loads(request.body)
        annotations = data.get('annotations', [])

        transcription = get_object_or_404(AudioTranscription, id=transcription_id)

        # Validate that all words are covered
        word_timestamp_ids = set(transcription.word_timestamps.values_list('id', flat=True))
        annotated_ids = set()

        for annotation in annotations:
            word_timestamp_id = annotation.get('word_timestamp_id')
            emotion = annotation.get('emotion')

            if not word_timestamp_id or not emotion:
                return JsonResponse({'error': 'Invalid annotation data'}, status=400)

            if emotion not in dict(EmotionAnnotation.EMOTION_CHOICES):
                return JsonResponse({'error': f'Invalid emotion: {emotion}'}, status=400)

            annotated_ids.add(word_timestamp_id)

            # Create or update annotation
            word_timestamp = get_object_or_404(WordTimestamp, id=word_timestamp_id, transcription=transcription)
            EmotionAnnotation.objects.update_or_create(
                word_timestamp=word_timestamp,
                defaults={'emotion': emotion}
            )

        # Check for missing annotations
        missing_ids = word_timestamp_ids - annotated_ids
        if missing_ids:
            # Auto-assign "content" emotion to missing words
            for word_id in missing_ids:
                word_timestamp = WordTimestamp.objects.get(id=word_id)
                EmotionAnnotation.objects.update_or_create(
                    word_timestamp=word_timestamp,
                    defaults={'emotion': 'content'}
                )

        return JsonResponse({'success': True, 'message': 'Annotations saved successfully'})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_POST
def auto_annotate(request, transcription_id):
    """Use Gemini API to automatically annotate emotions"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get Gemini API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return JsonResponse({'error': 'GEMINI_API_KEY not configured'}, status=500)

    # Get the full text of the transcription
    word_timestamps = transcription.word_timestamps.order_by('start_time_seconds')
    full_text = ' '.join([wt.word for wt in word_timestamps])

    if not full_text.strip():
        return JsonResponse({'error': 'No transcription text available'}, status=400)

    try:
        # Prepare Gemini API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

        prompt = f"""
        Analyze the following text and assign an emotion to each word. Return the result as a JSON array where each element contains the word and its emotion.

        Text: "{full_text}"

        Available emotions: angry, bore, content, glare, happy, sad, sarcasm, worried

        Return format:
        [
            {{"word": "word1", "emotion": "emotion1"}},
            {{"word": "word2", "emotion": "emotion2"}},
            ...
        ]

        Assign exactly one emotion per word. Be comprehensive and cover all words.
        """

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()

        result = response.json()
        ai_response = result['candidates'][0]['content']['parts'][0]['text']

        # Parse the AI response (remove markdown code blocks if present)
        if ai_response.startswith('```json'):
            ai_response = ai_response[7:]
        if ai_response.endswith('```'):
            ai_response = ai_response[:-3]

        ai_annotations = json.loads(ai_response.strip())

        # Map AI annotations to word timestamps
        word_list = [wt.word for wt in word_timestamps]
        annotations_created = 0

        for i, wt in enumerate(word_timestamps):
            if i < len(ai_annotations):
                ai_word = ai_annotations[i].get('word', '').strip()
                emotion = ai_annotations[i].get('emotion', 'content')

                # Basic validation that words match
                if ai_word.lower() == wt.word.lower():
                    EmotionAnnotation.objects.update_or_create(
                        word_timestamp=wt,
                        defaults={
                            'emotion': emotion,
                            'confidence': 0.8  # Assume reasonable confidence from AI
                        }
                    )
                    annotations_created += 1

        return JsonResponse({
            'success': True,
            'message': f'Auto-annotated {annotations_created} words',
            'annotations_created': annotations_created
        })

    except requests.RequestException as e:
        return JsonResponse({'error': f'Gemini API error: {str(e)}'}, status=500)
    except (KeyError, json.JSONDecodeError) as e:
        return JsonResponse({'error': f'Failed to parse AI response: {str(e)}'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

def annotate_mode(request, transcription_id):
    """Display the mode annotation interface for a transcription"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get all word timestamps for this transcription
    word_timestamps = transcription.word_timestamps.all()
    total_words = word_timestamps.count()

    # Calculate coverage status for all annotation types
    coverage_status = calculate_coverage_status(transcription)

    # Get existing annotations as a simple dict for template access
    existing_annotations_dict = {}
    for ann in ModeAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.mode

    # Annotate word_timestamps with their modes
    for wt in word_timestamps:
        wt.mode = existing_annotations_dict.get(str(wt.id), 'none')

    # Check for complete coverage (for current page)
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'mode_choices': ModeAnnotation.MODE_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
        'coverage_status': coverage_status,
    }

    return render(request, 'annotation/annotate_mode.html', context)

@require_POST
@csrf_exempt
def save_mode_annotations(request, transcription_id):
    """Save mode annotations for words"""
    try:
        data = json.loads(request.body)
        annotations = data.get('annotations', [])

        transcription = get_object_or_404(AudioTranscription, id=transcription_id)

        # Validate that all words are covered
        word_timestamp_ids = set(transcription.word_timestamps.values_list('id', flat=True))
        annotated_ids = set()

        for annotation in annotations:
            word_timestamp_id = annotation.get('word_timestamp_id')
            mode = annotation.get('mode')

            if not word_timestamp_id or not mode:
                return JsonResponse({'error': 'Invalid annotation data'}, status=400)

            if mode not in dict(ModeAnnotation.MODE_CHOICES):
                return JsonResponse({'error': f'Invalid mode: {mode}'}, status=400)

            annotated_ids.add(word_timestamp_id)

            # Create or update annotation
            word_timestamp = get_object_or_404(WordTimestamp, id=word_timestamp_id, transcription=transcription)
            ModeAnnotation.objects.update_or_create(
                word_timestamp=word_timestamp,
                defaults={'mode': mode}
            )

        # Check for missing annotations
        missing_ids = word_timestamp_ids - annotated_ids
        if missing_ids:
            # Auto-assign "big_center" mode to missing words (default display mode)
            for word_id in missing_ids:
                word_timestamp = WordTimestamp.objects.get(id=word_id)
                ModeAnnotation.objects.update_or_create(
                    word_timestamp=word_timestamp,
                    defaults={'mode': 'big_center'}
                )

        return JsonResponse({'success': True, 'message': 'Mode annotations saved successfully'})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_POST
def auto_annotate_mode(request, transcription_id):
    """Use Gemini API to automatically annotate display modes"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get Gemini API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return JsonResponse({'error': 'GEMINI_API_KEY not configured'}, status=500)

    # Get the full text of the transcription
    word_timestamps = transcription.word_timestamps.order_by('start_time_seconds')
    full_text = ' '.join([wt.word for wt in word_timestamps])

    if not full_text.strip():
        return JsonResponse({'error': 'No transcription text available'}, status=400)

    try:
        # Prepare Gemini API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

        prompt = f"""
        Analyze the following text and assign a display mode to each word. Return the result as a JSON array where each element contains the word and its display mode.

        Text: "{full_text}"

        Available display modes: big_center, big_side, small_side, no_avatar

        Return format:
        [
            {{"word": "word1", "mode": "mode1"}},
            {{"word": "word2", "mode": "mode2"}},
            ...
        ]

        Assign exactly one display mode per word. Be comprehensive and cover all words.
        """

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()

        result = response.json()
        ai_response = result['candidates'][0]['content']['parts'][0]['text']

        # Parse the AI response (remove markdown code blocks if present)
        if ai_response.startswith('```json'):
            ai_response = ai_response[7:]
        if ai_response.endswith('```'):
            ai_response = ai_response[:-3]

        ai_annotations = json.loads(ai_response.strip())

        # Map AI annotations to word timestamps
        word_list = [wt.word for wt in word_timestamps]
        annotations_created = 0

        for i, wt in enumerate(word_timestamps):
            if i < len(ai_annotations):
                ai_word = ai_annotations[i].get('word', '').strip()
                mode = ai_annotations[i].get('mode', 'big_center')

                # Basic validation that words match
                if ai_word.lower() == wt.word.lower():
                    ModeAnnotation.objects.update_or_create(
                        word_timestamp=wt,
                        defaults={
                            'mode': mode,
                            'confidence': 0.8  # Assume reasonable confidence from AI
                        }
                    )
                    annotations_created += 1

        return JsonResponse({
            'success': True,
            'message': f'Auto-annotated {annotations_created} words with display modes',
            'annotations_created': annotations_created
        })

    except requests.RequestException as e:
        return JsonResponse({'error': f'Gemini API error: {str(e)}'}, status=500)
    except (KeyError, json.JSONDecodeError) as e:
        return JsonResponse({'error': f'Failed to parse AI response: {str(e)}'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

def annotate_body_posture(request, transcription_id):
    """Display the body posture annotation interface for a transcription"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get all word timestamps for this transcription
    word_timestamps = transcription.word_timestamps.all()
    total_words = word_timestamps.count()

    # Calculate coverage status for all annotation types
    coverage_status = calculate_coverage_status(transcription)

    # Get existing annotations as a simple dict for template access
    existing_annotations_dict = {}
    for ann in BodyPostureAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.posture

    # Annotate word_timestamps with their postures
    for wt in word_timestamps:
        wt.posture = existing_annotations_dict.get(str(wt.id), 'none')

    # Check for complete coverage (for current page)
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'posture_choices': BodyPostureAnnotation.POSTURE_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
        'coverage_status': coverage_status,
    }

    return render(request, 'annotation/annotate_body_posture.html', context)

@require_POST
@csrf_exempt
def save_body_posture_annotations(request, transcription_id):
    """Save body posture annotations for words"""
    try:
        data = json.loads(request.body)
        annotations = data.get('annotations', [])

        transcription = get_object_or_404(AudioTranscription, id=transcription_id)

        # Validate that all words are covered
        word_timestamp_ids = set(transcription.word_timestamps.values_list('id', flat=True))
        annotated_ids = set()

        for annotation in annotations:
            word_timestamp_id = annotation.get('word_timestamp_id')
            posture = annotation.get('posture')

            if not word_timestamp_id or not posture:
                return JsonResponse({'error': 'Invalid annotation data'}, status=400)

            if posture not in dict(BodyPostureAnnotation.POSTURE_CHOICES):
                return JsonResponse({'error': f'Invalid posture: {posture}'}, status=400)

            annotated_ids.add(word_timestamp_id)

            # Create or update annotation
            word_timestamp = get_object_or_404(WordTimestamp, id=word_timestamp_id, transcription=transcription)
            BodyPostureAnnotation.objects.update_or_create(
                word_timestamp=word_timestamp,
                defaults={'posture': posture}
            )

        # Check for missing annotations
        missing_ids = word_timestamp_ids - annotated_ids
        if missing_ids:
            # Auto-assign "listen" posture to missing words (default neutral posture)
            for word_id in missing_ids:
                word_timestamp = WordTimestamp.objects.get(id=word_id)
                BodyPostureAnnotation.objects.update_or_create(
                    word_timestamp=word_timestamp,
                    defaults={'posture': 'listen'}
                )

        return JsonResponse({'success': True, 'message': 'Body posture annotations saved successfully'})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_POST
def auto_annotate_body_posture(request, transcription_id):
    """Use Gemini API to automatically annotate body postures"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get Gemini API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return JsonResponse({'error': 'GEMINI_API_KEY not configured'}, status=500)

    # Get the full text of the transcription
    word_timestamps = transcription.word_timestamps.order_by('start_time_seconds')
    full_text = ' '.join([wt.word for wt in word_timestamps])

    if not full_text.strip():
        return JsonResponse({'error': 'No transcription text available'}, status=400)

    try:
        # Prepare Gemini API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

        prompt = f"""
        Analyze the following text and assign a body posture to each word. Return the result as a JSON array where each element contains the word and its posture.

        Text: "{full_text}"

        Available postures: brave, cross_hands, hello, listen, me, no, point, that, think, this, why, wow

        Return format:
        [
            {{"word": "word1", "posture": "posture1"}},
            {{"word": "word2", "posture": "posture2"}},
            ...
        ]

        Assign exactly one posture per word. Be comprehensive and cover all words.
        """

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()

        result = response.json()
        ai_response = result['candidates'][0]['content']['parts'][0]['text']

        # Parse the AI response (remove markdown code blocks if present)
        if ai_response.startswith('```json'):
            ai_response = ai_response[7:]
        if ai_response.endswith('```'):
            ai_response = ai_response[:-3]

        ai_annotations = json.loads(ai_response.strip())

        # Map AI annotations to word timestamps
        word_list = [wt.word for wt in word_timestamps]
        annotations_created = 0

        for i, wt in enumerate(word_timestamps):
            if i < len(ai_annotations):
                ai_word = ai_annotations[i].get('word', '').strip()
                posture = ai_annotations[i].get('posture', 'listen')

                # Basic validation that words match
                if ai_word.lower() == wt.word.lower():
                    BodyPostureAnnotation.objects.update_or_create(
                        word_timestamp=wt,
                        defaults={
                            'posture': posture,
                            'confidence': 0.8  # Assume reasonable confidence from AI
                        }
                    )
                    annotations_created += 1

        return JsonResponse({
            'success': True,
            'message': f'Auto-annotated {annotations_created} words with body postures',
            'annotations_created': annotations_created
        })

    except requests.RequestException as e:
        return JsonResponse({'error': f'Gemini API error: {str(e)}'}, status=500)
    except (KeyError, json.JSONDecodeError) as e:
        return JsonResponse({'error': f'Failed to parse AI response: {str(e)}'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)


def annotate_characters(request, transcription_id):
    """Display the character annotation interface for a transcription"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    word_timestamps = transcription.word_timestamps.all()
    total_words = word_timestamps.count()

    # Calculate coverage status for all annotation types
    coverage_status = calculate_coverage_status(transcription)

    existing_annotations_dict = {}
    for ann in CharacterAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.character

    for wt in word_timestamps:
        wt.character = existing_annotations_dict.get(str(wt.id), 'none')

    # Check for complete coverage (for current page)
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'character_choices': CharacterAnnotation.CHARACTER_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
        'coverage_status': coverage_status,
    }

    return render(request, 'annotation/annotate_characters.html', context)

@require_POST
@csrf_exempt
def save_character_annotations(request, transcription_id):
    """Save character annotations for words"""
    try:
        data = json.loads(request.body)
        annotations = data.get('annotations', [])

        transcription = get_object_or_404(AudioTranscription, id=transcription_id)

        word_timestamp_ids = set(transcription.word_timestamps.values_list('id', flat=True))
        annotated_ids = set()

        for annotation in annotations:
            word_timestamp_id = annotation.get('word_timestamp_id')
            character = annotation.get('character')

            if not word_timestamp_id or not character:
                return JsonResponse({'error': 'Invalid annotation data'}, status=400)

            if character not in dict(CharacterAnnotation.CHARACTER_CHOICES):
                return JsonResponse({'error': f'Invalid character: {character}'}, status=400)

            annotated_ids.add(word_timestamp_id)

            word_timestamp = get_object_or_404(WordTimestamp, id=word_timestamp_id, transcription=transcription)
            CharacterAnnotation.objects.update_or_create(
                word_timestamp=word_timestamp,
                defaults={'character': character}
            )

        missing_ids = word_timestamp_ids - annotated_ids
        if missing_ids:
            for word_id in missing_ids:
                word_timestamp = WordTimestamp.objects.get(id=word_id)
                CharacterAnnotation.objects.update_or_create(
                    word_timestamp=word_timestamp,
                    defaults={'character': 'character1'} # Default to Character 1
                )

        return JsonResponse({'success': True, 'message': 'Character annotations saved successfully'})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def annotate_background(request, transcription_id):
    """Display the background annotation interface for a transcription"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get all word timestamps for this transcription
    word_timestamps = transcription.word_timestamps.all()
    total_words = word_timestamps.count()

    # Calculate coverage status for all annotation types
    coverage_status = calculate_coverage_status(transcription)

    # Get existing annotations as a simple dict for template access
    existing_annotations_dict = {}
    for ann in BackgroundAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = {
            'background_type': ann.background_type,
            'background_value': ann.background_value
        }

    # Annotate word_timestamps with their backgrounds
    for wt in word_timestamps:
        wt.background = existing_annotations_dict.get(str(wt.id), {'background_type': 'none', 'background_value': ''})

    # Check for complete coverage (for current page)
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    # Get list of available background images for this project
    available_images = []
    from django.conf import settings
    project_media_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(transcription.project.id), 'media')
    if os.path.exists(project_media_dir):
        for filename in os.listdir(project_media_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                available_images.append({
                    'filename': filename,
                    'path': f'projects/{transcription.project.id}/media/{filename}'
                })

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'background_choices': BackgroundAnnotation.BACKGROUND_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
        'available_images': available_images,
        'coverage_status': coverage_status,
    }

    return render(request, 'annotation/annotate_background.html', context)

@require_POST
@csrf_exempt
def save_background_annotations(request, transcription_id):
    """Save background annotations for words"""
    try:
        data = json.loads(request.body)
        annotations = data.get('annotations', [])

        transcription = get_object_or_404(AudioTranscription, id=transcription_id)

        # Validate that all words are covered
        word_timestamp_ids = set(transcription.word_timestamps.values_list('id', flat=True))
        annotated_ids = set()

        for annotation_data in annotations:
            word_timestamp_id = annotation_data.get('word_timestamp_id')
            background_type = annotation_data.get('background_type', '')
            background_value = annotation_data.get('background_value', '')

            if not word_timestamp_id:
                continue

            if background_type and background_type not in dict(BackgroundAnnotation.BACKGROUND_CHOICES):
                return JsonResponse({'error': f'Invalid background type: {background_type}'}, status=400)

            annotated_ids.add(word_timestamp_id)

            word_timestamp = get_object_or_404(WordTimestamp, id=word_timestamp_id, transcription=transcription)

            if background_type and background_type != 'none':
                BackgroundAnnotation.objects.update_or_create(
                    word_timestamp=word_timestamp,
                    defaults={
                        'background_type': background_type,
                        'background_value': background_value
                    }
                )
            else:
                # Remove annotation if background_type is 'none'
                BackgroundAnnotation.objects.filter(word_timestamp=word_timestamp).delete()

        # Check for missing annotations
        missing_ids = word_timestamp_ids - annotated_ids
        if missing_ids:
            # Auto-assign default background "white" to missing words
            for word_id in missing_ids:
                word_timestamp = WordTimestamp.objects.get(id=word_id)
                BackgroundAnnotation.objects.update_or_create(
                    word_timestamp=word_timestamp,
                    defaults={
                        'background_type': 'white',
                        'background_value': ''
                    }
                )

        return JsonResponse({'success': True, 'message': 'Background annotations saved successfully'})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def combined_annotations(request, transcription_id):
    """Display all annotations combined in a single table"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    word_timestamps = transcription.word_timestamps.all()
    total_words = word_timestamps.count()

    # Initialize G2p for phoneme conversion
    g2p = G2p()

    # Calculate coverage status for all annotation types
    coverage_status = calculate_coverage_status(transcription)

    # Check if frames are already saved in database
    existing_frames = FrameAnnotation.objects.filter(transcription=transcription)
    frames_saved = existing_frames.exists()

    # Build combined data for each word
    combined_data = []
    for wt in word_timestamps:
        # Safely get related annotation values
        try:
            emotion = wt.emotion_annotation.emotion
        except EmotionAnnotation.DoesNotExist:
            emotion = '-'

        try:
            body_posture = wt.body_posture_annotation.posture
        except BodyPostureAnnotation.DoesNotExist:
            body_posture = '-'

        try:
            mode = wt.mode_annotation.mode
        except ModeAnnotation.DoesNotExist:
            mode = '-'

        try:
            character = wt.character_annotation.character
        except CharacterAnnotation.DoesNotExist:
            character = '-'

        try:
            background = wt.background_annotation.background_type
        except BackgroundAnnotation.DoesNotExist:
            background = '-'

        # Generate phonemes for the word
        try:
            # Clean the word for phoneme conversion (remove punctuation, etc.)
            clean_word = ''.join(char for char in wt.word if char.isalnum())
            if clean_word:
                phonemes = list(g2p(clean_word.lower()))
            else:
                phonemes = []
        except Exception:
            # If phoneme conversion fails, return empty list
            phonemes = []

        row = {
            'word': wt.word,
            'start_time': wt.start_time_seconds,
            'end_time': wt.end_time_seconds,
            'emotion': emotion,
            'body_posture': body_posture,
            'mode': mode,
            'character': character,
            'background': background,
            'phonemes': phonemes,
        }
        combined_data.append(row)

    context = {
        'transcription': transcription,
        'combined_data': combined_data,
        'coverage_status': coverage_status,
        'frames_saved': frames_saved,
        'existing_frames': existing_frames if frames_saved else None,
    }

    return render(request, 'annotation/combined_annotations.html', context)


@require_POST
@csrf_exempt
def upload_background_image(request, transcription_id):
    """Handle background image upload for a transcription"""
    try:
        transcription = get_object_or_404(AudioTranscription, id=transcription_id)

        if 'background_image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)

        image_file = request.FILES['background_image']

        # Validate file type
        allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/gif']
        if hasattr(image_file, 'content_type') and image_file.content_type not in allowed_types:
            return JsonResponse({'error': 'Invalid file type. Only PNG, JPG, BMP, and GIF are allowed.'}, status=400)

        # Create project media directory within MEDIA_ROOT
        from django.conf import settings
        project_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(transcription.project.id))
        media_dir = os.path.join(project_dir, 'media')
        os.makedirs(media_dir, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.splitext(image_file.name)[0]
        extension = os.path.splitext(image_file.name)[1]
        unique_filename = f"{original_name}_{timestamp}{extension}"

        # Save the file
        file_path = os.path.join(media_dir, unique_filename)
        with open(file_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # Return the relative path for storage in annotation
        relative_path = f"projects/{transcription.project.id}/media/{unique_filename}"

        return JsonResponse({
            'success': True,
            'image_path': relative_path,
            'filename': unique_filename
        })

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@require_POST
def save_frames_to_db(request, transcription_id):
    """Save the current frame table to database"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    try:
        # Delete existing frames for this transcription
        FrameAnnotation.objects.filter(transcription=transcription).delete()

        # Generate frames using the same logic as the frontend
        word_timestamps = transcription.word_timestamps.all()
        g2p_obj = G2p()

        # Build word data
        word_data = []
        for wt in word_timestamps:
            try:
                emotion = wt.emotion_annotation.emotion
            except EmotionAnnotation.DoesNotExist:
                emotion = 'neutral'

            try:
                body_posture = wt.body_posture_annotation.posture
            except BodyPostureAnnotation.DoesNotExist:
                body_posture = 'neutral'

            try:
                mode = wt.mode_annotation.mode
            except ModeAnnotation.DoesNotExist:
                mode = 'big_side'

            try:
                character = wt.character_annotation.character
            except CharacterAnnotation.DoesNotExist:
                character = 'character1'

            try:
                background = wt.background_annotation.background_type
            except BackgroundAnnotation.DoesNotExist:
                background = 'white'

            try:
                clean_word = ''.join(char for char in wt.word if char.isalnum())
                phonemes = list(g2p_obj(clean_word.lower())) if clean_word else []
            except:
                phonemes = []

            word_data.append({
                'word': wt.word,
                'start_time': wt.start_time_seconds,
                'end_time': wt.end_time_seconds,
                'emotion': emotion,
                'body_posture': body_posture,
                'mode': mode,
                'character': character,
                'background': background,
                'phonemes': phonemes
            })

        # Generate frames using the same algorithm as frontend
        frames = generate_frames_from_words(word_data)

        # Assign media to frames based on mode chunks
        frames_with_media = assign_media_to_frames_list(frames, transcription)

        # Save frames to database
        frame_objects = []
        for frame in frames_with_media:
            frame_obj = FrameAnnotation(
                transcription=transcription,
                frame_number=frame['frame'],
                time_seconds=frame['time'],
                word=frame['word'],
                phoneme=frame['phoneme'],
                emotion=frame['emotion'],
                body_posture=frame['body_posture'],
                mode=frame['mode'],
                character=frame['character'],
                background=frame['background'],
                head_direction=frame.get('head_direction', 'M'),
                eye_direction=frame.get('eye_direction', 'M'),
                head_tilt=frame.get('head_tilt', 0),
                zoom_level=frame.get('zoom_level', 1.0),
                blink=frame.get('blink', False),
                media=frame.get('media', ''),  # Now includes media assignments
                intensity=frame.get('intensity', False)
            )
            frame_objects.append(frame_obj)

        # Bulk create for efficiency
        FrameAnnotation.objects.bulk_create(frame_objects)

        return JsonResponse({
            'success': True,
            'message': f'Successfully saved {len(frame_objects)} frames to database. Previous frames (if any) have been overwritten.'
        })

    except Exception as e:
        return JsonResponse({'error': f'Failed to save frames: {str(e)}'}, status=500)

@require_POST
def update_frames_from_annotations(request, transcription_id):
    """Regenerate frames when annotations change"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    try:
        # Delete existing frames
        FrameAnnotation.objects.filter(transcription=transcription).delete()

        # Redirect to save frames (reuse the same logic)
        return save_frames_to_db(request, transcription_id)

    except Exception as e:
        return JsonResponse({'error': f'Failed to update frames: {str(e)}'}, status=500)

def generate_frames_from_words(word_data):
    """Generate frame data from word data (same logic as frontend)"""
    if not word_data:
        return []

    FPS = 30
    DEFAULT_PHONEME = 'CLOSED'

    # Calculate total frames
    max_end_time = max(row['end_time'] for row in word_data)
    total_frames = int(max_end_time * FPS) + 1

    # Pre-calculate frame assignments for each word
    word_frame_assignments = {}
    for idx, row in enumerate(word_data):
        start_frame = int(row['start_time'] * FPS) + 1
        end_frame = int(row['end_time'] * FPS) + 1
        total_word_frames = end_frame - start_frame

        phonemes = row['phonemes']
        num_phonemes = len(phonemes)

        if num_phonemes == 0:
            word_frame_assignments[idx] = {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'frame_to_phoneme': {},
                'row': row
            }
            continue

        # Distribute frames among phonemes
        base_frames = total_word_frames // num_phonemes
        remainder = total_word_frames % num_phonemes

        phoneme_frames = [base_frames] * num_phonemes
        for i in range(remainder):
            phoneme_frames[i] += 1

        frame_to_phoneme = {}
        current_frame = start_frame
        for phoneme_idx, num_frames in enumerate(phoneme_frames):
            for _ in range(num_frames):
                if current_frame <= end_frame:
                    frame_to_phoneme[current_frame] = phonemes[phoneme_idx]
                    current_frame += 1

        word_frame_assignments[idx] = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frame_to_phoneme': frame_to_phoneme,
            'row': row
        }

    # Generate frames
    frames = []
    for frame_num in range(1, total_frames + 1):
        word_found = False
        current_phoneme = DEFAULT_PHONEME
        row = None

        for idx, word_data_item in word_frame_assignments.items():
            if word_data_item['start_frame'] <= frame_num <= word_data_item['end_frame']:
                word_found = True
                row = word_data_item['row']
                current_phoneme = word_data_item['frame_to_phoneme'].get(frame_num,
                    row['phonemes'][-1] if row['phonemes'] else DEFAULT_PHONEME)
                break

        frame_time = (frame_num - 0.5) / FPS

        if word_found and row:
            frames.append({
                'frame': frame_num,
                'time': round(frame_time, 3),
                'word': row['word'],
                'phoneme': current_phoneme,
                'emotion': row['emotion'],
                'body_posture': row['body_posture'],
                'mode': row['mode'],
                'character': row['character'],
                'background': row['background'],
                'head_direction': 'M',
                'eye_direction': 'M',
                'head_tilt': 0,
                'zoom_level': 1.0,
                'blink': False,
                'intensity': False
            })
        else:
            last_frame = frames[-1] if frames else None
            if last_frame:
                frames.append({
                    'frame': frame_num,
                    'time': round(frame_time, 3),
                    'word': '',
                    'phoneme': DEFAULT_PHONEME,
                    'emotion': last_frame['emotion'],
                    'body_posture': last_frame['body_posture'],
                    'mode': last_frame['mode'],
                    'character': last_frame['character'],
                    'background': last_frame['background'],
                    'head_direction': 'M',
                    'eye_direction': 'M',
                    'head_tilt': 0,
                    'zoom_level': 1.0,
                    'blink': False,
                    'intensity': False
                })
            else:
                frames.append({
                    'frame': frame_num,
                    'time': round(frame_time, 3),
                    'word': '',
                    'phoneme': DEFAULT_PHONEME,
                    'emotion': 'neutral',
                    'body_posture': 'neutral',
                    'mode': 'big_side',
                    'character': 'character1',
                    'background': 'white',
                    'head_direction': 'M',
                    'eye_direction': 'M',
                    'head_tilt': 0,
                    'zoom_level': 1.0,
                    'blink': False,
                    'intensity': False
                })

    return frames


def media_chunks(request, transcription_id):
    """Display media chunks management interface for mode chunks"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Calculate coverage status for all annotation types
    coverage_status = calculate_coverage_status(transcription)

    # Generate mode chunks
    chunks_data = generate_mode_chunks(transcription)

    context = {
        'transcription': transcription,
        'chunks_data': chunks_data,
        'coverage_status': coverage_status,
        'active_page': 'media',
    }

    return render(request, 'annotation/media_chunks.html', context)


@require_POST
@csrf_exempt
def upload_media(request, transcription_id):
    """Handle media upload for a chunk"""
    try:
        transcription = get_object_or_404(AudioTranscription, id=transcription_id)
        chunk_idx = int(request.POST.get('chunk_idx', 0))
        description = request.POST.get('description', '')

        # Create media directory within MEDIA_ROOT
        from django.conf import settings
        media_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(transcription.project.id), 'media')
        os.makedirs(media_dir, exist_ok=True)

        uploaded_files = []

        if 'media_files' in request.FILES:
            files = request.FILES.getlist('media_files')

            for uploaded_file in files:
                # Validate file type
                allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.gif']
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()

                if file_extension not in allowed_extensions:
                    return JsonResponse({'error': f'Invalid file type: {file_extension}'}, status=400)

                # Generate unique filename with chunk prefix
                base_name = os.path.splitext(uploaded_file.name)[0]
                unique_name = f"chunk_{chunk_idx}_{uploaded_file.name}"
                dest_path = os.path.join(media_dir, unique_name)

                # Check if file already exists in the media directory (across all chunks)
                counter = 1
                while os.path.exists(dest_path):
                    name_parts = os.path.splitext(uploaded_file.name)
                    unique_name = f"chunk_{chunk_idx}_{name_parts[0]}_{counter}{name_parts[1]}"
                    dest_path = os.path.join(media_dir, unique_name)
                    counter += 1

                # Save the file
                with open(dest_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                uploaded_files.append(unique_name)

        # Update FrameAnnotation records in database
        update_frame_annotations_media(transcription, chunk_idx, uploaded_files)

        return JsonResponse({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} file(s) and updated database',
            'uploaded_files': uploaded_files
        })

    except Exception as e:
        return JsonResponse({'error': f'Upload failed: {str(e)}'}, status=500)


@require_POST
def deselect_media(request, transcription_id):
    """Deselect media for a chunk"""
    try:
        transcription = get_object_or_404(AudioTranscription, id=transcription_id)
        chunk_idx = int(request.POST.get('chunk_idx', 0))

        # Clear media for this chunk in database
        update_frame_annotations_media(transcription, chunk_idx, [])

        # Delete associated files
        from django.conf import settings
        media_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(transcription.project.id), 'media')
        if os.path.exists(media_dir):
            for filename in os.listdir(media_dir):
                if filename.startswith(f'chunk_{chunk_idx}_'):
                    try:
                        os.remove(os.path.join(media_dir, filename))
                    except OSError:
                        pass  # Ignore if file doesn't exist or can't be deleted

        return JsonResponse({'success': True, 'message': 'Media deselected successfully'})

    except Exception as e:
        return JsonResponse({'error': f'Deselect failed: {str(e)}'}, status=500)


def generate_mode_chunks(transcription):
    """Generate mode chunks from word annotations"""
    word_timestamps = transcription.word_timestamps.order_by('start_time_seconds')

    if not word_timestamps.exists():
        return []

    chunks = []
    current_chunk = None

    for wt in word_timestamps:
        try:
            mode = wt.mode_annotation.mode
        except ModeAnnotation.DoesNotExist:
            mode = 'big_center'  # Default mode

        # Get all words in this mode segment
        words = []
        start_time = wt.start_time_seconds
        end_time = wt.end_time_seconds

        # Group consecutive words with same mode
        if current_chunk and current_chunk['Mode'] == mode:
            # Extend current chunk
            current_chunk['Words'] += f" {wt.word}"
            current_chunk['Duration'] = end_time - current_chunk['Start']
        else:
            # Start new chunk
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = {
                'Mode': mode,
                'Words': wt.word,
                'Start': start_time,
                'Duration': end_time - start_time,
                'media': ''
            }

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Load existing media assignments from database
    # Check if we have FrameAnnotation records for this transcription
    if FrameAnnotation.objects.filter(transcription=transcription).exists():
        # Get media assignments by analyzing FrameAnnotation records
        chunk_media_assignments = get_chunk_media_from_frames(transcription, chunks)
        for idx, chunk in enumerate(chunks):
            media_value = chunk_media_assignments.get(idx, '')
            if media_value:
                # Convert pipe-separated string to list for template
                media_files = [f.strip() for f in str(media_value).split('|') if f.strip()]
                chunks[idx]['media_files'] = media_files
                chunks[idx]['media'] = str(media_value).strip()
            else:
                chunks[idx]['media_files'] = []
    else:
        # No FrameAnnotation records yet, initialize empty media
        for chunk in chunks:
            chunk['media_files'] = []

    return chunks


def get_chunk_media_from_frames(transcription, chunks):
    """Extract media assignments from FrameAnnotation records for each chunk"""
    chunk_media = {}

    # Create mode segments (same logic as update_frame_annotations_media)
    mode_segments = []
    current_time = 0

    for chunk in chunks:
        start_time = current_time
        duration = chunk['Duration']
        end_time = start_time + duration
        mode = chunk['Mode']

        mode_segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'mode': mode,
            'chunk_idx': len(mode_segments)
        })

        current_time = end_time

    # Query FrameAnnotation records and group media by chunk
    frame_annotations = FrameAnnotation.objects.filter(transcription=transcription).order_by('time_seconds')

    for frame in frame_annotations:
        frame_time = frame.time_seconds
        frame_mode = frame.mode
        frame_media = frame.media.strip()

        if frame_media:
            # Find which chunk this frame belongs to
            for segment in mode_segments:
                if segment['start_time'] <= frame_time < segment['end_time'] and segment['mode'] == frame_mode:
                    chunk_idx = segment['chunk_idx']
                    if chunk_idx not in chunk_media:
                        chunk_media[chunk_idx] = set()
                    # Split media if it contains multiple files
                    for media_file in frame_media.split('|'):
                        chunk_media[chunk_idx].add(media_file.strip())
                    break

    # Convert sets back to pipe-separated strings
    result = {}
    for chunk_idx, media_set in chunk_media.items():
        if media_set:
            result[chunk_idx] = '|'.join(sorted(media_set))

    return result


def update_chunk_media_csv(transcription, chunk_idx, new_files):
    """Update the media column in mode_durations.csv"""
    import pandas as pd
    from django.conf import settings

    chunks_csv_path = os.path.join(settings.MEDIA_ROOT, 'projects', str(transcription.project.id), 'annotations', 'mode_durations.csv')

    # Generate chunks data
    chunks_data = generate_mode_chunks(transcription)

    # Create DataFrame
    df_data = []
    for chunk in chunks_data:
        df_data.append({
            'Mode': chunk['Mode'],
            'Duration': chunk['Duration'],
            'Words': chunk['Words'],
            'media': chunk['media']
        })

    df = pd.DataFrame(df_data)

    # Update media for specific chunk
    if chunk_idx < len(df):
        if new_files:
            df.at[chunk_idx, 'media'] = '|'.join(new_files)
        else:
            df.at[chunk_idx, 'media'] = ''

    # Save CSV
    df.to_csv(chunks_csv_path, index=False)

    # Update frame phoneme data with media assignments
    update_frame_media_assignments(transcription, df)


def update_frame_media_assignments(transcription, df_mode_durations):
    """Update frame phoneme data CSV with media assignments"""
    try:
        from django.conf import settings
        frame_phoneme_csv_path = os.path.join(settings.MEDIA_ROOT, 'projects', str(transcription.project.id), 'annotations', 'frame_phoneme_data.csv')

        # Check if frame phoneme data exists
        if not os.path.exists(frame_phoneme_csv_path):
            print(f"Frame phoneme data CSV not found: {frame_phoneme_csv_path}. Skipping media assignment.")
            return

        print("Updating frame phonemes data with media assignments...")

        # Read frame phoneme data
        df_frames = pd.read_csv(frame_phoneme_csv_path)

        # Ensure Frame column exists
        if 'Frame' not in df_frames.columns:
            print("Warning: 'Frame' column not found in frame phoneme data. Skipping media assignment.")
            return

        # Update media assignments using the DataCombiner pattern
        updated_df_frames = assign_media_to_frames(df_frames, df_mode_durations, fps=30)

        # Save the updated frame phoneme data
        updated_df_frames.to_csv(frame_phoneme_csv_path, index=False)

        print(f"Successfully updated frame phoneme data with media assignments at: {frame_phoneme_csv_path}")

    except Exception as e:
        print(f"Error updating frame phoneme data: {str(e)}")


def assign_media_to_frames(df_frames, df_mode_durations, fps=30):
    """Assign media to frames based on mode chunks"""
    # Add media column if it doesn't exist
    if 'media' not in df_frames.columns:
        df_frames['media'] = ''

    # Create time-to-mode mapping
    mode_segments = []
    current_time = 0

    for idx, chunk in df_mode_durations.iterrows():
        start_time = current_time
        duration = chunk['Duration']
        end_time = start_time + duration
        mode = chunk['Mode']
        media = chunk.get('media', '')

        mode_segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'mode': mode,
            'media': media
        })

        current_time = end_time

    # Assign media to frames
    for idx, frame in df_frames.iterrows():
        frame_time = frame['time_seconds']
        frame_mode = frame.get('mode', '')

        # Find matching mode segment
        for segment in mode_segments:
            if segment['start_time'] <= frame_time < segment['end_time'] and segment['mode'] == frame_mode:
                df_frames.at[idx, 'media'] = segment['media']
                break

    return df_frames


def update_frame_annotations_media(transcription, chunk_idx, uploaded_files):
    """Update FrameAnnotation records in database with media assignments"""
    try:
        # Generate mode chunks from current database annotations
        word_timestamps = transcription.word_timestamps.order_by('start_time_seconds')

        if not word_timestamps.exists():
            return

        chunks = []
        current_chunk = None

        for wt in word_timestamps:
            try:
                mode = wt.mode_annotation.mode
            except ModeAnnotation.DoesNotExist:
                mode = 'big_center'  # Default mode

            start_time = wt.start_time_seconds
            end_time = wt.end_time_seconds

            # Group consecutive words with same mode
            if current_chunk and current_chunk['Mode'] == mode:
                # Extend current chunk
                current_chunk['Words'] += f" {wt.word}"
                current_chunk['Duration'] = end_time - current_chunk['Start']
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk)

                current_chunk = {
                    'Mode': mode,
                    'Words': wt.word,
                    'Start': start_time,
                    'Duration': end_time - start_time,
                    'media': ''
                }

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Load existing media assignments from database to preserve other chunks
        existing_media_assignments = get_chunk_media_from_frames(transcription, chunks)

        # Update media assignment for the specific chunk
        if chunk_idx < len(chunks):
            if uploaded_files:
                chunks[chunk_idx]['media'] = '|'.join(uploaded_files)
            else:
                chunks[chunk_idx]['media'] = ''
        # Preserve existing media for other chunks
        for idx, chunk in enumerate(chunks):
            if idx != chunk_idx and idx in existing_media_assignments:
                chunk['media'] = existing_media_assignments[idx]

        # Create mode segments for media assignment
        mode_segments = []
        current_time = 0

        for chunk in chunks:
            start_time = current_time
            duration = chunk['Duration']
            end_time = start_time + duration
            mode = chunk['Mode']
            media = chunk.get('media', '')

            mode_segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'mode': mode,
                'media': media
            })

            current_time = end_time

        # Update FrameAnnotation records in database
        frame_annotations = FrameAnnotation.objects.filter(transcription=transcription)

        for frame_annotation in frame_annotations:
            frame_time = frame_annotation.time_seconds
            frame_mode = frame_annotation.mode

            # Find matching mode segment
            matching_media = ''
            for segment in mode_segments:
                if segment['start_time'] <= frame_time < segment['end_time'] and segment['mode'] == frame_mode:
                    matching_media = segment['media']
                    break

            # Update the media field
            frame_annotation.media = matching_media
            frame_annotation.save(update_fields=['media'])

    except Exception as e:
        print(f"Error updating frame annotations media: {str(e)}")
        # Don't raise exception to avoid breaking the upload process


def assign_media_to_frames_list(frames, transcription):
    """Assign media to frames list based on mode chunks"""
    import pandas as pd
    from django.conf import settings

    # Load mode durations CSV
    chunks_csv_path = os.path.join(settings.MEDIA_ROOT, 'projects', str(transcription.project.id), 'annotations', 'mode_durations.csv')
    if not os.path.exists(chunks_csv_path):
        # No media assignments yet, return frames as-is
        for frame in frames:
            frame['media'] = ''
        return frames

    try:
        df_mode_durations = pd.read_csv(chunks_csv_path)

        # Create time-to-mode mapping
        mode_segments = []
        current_time = 0

        for idx, chunk in df_mode_durations.iterrows():
            start_time = current_time
            duration = chunk['Duration']
            end_time = start_time + duration
            mode = chunk['Mode']
            media = chunk.get('media', '')

            mode_segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'mode': mode,
                'media': media
            })

            current_time = end_time

        # Assign media to frames
        for frame in frames:
            frame_time = frame['time']
            frame_mode = frame.get('mode', '')

            # Find matching mode segment
            for segment in mode_segments:
                if segment['start_time'] <= frame_time < segment['end_time'] and segment['mode'] == frame_mode:
                    frame['media'] = segment['media']
                    break
            else:
                frame['media'] = ''

        return frames

    except Exception as e:
        print(f"Error assigning media to frames: {e}")
        # Return frames with empty media on error
        for frame in frames:
            frame['media'] = ''
        return frames


def export_frames_dataframe(transcription_id):
    """
    Export FrameAnnotation data as a pandas DataFrame with correct data types.
    Ensures boolean fields remain boolean and media paths remain strings.
    """
    import pandas as pd
    from django.db import connection

    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Query FrameAnnotation data
    frames = FrameAnnotation.objects.filter(transcription=transcription).order_by('frame_number')

    if not frames.exists():
        return pd.DataFrame()  # Return empty DataFrame if no frames

    # Convert to DataFrame with explicit type handling
    data = []
    for frame in frames:
        data.append({
            'Frame': frame.frame_number,
            'Word': frame.word,
            'Start Time (s)': frame.time_seconds,  # Will be calculated from word timestamps if needed
            'End Time (s)': frame.time_seconds,    # Will be calculated from word timestamps if needed
            'Assigned_Phoneme': frame.phoneme,
            'starting_frame': frame.frame_number,  # Same as Frame for this implementation
            'end_frame': frame.frame_number,       # Same as Frame for this implementation
            'Frame_Range': f"{frame.frame_number}-{frame.frame_number}",
            'Phonemes': [frame.phoneme] if frame.phoneme else [],  # List format
            'Emotion': frame.emotion,
            'Character': frame.character,
            'Mode': frame.mode,
            'Body Posture': frame.body_posture,
            'Intensity': bool(frame.intensity),  # Ensure boolean type
            'Background': frame.background,
            'Head_Direction': frame.head_direction,
            'Eye_Direction': frame.eye_direction,
            'Head_Tilt': frame.head_tilt,
            'Zoom_Level': frame.zoom_level,
            'Blink': bool(frame.blink),  # Ensure boolean type
            'media': frame.media  # Ensure string/path type
        })

    df = pd.DataFrame(data)

    # Explicitly set data types to ensure consistency
    df = df.astype({
        'Frame': 'int64',
        'Start Time (s)': 'float64',
        'End Time (s)': 'float64',
        'starting_frame': 'int64',
        'end_frame': 'int64',
        'Head_Tilt': 'int64',
        'Zoom_Level': 'float64',
        'Intensity': 'bool',
        'Blink': 'bool'
    })

    # Ensure Phonemes column contains lists
    df['Phonemes'] = df['Phonemes'].apply(lambda x: x if isinstance(x, list) else [])

    return df
