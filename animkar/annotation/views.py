import os
import json
import requests
import shutil
from datetime import datetime
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.serializers.json import DjangoJSONEncoder
from audio_transcription.models import AudioTranscription, WordTimestamp
from .models import EmotionAnnotation, BodyPostureAnnotation, ModeAnnotation, CharacterAnnotation, BackgroundAnnotation

def annotate_transcription(request, transcription_id):
    """Display the emotion annotation interface for a transcription"""
    transcription = get_object_or_404(AudioTranscription, id=transcription_id)

    # Get all word timestamps for this transcription
    word_timestamps = transcription.word_timestamps.all()

    # Get existing annotations as a simple dict for template access
    existing_annotations_dict = {}
    for ann in EmotionAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.emotion

    # Annotate word_timestamps with their emotions
    for wt in word_timestamps:
        wt.emotion = existing_annotations_dict.get(str(wt.id), 'none')

    # Check for complete coverage
    total_words = word_timestamps.count()
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'emotion_choices': EmotionAnnotation.EMOTION_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
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

    # Get existing annotations as a simple dict for template access
    existing_annotations_dict = {}
    for ann in ModeAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.mode

    # Annotate word_timestamps with their modes
    for wt in word_timestamps:
        wt.mode = existing_annotations_dict.get(str(wt.id), 'none')

    # Check for complete coverage
    total_words = word_timestamps.count()
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'mode_choices': ModeAnnotation.MODE_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
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

    # Get existing annotations as a simple dict for template access
    existing_annotations_dict = {}
    for ann in BodyPostureAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.posture

    # Annotate word_timestamps with their postures
    for wt in word_timestamps:
        wt.posture = existing_annotations_dict.get(str(wt.id), 'none')

    # Check for complete coverage
    total_words = word_timestamps.count()
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'posture_choices': BodyPostureAnnotation.POSTURE_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
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

    existing_annotations_dict = {}
    for ann in CharacterAnnotation.objects.filter(word_timestamp__transcription=transcription):
        existing_annotations_dict[str(ann.word_timestamp_id)] = ann.character

    for wt in word_timestamps:
        wt.character = existing_annotations_dict.get(str(wt.id), 'none')

    total_words = word_timestamps.count()
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'character_choices': CharacterAnnotation.CHARACTER_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
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

    # Check for complete coverage
    total_words = word_timestamps.count()
    annotated_words = len(existing_annotations_dict)
    coverage_complete = total_words == annotated_words

    context = {
        'transcription': transcription,
        'word_timestamps': word_timestamps,
        'background_choices': BackgroundAnnotation.BACKGROUND_CHOICES,
        'coverage_complete': coverage_complete,
        'missing_words': total_words - annotated_words,
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

        for annotation_data in annotations:
            word_timestamp_id = annotation_data.get('word_timestamp_id')
            background_type = annotation_data.get('background_type', '')
            background_value = annotation_data.get('background_value', '')

            if not word_timestamp_id:
                continue

            if background_type and background_type not in dict(BackgroundAnnotation.BACKGROUND_CHOICES):
                return JsonResponse({'error': f'Invalid background type: {background_type}'}, status=400)

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

        return JsonResponse({'success': True, 'message': 'Background annotations saved successfully'})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

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

        # Create project media directory if it doesn't exist
        project_dir = os.path.join('projects', str(transcription.project.id))
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
