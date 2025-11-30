import os
import json
import requests
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib import messages
from audio_transcription.models import AudioTranscription, WordTimestamp
from .models import EmotionAnnotation

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
