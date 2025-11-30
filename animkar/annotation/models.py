from django.db import models
from audio_transcription.models import WordTimestamp

class EmotionAnnotation(models.Model):
    EMOTION_CHOICES = [
        ('angry', 'Angry'),
        ('bore', 'Bore'),
        ('content', 'Content'),  # Default emotion for uncovered words
        ('glare', 'Glare'),
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('sarcasm', 'Sarcasm'),
        ('worried', 'Worried'),
    ]

    word_timestamp = models.OneToOneField(
        WordTimestamp,
        on_delete=models.CASCADE,
        related_name='emotion_annotation'
    )
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional: confidence score from AI annotation
    confidence = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['word_timestamp__start_time_seconds']

    def __str__(self):
        return f"{self.word_timestamp.word}: {self.emotion}"
