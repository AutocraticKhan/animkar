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

class BodyPostureAnnotation(models.Model):
    POSTURE_CHOICES = [
        ('brave', 'Brave'),
        ('cross_hands', 'Cross Hands'),
        ('hello', 'Hello'),
        ('listen', 'Listen'),
        ('me', 'Me'),
        ('no', 'No'),
        ('point', 'Point'),
        ('that', 'That'),
        ('think', 'Think'),
        ('this', 'This'),
        ('why', 'Why'),
        ('wow', 'Wow'),
    ]

    word_timestamp = models.OneToOneField(
        WordTimestamp,
        on_delete=models.CASCADE,
        related_name='body_posture_annotation'
    )
    posture = models.CharField(max_length=20, choices=POSTURE_CHOICES)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional: confidence score from AI annotation
    confidence = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['word_timestamp__start_time_seconds']

    def __str__(self):
        return f"{self.word_timestamp.word}: {self.posture}"

class ModeAnnotation(models.Model):
    MODE_CHOICES = [
        ('big_center', 'Big Center'),
        ('big_side', 'Big Side'),
        ('small_side', 'Small Side'),
        ('no_avatar', 'No Avatar'),
    ]

    word_timestamp = models.OneToOneField(
        WordTimestamp,
        on_delete=models.CASCADE,
        related_name='mode_annotation'
    )
    mode = models.CharField(max_length=20, choices=MODE_CHOICES)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional: confidence score from AI annotation
    confidence = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['word_timestamp__start_time_seconds']

    def __str__(self):
        return f"{self.word_timestamp.word}: {self.mode}"
