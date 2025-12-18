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


class CharacterAnnotation(models.Model):
    CHARACTER_CHOICES = [
        ('character1', 'Character 1'),
        ('character2', 'Character 2'),
    ]

    word_timestamp = models.OneToOneField(
        WordTimestamp,
        on_delete=models.CASCADE,
        related_name='character_annotation'
    )
    character = models.CharField(max_length=20, choices=CHARACTER_CHOICES)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['word_timestamp__start_time_seconds']

    def __str__(self):
        return f"{self.word_timestamp.word}: {self.character}"

class BackgroundAnnotation(models.Model):
    BACKGROUND_CHOICES = [
        ('green', 'Green Background'),
        ('white', 'White Background'),
        ('custom_color', 'Custom Color'),
        ('custom_image', 'Custom Image'),
    ]

    word_timestamp = models.OneToOneField(
        WordTimestamp,
        on_delete=models.CASCADE,
        related_name='background_annotation'
    )
    background_type = models.CharField(max_length=20, choices=BACKGROUND_CHOICES)
    background_value = models.CharField(max_length=500, blank=True, null=True)  # Hex color or image path

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional: confidence score if auto-generated
    confidence = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['word_timestamp__start_time_seconds']

    def __str__(self):
        return f"{self.word_timestamp.word}: {self.background_type}"

class FrameAnnotation(models.Model):
    """Frame-by-frame annotations at 30 FPS"""
    transcription = models.ForeignKey(
        'audio_transcription.AudioTranscription',
        on_delete=models.CASCADE,
        related_name='frame_annotations'
    )
    frame_number = models.PositiveIntegerField()  # Frame number (1-based)
    time_seconds = models.FloatField()  # Time in seconds for this frame

    # Word-level data interpolated to frames
    word = models.CharField(max_length=100, blank=True)  # Word being spoken
    phoneme = models.CharField(max_length=20, blank=True)  # Current phoneme

    # Annotations
    emotion = models.CharField(max_length=20, blank=True)
    body_posture = models.CharField(max_length=20, blank=True)
    mode = models.CharField(max_length=20, blank=True)
    character = models.CharField(max_length=20, blank=True)
    background = models.CharField(max_length=20, blank=True)

    # Animation data
    head_direction = models.CharField(max_length=5, choices=[('L', 'Left'), ('M', 'Middle'), ('R', 'Right')], default='M')
    eye_direction = models.CharField(max_length=5, choices=[('L', 'Left'), ('M', 'Middle'), ('R', 'Right')], default='M')
    head_tilt = models.IntegerField(default=0)  # Degrees (-10 to 10)
    zoom_level = models.FloatField(default=1.0)  # Zoom multiplier
    blink = models.BooleanField(default=False)

    # Media assignment (initially empty, filled later)
    media = models.CharField(max_length=500, blank=True, null=True)  # Path to media file or description

    # Animation intensity
    intensity = models.BooleanField(default=False)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['frame_number']
        unique_together = ['transcription', 'frame_number']

    def __str__(self):
        return f"Frame {self.frame_number} ({self.time_seconds:.3f}s): {self.word}"
