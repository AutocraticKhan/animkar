from django.db import models
from project_manager.models import Project
import os

class AudioTranscription(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='transcriptions')
    audio_file = models.FileField(upload_to='audio_transcriptions/')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Metadata
    original_filename = models.CharField(max_length=255)
    file_size = models.PositiveIntegerField(help_text="File size in bytes")
    duration_seconds = models.FloatField(null=True, blank=True)
    language = models.CharField(max_length=10, default='hi')  # Default to Hindi
    model_name = models.CharField(max_length=50, default='tiny')

    # Processing results
    total_words = models.PositiveIntegerField(null=True, blank=True)
    high_confidence_words = models.PositiveIntegerField(null=True, blank=True)
    average_confidence = models.FloatField(null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.project.name} - {self.original_filename}"

    def get_audio_file_path(self):
        """Get the full path to the audio file"""
        if self.audio_file:
            return self.audio_file.path
        return None

    def get_file_extension(self):
        """Get file extension from original filename"""
        return os.path.splitext(self.original_filename)[1].lower()

class WordTimestamp(models.Model):
    transcription = models.ForeignKey(AudioTranscription, on_delete=models.CASCADE, related_name='word_timestamps')

    # Word data
    word = models.CharField(max_length=100)
    transliterated_word = models.CharField(max_length=100, blank=True, null=True)

    # Timing
    start_time_seconds = models.FloatField()
    end_time_seconds = models.FloatField()

    # Optional sample-accurate timing
    start_sample = models.PositiveIntegerField(null=True, blank=True)
    end_sample = models.PositiveIntegerField(null=True, blank=True)

    # Confidence
    confidence = models.FloatField()

    # Metadata
    duration_seconds = models.FloatField()
    is_high_confidence = models.BooleanField(default=False)

    class Meta:
        ordering = ['start_time_seconds']

    def __str__(self):
        return f"{self.word} ({self.start_time_seconds:.2f}s - {self.end_time_seconds:.2f}s)"
