from django.db import models
from django.conf import settings
import os
import shutil

class Project(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def delete(self, *args, **kwargs):
        """
        Override delete to clean up associated files and database records.
        """
        # Delete all associated audio files from disk
        for transcription in self.transcriptions.all():
            if transcription.audio_file:
                file_path = transcription.audio_file.path
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass  # File may already be deleted or inaccessible

        # Delete project directory if it exists and is empty
        project_dir = os.path.join(settings.MEDIA_ROOT, f"audio_transcriptions/project_{self.id}")
        if os.path.exists(project_dir):
            try:
                # Try to remove directory (will fail if not empty, which is fine)
                os.rmdir(project_dir)
            except OSError:
                # Directory not empty or other error - keep it
                pass

        # Call parent delete method (this will cascade delete all related objects)
        super().delete(*args, **kwargs)
