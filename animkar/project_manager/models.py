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
        Override delete to clean up all associated files and directories.
        """
        # Delete the entire project directory and all its contents
        project_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(self.id))
        if os.path.exists(project_dir):
            try:
                shutil.rmtree(project_dir)
            except OSError:
                pass  # Directory may not exist or other error

        # Call parent delete method (this will cascade delete all related objects)
        super().delete(*args, **kwargs)
