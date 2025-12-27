"""
Management command to migrate existing files to the new unified project structure within MEDIA_ROOT.
"""
import os
import shutil
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from project_manager.models import Project
from audio_transcription.models import AudioTranscription


class Command(BaseCommand):
    help = 'Migrate existing files to the new unified project structure within MEDIA_ROOT'

    def handle(self, *args, **options):
        self.stdout.write('Starting file structure migration...')

        # Migrate audio files from old location
        self.migrate_audio_files()

        # Migrate media chunks and CSV files
        self.migrate_project_files()

        self.stdout.write(self.style.SUCCESS('File structure migration completed!'))

    def migrate_audio_files(self):
        """Migrate audio files from audio_transcriptions/ to projects/{id}/audio/"""
        old_audio_base = os.path.join(settings.MEDIA_ROOT, 'audio_transcriptions')
        if not os.path.exists(old_audio_base):
            self.stdout.write('No old audio files to migrate')
            return

        # Get all project directories in old location
        for project_dir_name in os.listdir(old_audio_base):
            project_path = os.path.join(old_audio_base, project_dir_name)
            if not os.path.isdir(project_path):
                continue

            # Extract project ID from directory name (project_{id})
            if not project_dir_name.startswith('project_'):
                continue

            try:
                project_id = int(project_dir_name.split('_')[1])
            except (IndexError, ValueError):
                self.stdout.write(f'Skipping invalid project directory: {project_dir_name}')
                continue

            # Create new directory structure
            new_project_base = os.path.join(settings.MEDIA_ROOT, 'projects', str(project_id))
            new_audio_dir = os.path.join(new_project_base, 'audio')
            os.makedirs(new_audio_dir, exist_ok=True)

            # Move all files from old project directory to new audio directory
            for filename in os.listdir(project_path):
                old_file_path = os.path.join(project_path, filename)
                new_file_path = os.path.join(new_audio_dir, filename)

                if os.path.isfile(old_file_path):
                    shutil.move(old_file_path, new_file_path)
                    self.stdout.write(f'Moved audio file: {old_file_path} -> {new_file_path}')

            # Remove old empty directory
            try:
                os.rmdir(project_path)
            except OSError:
                self.stdout.write(f'Could not remove old directory: {project_path}')

        # Remove old base directory if empty
        try:
            os.rmdir(old_audio_base)
        except OSError:
            pass

    def migrate_project_files(self):
        """Migrate media chunks and CSV files from projects/ to new structure within MEDIA_ROOT"""
        old_projects_base = os.path.join(settings.BASE_DIR, 'projects')
        if not os.path.exists(old_projects_base):
            self.stdout.write('No old project files to migrate')
            return

        # Get all project directories
        for project_dir_name in os.listdir(old_projects_base):
            project_path = os.path.join(old_projects_base, project_dir_name)
            if not os.path.isdir(project_path):
                continue

            try:
                project_id = int(project_dir_name)
            except ValueError:
                self.stdout.write(f'Skipping invalid project directory: {project_dir_name}')
                continue

            # Create new directory structure within MEDIA_ROOT
            new_project_base = os.path.join(settings.MEDIA_ROOT, 'projects', str(project_id))
            new_media_dir = os.path.join(new_project_base, 'media')
            new_annotations_dir = os.path.join(new_project_base, 'annotations')
            os.makedirs(new_media_dir, exist_ok=True)
            os.makedirs(new_annotations_dir, exist_ok=True)

            # Move media directory if it exists
            old_media_dir = os.path.join(project_path, 'media')
            if os.path.exists(old_media_dir):
                for filename in os.listdir(old_media_dir):
                    old_file_path = os.path.join(old_media_dir, filename)
                    new_file_path = os.path.join(new_media_dir, filename)

                    if os.path.isfile(old_file_path):
                        shutil.move(old_file_path, new_file_path)
                        self.stdout.write(f'Moved media file: {old_file_path} -> {new_file_path}')

                # Remove old media directory
                try:
                    os.rmdir(old_media_dir)
                except OSError:
                    pass

            # Move CSV files to annotations directory
            for filename in os.listdir(project_path):
                if filename.endswith('.csv'):
                    old_file_path = os.path.join(project_path, filename)
                    new_file_path = os.path.join(new_annotations_dir, filename)

                    if os.path.isfile(old_file_path):
                        shutil.move(old_file_path, new_file_path)
                        self.stdout.write(f'Moved CSV file: {old_file_path} -> {new_file_path}')

            # Remove old project directory if empty
            try:
                os.rmdir(project_path)
            except OSError:
                self.stdout.write(f'Could not remove old project directory: {project_path}')

        # Remove old projects base directory if empty
        try:
            os.rmdir(old_projects_base)
        except OSError:
            pass
