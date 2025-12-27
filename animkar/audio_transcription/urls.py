from django.urls import path
from . import views

app_name = 'audio_transcription'

urlpatterns = [
    path('upload/<int:project_id>/', views.upload_audio, name='upload_audio'),
    path('transcription/<int:transcription_id>/', views.transcription_detail, name='transcription_detail'),
    path('transcription/<int:transcription_id>/retry/', views.retry_transcription, name='retry_transcription'),
    path('transcription/<int:pk>/delete/', views.AudioTranscriptionDeleteView.as_view(), name='delete_transcription'),
]
