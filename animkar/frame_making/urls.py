from django.urls import path
from . import views

app_name = 'frame_making'

urlpatterns = [
    path('generate-video/<int:transcription_id>/', views.generate_video_view, name='generate_video'),
    path('download-video/<int:transcription_id>/', views.download_video_view, name='download_video'),
    path('video-status/<int:transcription_id>/', views.video_status_view, name='video_status'),
    path('generate-video-ajax/<int:transcription_id>/', views.generate_video_ajax_view, name='generate_video_ajax'),
]
