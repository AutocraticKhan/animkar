from django.urls import path
from . import views

app_name = 'annotation'

urlpatterns = [
    path('transcription/<int:transcription_id>/', views.annotate_transcription, name='annotate_transcription'),
    path('transcription/<int:transcription_id>/save/', views.save_annotations, name='save_annotations'),
    path('transcription/<int:transcription_id>/auto-annotate/', views.auto_annotate, name='auto_annotate'),
    path('transcription/<int:transcription_id>/body-posture/', views.annotate_body_posture, name='annotate_body_posture'),
    path('transcription/<int:transcription_id>/body-posture/save/', views.save_body_posture_annotations, name='save_body_posture_annotations'),
    path('transcription/<int:transcription_id>/body-posture/auto-annotate/', views.auto_annotate_body_posture, name='auto_annotate_body_posture'),
]
