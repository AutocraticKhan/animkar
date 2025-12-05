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
    path('transcription/<int:transcription_id>/mode/', views.annotate_mode, name='annotate_mode'),
    path('transcription/<int:transcription_id>/mode/save/', views.save_mode_annotations, name='save_mode_annotations'),
    path('transcription/<int:transcription_id>/mode/auto-annotate/', views.auto_annotate_mode, name='auto_annotate_mode'),
    path('transcription/<int:transcription_id>/characters/', views.annotate_characters, name='annotate_characters'),
    path('transcription/<int:transcription_id>/characters/save/', views.save_character_annotations, name='save_character_annotations'),
    path('transcription/<int:transcription_id>/background/', views.annotate_background, name='annotate_background'),
    path('transcription/<int:transcription_id>/background/save/', views.save_background_annotations, name='save_background_annotations'),
    path('transcription/<int:transcription_id>/background/upload/', views.upload_background_image, name='upload_background_image'),
    path('transcription/<int:transcription_id>/combined/', views.combined_annotations, name='combined_annotations'),
]
