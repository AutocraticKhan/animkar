import json
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

def load_frame_config():
    """
    Load frame configuration from JSON file.
    Returns the parsed JSON data containing frame sizes and coordinates.
    """
    config_path = Path(settings.BASE_DIR) / 'frame_making' / 'data' / 'frame_config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def frame_list(request):
    """
    View to display available frames with their sizes and coordinates.
    """
    config = load_frame_config()
    return JsonResponse(config)

# Create your views here.
