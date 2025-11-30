import os
import whisper_timestamped as whisper
import json
import pandas as pd
import torch
from pydub import AudioSegment
import numpy as np
from datetime import datetime

# Check for optional dependencies
try:
    import librosa
    import scipy.signal
    import soundfile as sf
    ENHANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSING_AVAILABLE = False

# Silence detection parameters
SILENCE_THRESHOLD_DB = -30  # dB threshold for silence detection
SILENCE_WINDOW_MS = 100     # Window size for silence analysis (ms)
MIN_WORD_DURATION_S = 0.1   # Minimum duration for a valid word (s)
LOW_CONFIDENCE_THRESHOLD = 0.5  # Words below this confidence are more suspicious

# Set up local model caching directory
MODEL_CACHE_DIR = "./whisper_models"
MODEL_NAME = "tiny"

# Create model cache directory if it doesn't exist
if not os.path.exists(MODEL_CACHE_DIR):
    os.makedirs(MODEL_CACHE_DIR)

# Hindi to Latin transliteration dictionary
hindi_to_latin = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ं': 'n', 'ः': 'h',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'n',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क्ष': 'ksh', 'त्र': 'tra', 'ज्ञ': 'gya',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'े': 'e', 'ै': 'ai', 'о': 'o', 'ौ': 'au',
    '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h', 'ॉ': 'o',
    '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9', '०': '0'
}

def transliterate_hindi(text):
    """Transliterate Hindi text to Latin script"""
    transliterated = ""
    i = 0
    while i < len(text):
        # Check for multi-character combinations first
        if i + 2 <= len(text) and text[i:i+2] in hindi_to_latin:
            transliterated += hindi_to_latin[text[i:i+2]]
            i += 2
        elif text[i] in hindi_to_latin:
            transliterated += hindi_to_latin[text[i]]
            i += 1
        else:
            # Keep non-Hindi characters as is
            transliterated += text[i]
            i += 1
    return transliterated

def validate_audio_file(audio_path):
    """
    Validate that the audio file exists and is readable.
    Returns (is_valid, error_message)
    """
    if not os.path.exists(audio_path):
        return False, f"Audio file does not exist: {audio_path}"

    if not os.path.isfile(audio_path):
        return False, f"Path is not a file: {audio_path}"

    # Check file extension
    valid_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in valid_extensions:
        return False, f"Unsupported audio format: {file_ext}. Supported: {', '.join(valid_extensions)}"

    # Try to open with pydub to verify it's a valid audio file
    try:
        audio = AudioSegment.from_file(audio_path)
        if audio.duration_seconds == 0:
            return False, "Audio file has zero duration"
        if audio.duration_seconds > 36000:  # 10 hours max
            return False, "Audio file is too long (> 10 hours)"
        return True, None
    except Exception as e:
        return False, f"Invalid audio file: {str(e)}"

def cleanup_temp_files(temp_files):
    """
    Clean up temporary files, ignoring any errors.
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"⚠️ Failed to cleanup {temp_file}: {e}")

def analyze_audio_energy(audio_path, window_ms=SILENCE_WINDOW_MS, silence_threshold_db=SILENCE_THRESHOLD_DB):
    """
    Analyze audio file to detect silence regions using RMS energy analysis.
    """
    if not ENHANCED_PROCESSING_AVAILABLE:
        raise ImportError("Enhanced audio processing libraries required for silence detection")

    try:
        # Load audio with librosa
        audio_np, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        audio_duration = len(audio_np) / sample_rate

        # Convert window size to samples
        window_samples = int(window_ms * sample_rate / 1000)
        if window_samples < 1:
            window_samples = 1

        # Calculate RMS energy in sliding windows
        rms_energy = []
        for i in range(0, len(audio_np), window_samples):
            window_end = min(i + window_samples, len(audio_np))
            window_data = audio_np[i:window_end]
            rms = np.sqrt(np.mean(window_data**2))
            rms_energy.append(rms)

        # Convert RMS to dB scale
        rms_energy = np.array(rms_energy)
        rms_energy_db = 20 * np.log10(rms_energy + 1e-10)

        # Create silence mask based on threshold
        silence_mask = rms_energy_db < silence_threshold_db

        # Expand silence mask back to full sample resolution
        full_silence_mask = np.zeros(len(audio_np), dtype=bool)
        for i, is_silent in enumerate(silence_mask):
            window_start = i * window_samples
            window_end = min(window_start + window_samples, len(audio_np))
            full_silence_mask[window_start:window_end] = is_silent

        return full_silence_mask, audio_duration, sample_rate

    except Exception as e:
        raise

def filter_silence_false_positives(word_data, silence_mask, sample_rate, confidence_threshold=0.3):
    """
    Filter out words that occur entirely within detected silence regions.
    """
    if not word_data:
        return []

    filtered_words = []
    for word in word_data:
        start_time = word['Start Time (s)']
        end_time = word['End Time (s)']
        duration = end_time - start_time
        confidence = word['Confidence']

        # Convert time to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Ensure indices are within bounds
        start_sample = max(0, min(start_sample, len(silence_mask) - 1))
        end_sample = max(0, min(end_sample, len(silence_mask) - 1))

        # Check if word duration falls entirely within silence
        word_samples = slice(start_sample, end_sample)
        silence_in_word = silence_mask[word_samples]

        # Determine if this word should be filtered
        should_filter = False

        # Check 1: Word entirely in silence
        if np.all(silence_in_word):
            should_filter = True
        # Check 2: Suspiciously low confidence and short duration
        elif confidence < LOW_CONFIDENCE_THRESHOLD and duration < MIN_WORD_DURATION_S:
            should_filter = True
        # Check 3: Very short words (< 50ms) regardless of confidence if mostly silent
        elif duration < 0.05 and np.mean(silence_in_word) > 0.7:
            should_filter = True

        if not should_filter:
            filtered_words.append(word)

    return filtered_words

def convert_timestamps_to_samples(word_timestamps, sample_rate=16000):
    """
    Convert float timestamp-based word data to sample-based for precision.
    """
    sample_based_data = []
    for word in word_timestamps:
        start_sample = int(round(word['Start Time (s)'] * sample_rate))
        end_sample = int(round(word['End Time (s)'] * sample_rate))

        sample_based_word = word.copy()
        sample_based_word['Start Sample'] = start_sample
        sample_based_word['End Sample'] = end_sample
        sample_based_word['Duration Samples'] = end_sample - start_sample

        sample_based_data.append(sample_based_word)

    return sample_based_data

def load_or_download_model(model_name, cache_dir, device):
    """
    Load model from cache or download if not available.
    """
    model_path = os.path.join(cache_dir, f"{model_name}.pt")

    if os.path.exists(model_path):
        try:
            model = whisper.load_model(model_name, device=device, download_root=cache_dir)
            return model
        except Exception as e:
            pass

    try:
        model = whisper.load_model(model_name, device=device, download_root=cache_dir)
        return model
    except Exception as e:
        return None

def preprocess_audio_librosa(audio_path, target_sr=16000):
    """
    Preprocesses audio using Librosa for frame-accurate processing.
    """
    if not ENHANCED_PROCESSING_AVAILABLE:
        raise ImportError("Enhanced processing libraries not available")

    # Load audio with librosa
    audio_np, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Apply pre-emphasis for speech clarity
    audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)

    # High-pass filter to remove low-frequency noise
    nyquist = sr / 2
    cutoff = 80 / nyquist
    b, a = scipy.signal.butter(4, cutoff, btype='high')
    audio_np = scipy.signal.filtfilt(b, a, audio_np)

    # Normalize amplitude
    audio_np = librosa.util.normalize(audio_np)

    # Compute duration in seconds
    duration = len(audio_np) / sr

    # Convert back to 16-bit PCM for Whisper compatibility
    audio_int16 = np.int16(audio_np * 32767)

    preprocessed_path = f"preprocessed_sample_accurate_{os.path.basename(audio_path)}"
    sf.write(preprocessed_path, audio_int16, sr, format='WAV', subtype='PCM_16')

    return preprocessed_path, duration

def preprocess_audio_pydub(audio_path, target_sr=16000):
    """
    Preprocesses audio using Pydub (millisecond precision).
    """
    audio = AudioSegment.from_file(audio_path)

    if audio.channels > 1:
        audio = audio.set_channels(1)

    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)

    audio = audio.normalize()
    audio = audio.high_pass_filter(80)

    preprocessed_path = f"preprocessed_{os.path.basename(audio_path)}"
    audio.export(preprocessed_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])

    return preprocessed_path, audio.duration_seconds

def preprocess_audio(audio_path, target_sr=16000, use_frame_accurate=False):
    """
    Preprocesses the audio file to a format optimal for Whisper.
    """
    if use_frame_accurate:
        try:
            return preprocess_audio_librosa(audio_path, target_sr)
        except Exception:
            return preprocess_audio_pydub(audio_path, target_sr)
    else:
        return preprocess_audio_pydub(audio_path, target_sr)

def transcribe_audio_with_word_timestamps(audio_path, model, language="hi", confidence_threshold=0.3, use_sample_accurate=False):
    """
    Transcribes audio with high accuracy using VAD and preprocessing.
    Returns processed word data.
    """
    temp_files = []  # Track files to cleanup
    audio_duration = 0

    try:
        # Validate audio file first
        is_valid, error_msg = validate_audio_file(audio_path)
        if not is_valid:
            raise ValueError(f"Audio validation failed: {error_msg}")

        # Use frame-accurate preprocessing if requested
        preprocessed_audio_path, audio_duration = preprocess_audio(audio_path, use_frame_accurate=use_sample_accurate)
        temp_files.append(preprocessed_audio_path)

        # Use enhanced parameters only if we're confident they work
        use_enhanced = ENHANCED_PROCESSING_AVAILABLE and use_sample_accurate

        if use_enhanced:
            # Enhanced mode with all improvements
            result = whisper.transcribe(
                model,
                preprocessed_audio_path,
                language=language,
                vad=True,
                verbose=False,
                condition_on_previous_text=True,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
        else:
            # Basic mode
            result = whisper.transcribe(
                model,
                preprocessed_audio_path,
                language=language,
                vad=True,
                verbose=False,
                condition_on_previous_text=False,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )

        # Analyze original audio for silence detection
        silence_mask = None
        sample_rate = 16000  # Default fallback

        if ENHANCED_PROCESSING_AVAILABLE:
            try:
                silence_mask, detected_duration, sample_rate = analyze_audio_energy(audio_path)
            except Exception:
                silence_mask = None

        # Process transcription results
        final_word_data = []

        for segment in result.get("segments", []):
            if "words" not in segment:
                continue

            for word in segment["words"]:
                try:
                    word_text = word.get('text', '').strip()
                    start_time = round(float(word.get('start', 0)), 4)
                    end_time = round(float(word.get('end', 0)), 4)
                    confidence = round(float(word.get('confidence', 0)), 4)

                    if not word_text or start_time >= end_time or confidence < 0:
                        continue

                    # Transliterate if language is Hindi
                    if language == "hi":
                        transliterated_text = transliterate_hindi(word_text)
                    else:
                        transliterated_text = word_text

                    word_entry = {
                        "Word": transliterated_text,
                        "Start Time (s)": start_time,
                        "End Time (s)": end_time,
                        "Confidence": confidence
                    }

                    final_word_data.append(word_entry)

                except (KeyError, ValueError, TypeError):
                    continue

        if not final_word_data:
            raise ValueError("No valid words were transcribed")

        # Filter out words that occur in silence regions
        if silence_mask is not None and len(final_word_data) > 0:
            try:
                final_word_data = filter_silence_false_positives(final_word_data, silence_mask, sample_rate, confidence_threshold)
            except Exception:
                pass

        # Calculate statistics
        total_words = len(final_word_data)
        high_confidence_words = len([word for word in final_word_data if word['Confidence'] >= confidence_threshold])
        avg_confidence = sum(w['Confidence'] for w in final_word_data) / total_words if total_words > 0 else 0

        # Convert to sample-based timestamps if enabled
        sample_based_data = None
        if use_sample_accurate and final_word_data:
            sample_based_data = convert_timestamps_to_samples(final_word_data)

        # Clean up temporary files
        cleanup_temp_files(temp_files)

        return {
            'word_data': final_word_data,
            'sample_based_data': sample_based_data,
            'total_words': total_words,
            'high_confidence_words': high_confidence_words,
            'average_confidence': avg_confidence,
            'audio_duration': audio_duration,
            'silence_detection_used': silence_mask is not None,
            'transcription_result': result
        }

    except Exception as e:
        # Always cleanup on any error
        cleanup_temp_files(temp_files)
        raise e
