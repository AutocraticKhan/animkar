import os
import shutil
import torch
import json
import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper_timestamped as whisper
import numpy as np
from datetime import datetime

# Optional enhanced processing libraries
try:
    import librosa
    import scipy.signal
    import soundfile as sf
    ENHANCED_PROCESSING = True
except ImportError:
    ENHANCED_PROCESSING = False

# Configuration
MODEL_CACHE_DIR = "./whisper_models"
MODEL_NAME = "large-v3"  # Better model for accuracy
SILENCE_THRESH = -40  # dB threshold for silence detection
MIN_SILENCE_LEN = 500  # Minimum silence length in ms
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for words

# Hindi to Latin transliteration dictionary
HINDI_TO_LATIN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ं': 'n', 'ः': 'h',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क्ष': 'ksh', 'त्र': 'tra', 'ज्ञ': 'gya',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h', 'ॉ': 'o',
    '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9', '०': '0'
}

def has_hindi_characters(text):
    """Check if text contains Hindi/Devanagari characters."""
    if not text:
        return False
    for char in text:
        if '\u0900' <= char <= '\u097F':  # Devanagari Unicode range
            return True
    return False

def transliterate_hindi_to_latin(text):
    """Convert Hindi/Devanagari text to Latin script."""
    if not text:
        return text

    # If no Hindi characters, return as is
    if not has_hindi_characters(text):
        return text

    transliterated = ""
    i = 0
    while i < len(text):
        # Check for multi-character combinations first
        if i + 2 <= len(text) and text[i:i+2] in HINDI_TO_LATIN:
            transliterated += HINDI_TO_LATIN[text[i:i+2]]
            i += 2
        elif text[i] in HINDI_TO_LATIN:
            transliterated += HINDI_TO_LATIN[text[i]]
            i += 1
        else:
            # Keep non-Hindi characters as is (English, numbers, punctuation)
            transliterated += text[i]
            i += 1

    return transliterated.strip()

def preprocess_audio(audio_path, target_sr=16000):
    """
    Preprocess audio for optimal Whisper transcription.
    Uses enhanced processing if available, otherwise falls back to basic.
    """
    if ENHANCED_PROCESSING:
        try:
            # Enhanced preprocessing with librosa
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

            # Convert to 16-bit PCM
            audio_int16 = np.int16(audio_np * 32767)

            preprocessed_path = f"preprocessed_{os.path.basename(audio_path)}"
            sf.write(preprocessed_path, audio_int16, sr, format='WAV', subtype='PCM_16')

            return preprocessed_path, len(audio_np) / sr

        except Exception as e:
            print(f"⚠️ Enhanced preprocessing failed: {e}. Using basic preprocessing.")

    # Basic preprocessing with pydub
    audio = AudioSegment.from_file(audio_path)

    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Set sample rate
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)

    # Normalize and filter
    audio = audio.normalize()
    audio = audio.high_pass_filter(80)

    preprocessed_path = f"preprocessed_{os.path.basename(audio_path)}"
    audio.export(preprocessed_path, format="wav", parameters=["-ar", str(target_sr), "-ac", "1"])

    return preprocessed_path, audio.duration_seconds

def split_audio_with_precise_timestamps(audio_path, silence_thresh=-40, min_silence_len=500, keep_silence=200):
    """
    Split audio while maintaining precise timestamp information for each chunk.
    """
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)

    # Detect non-silent ranges
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if not nonsilent_ranges:
        print("⚠️ Warning: No speech detected in audio file!")
        return []

    print(f"✓ Detected {len(nonsilent_ranges)} speech segments")

    chunk_info = []

    for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
        # Add padding around the non-silent segment
        chunk_start_ms = max(0, start_ms - keep_silence)
        chunk_end_ms = min(total_duration_ms, end_ms + keep_silence)

        chunk = audio[chunk_start_ms:chunk_end_ms]
        chunk_duration_ms = len(chunk)

        actual_start_padding = start_ms - chunk_start_ms
        actual_end_padding = chunk_end_ms - end_ms

        chunk_info.append({
            "chunk": chunk,
            "chunk_index": i,
            "absolute_start_ms": chunk_start_ms,
            "absolute_end_ms": chunk_end_ms,
            "duration_ms": chunk_duration_ms,
            "speech_start_in_chunk_ms": actual_start_padding,
            "speech_end_in_chunk_ms": chunk_duration_ms - actual_end_padding,
            "original_speech_start_ms": start_ms,
            "original_speech_end_ms": end_ms
        })

    return chunk_info

def save_chunks(chunks_info, output_dir="temp_chunks"):
    """Save audio chunks to temporary files."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    chunk_files = []
    for chunk_data in chunks_info:
        i = chunk_data["chunk_index"]
        chunk_file = f"{output_dir}/chunk_{i:04d}.wav"
        chunk_data["chunk"].export(chunk_file, format="wav")
        chunk_files.append(chunk_file)

    return chunk_files

def process_chunk_with_whisper(model, chunk_file, chunk_info, language="hi"):
    """
    Transcribe chunk and map word timestamps back to original audio timeline.
    Automatically transliterates Hindi to Latin.
    """
    result = whisper.transcribe(
        model,
        chunk_file,
        language=language,
        vad=True,
        verbose=False,
        condition_on_previous_text=True,
        temperature=[0.0, 0.2, 0.4],
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6
    )

    words_with_timestamps = []
    chunk_start_ms = chunk_info["absolute_start_ms"]

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            word_text = word.get("text", "").strip()
            if not word_text:
                continue

            # Get original text (before transliteration)
            original_text = word_text

            # Automatically transliterate if Hindi text is detected
            transliterated_text = transliterate_hindi_to_latin(word_text)
            if not transliterated_text:  # Skip if transliteration resulted in empty string
                continue

            word_start_in_chunk_s = word["start"]
            word_end_in_chunk_s = word["end"]

            word_start_in_chunk_ms = word_start_in_chunk_s * 1000
            word_end_in_chunk_ms = word_end_in_chunk_s * 1000

            absolute_start_ms = chunk_start_ms + word_start_in_chunk_ms
            absolute_end_ms = chunk_start_ms + word_end_in_chunk_ms

            confidence = word.get("confidence", word.get("probability", 1.0))

            # Detect if word was transliterated
            was_transliterated = has_hindi_characters(original_text)

            words_with_timestamps.append({
                "word": transliterated_text,
                "original_script": original_text if was_transliterated else transliterated_text,
                "start_time_s": absolute_start_ms / 1000,
                "end_time_s": absolute_end_ms / 1000,
                "duration_s": (absolute_end_ms - absolute_start_ms) / 1000,
                "confidence": confidence,
                "transliterated": was_transliterated,
                "chunk_index": chunk_info["chunk_index"]
            })

    return words_with_timestamps

def validate_and_fix_timestamps(words_data, confidence_threshold=0.3):
    """
    Validate timestamp continuity and filter low-confidence words.
    """
    if not words_data:
        return words_data

    filtered_data = []
    issues_fixed = 0
    words_filtered = 0

    for i, word in enumerate(words_data):
        current_word = word.copy()

        # Filter out very low confidence words
        if current_word["confidence"] < confidence_threshold:
            words_filtered += 1
            continue

        # Check if timestamps are valid
        if current_word["start_time_s"] >= current_word["end_time_s"]:
            print(f"⚠️ Invalid timestamp for word '{current_word['word']}' at index {i}")
            current_word["end_time_s"] = current_word["start_time_s"] + 0.1
            issues_fixed += 1

        # Check for overlaps with previous word
        if filtered_data:
            prev_word = filtered_data[-1]
            time_gap = current_word["start_time_s"] - prev_word["end_time_s"]

            if time_gap < 0:  # Overlap
                current_word["start_time_s"] = prev_word["end_time_s"]
                current_word["end_time_s"] = max(current_word["end_time_s"], current_word["start_time_s"] + 0.1)
                issues_fixed += 1

        filtered_data.append(current_word)

    if words_filtered > 0:
        print(f"✓ Filtered {words_filtered} low-confidence words (< {confidence_threshold})")
    if issues_fixed > 0:
        print(f"✓ Fixed {issues_fixed} timestamp issues")

    return filtered_data

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

def transcribe_audio_with_accurate_timestamps(
    audio_path,
    language="hi",
    silence_thresh=-40,
    min_silence_len=500,
    confidence_threshold=0.3
):
    """
    Complete transcription pipeline with accurate timestamps and automatic Hindi-to-Latin transliteration.
    """
    print("="*80)
    print("AUDIO TRANSCRIPTION WITH ACCURATE TIMESTAMPS")
    print("Hindi text will be automatically converted to Latin script")
    print("="*80)

    # Validate audio file
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    temp_files = []

    try:
        # Load Whisper model
        print(f"\n[1/7] Loading Whisper model: {MODEL_NAME}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if device == "cpu":
            print("⚠️  Warning: Running on CPU. This will be VERY slow with large-v3.")
            print("    Consider using a smaller model (base/small) or enabling GPU.")

        try:
            model = whisper.load_model(MODEL_NAME, device=device)
            print(f"✓ Loaded {MODEL_NAME} successfully")
        except Exception as e:
            print(f"⚠️ Failed to load {MODEL_NAME}: {e}")
            print("   Falling back to 'base' model...")
            model = whisper.load_model("base", device=device)

        # Preprocess audio
        print(f"\n[2/7] Preprocessing audio...")
        preprocessed_audio, audio_duration = preprocess_audio(audio_path)
        temp_files.append(preprocessed_audio)
        print(f"✓ Audio duration: {audio_duration:.2f} seconds")

        # Split audio
        print(f"\n[3/7] Splitting audio (threshold={silence_thresh}dB, min_silence={min_silence_len}ms)...")
        chunks_info = split_audio_with_precise_timestamps(
            preprocessed_audio,
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            keep_silence=200
        )

        if not chunks_info:
            raise ValueError("No speech detected. Try adjusting silence_thresh parameter.")

        print(f"✓ Created {len(chunks_info)} chunks")

        # Save chunks
        print(f"\n[4/7] Saving temporary chunks...")
        chunk_files = save_chunks(chunks_info)

        # Transcribe chunks
        print(f"\n[5/7] Transcribing chunks (language: {language.upper()})...")
        all_words = []
        total_transliterated = 0

        for i, (chunk_file, chunk_info) in enumerate(zip(chunk_files, chunks_info)):
            print(f"  Chunk {i+1}/{len(chunks_info)}: {chunk_info['absolute_start_ms']/1000:.2f}s - {chunk_info['absolute_end_ms']/1000:.2f}s", end="")

            words = process_chunk_with_whisper(model, chunk_file, chunk_info, language)
            all_words.extend(words)

            # Count transliterated words in this chunk
            chunk_transliterated = sum(1 for w in words if w['transliterated'])
            total_transliterated += chunk_transliterated

            print(f" → {len(words)} words ({chunk_transliterated} transliterated)")

        # Validate timestamps
        print(f"\n[6/7] Validating {len(all_words)} word timestamps...")
        all_words = validate_and_fix_timestamps(all_words, confidence_threshold)

        # Clean up
        print(f"\n[7/7] Cleaning up temporary files...")
        if os.path.exists("temp_chunks"):
            shutil.rmtree("temp_chunks")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Display summary
        print("\n" + "="*80)
        print("TRANSCRIPTION COMPLETE")
        print("="*80)
        print(f"Language: {language.upper()}")
        print(f"Total words: {len(all_words)}")
        print(f"Words transliterated (Hindi→Latin): {total_transliterated}")
        print(f"Words kept as-is (English/numbers): {len(all_words) - total_transliterated}")

        if all_words:
            print(f"Time range: {all_words[0]['start_time_s']:.2f}s - {all_words[-1]['end_time_s']:.2f}s")
            print(f"Average confidence: {np.mean([w['confidence'] for w in all_words]):.3f}")

        # Show sample with both scripts
        print("\nSample transcription (first 15 words):")
        print("-" * 80)
        for i, word in enumerate(all_words[:15]):
            if word['transliterated']:
                print(f"{word['word']:<25} [{word['start_time_s']:>7.2f}s - {word['end_time_s']:>7.2f}s] (conf: {word['confidence']:.3f}) [Hindi: {word['original_script']}]")
            else:
                print(f"{word['word']:<25} [{word['start_time_s']:>7.2f}s - {word['end_time_s']:>7.2f}s] (conf: {word['confidence']:.3f})")

        return all_words

    except Exception as e:
        # Clean up on error
        if os.path.exists("temp_chunks"):
            shutil.rmtree("temp_chunks")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise e

# Legacy function for backward compatibility
def transcribe_audio_with_word_timestamps(audio_path, model, language="hi", confidence_threshold=0.3, use_sample_accurate=False):
    """
    Legacy wrapper for Django compatibility
    """
    words_data = transcribe_audio_with_accurate_timestamps(
        audio_path=audio_path,
        language=language,
        confidence_threshold=confidence_threshold
    )

    # Convert to expected format
    word_data = []
    for word in words_data:
        word_entry = {
            "Word": word["word"],
            "Start Time (s)": word["start_time_s"],
            "End Time (s)": word["end_time_s"],
            "Confidence": word["confidence"]
        }
        word_data.append(word_entry)

    # Calculate statistics
    total_words = len(word_data)
    high_confidence_words = len([word for word in word_data if word['Confidence'] >= confidence_threshold])
    avg_confidence = sum(w['Confidence'] for w in word_data) / total_words if total_words > 0 else 0

    return {
        'word_data': word_data,
        'total_words': total_words,
        'high_confidence_words': high_confidence_words,
        'average_confidence': avg_confidence,
        'audio_duration': words_data[-1]['end_time_s'] if words_data else 0,
    }
