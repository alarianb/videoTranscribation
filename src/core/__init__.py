"""
Core модуль - бизнес-логика приложения
"""

from .models import (
    get_models_cache_dir, load_model, download_model,
    is_model_downloaded, get_downloaded_models, get_model_info, get_all_models_info,
    get_cached_model, clear_model_cache,
    MODEL_REPO_MAPPING, MODEL_SIZES_MB
)
from .audio import (
    find_ffmpeg, extract_audio, check_ffmpeg,
    AUDIO_FILTER_PROFILES, DEFAULT_AUDIO_FILTER
)
from .transcription import (
    transcribe_audio, validate_segment,
    finalize_transcription, get_full_text
)
from .nemo_asr import load_nemo_model, transcribe_nemo_audio
from .diarization import apply_diarization

__all__ = [
    # Models
    'get_models_cache_dir', 'load_model', 'download_model',
    'is_model_downloaded', 'get_downloaded_models', 'get_model_info', 'get_all_models_info',
    'get_cached_model', 'clear_model_cache',
    'MODEL_REPO_MAPPING', 'MODEL_SIZES_MB',
    # Audio
    'find_ffmpeg', 'extract_audio', 'check_ffmpeg',
    'AUDIO_FILTER_PROFILES', 'DEFAULT_AUDIO_FILTER',
    # Transcription
    'transcribe_audio', 'validate_segment',
    'finalize_transcription', 'get_full_text',
    'load_nemo_model', 'transcribe_nemo_audio',
    # Diarization
    'apply_diarization'
]
