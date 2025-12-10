"""
Утилиты приложения
"""

from .config import (
    APP_NAME, APP_VERSION, AUTHOR, DEFAULT_HF_TOKEN,
    SUPPORTED_FORMATS, SPEAKER_COLORS, SPEAKER_COLORS_MAP,
    FASTER_WHISPER_AVAILABLE, DOCX_AVAILABLE, TORCH_AVAILABLE, DEVICE
)
from .memory import CrashSafeMemoryManager, MEMORY_MUTEX
from .formatters import (
    format_time, format_time_bracket, clean_text,
    format_simple_text, format_diarized_text, extract_plain_text
)
from .text_processing import (
    post_process_text, process_segment,
    restore_capitalization, restore_punctuation,
    convert_numbers, fix_asr_errors, clean_repetitions
)

__all__ = [
    'APP_NAME', 'APP_VERSION', 'AUTHOR', 'DEFAULT_HF_TOKEN',
    'SUPPORTED_FORMATS', 'SPEAKER_COLORS', 'SPEAKER_COLORS_MAP',
    'FASTER_WHISPER_AVAILABLE', 'DOCX_AVAILABLE', 'TORCH_AVAILABLE', 'DEVICE',
    'CrashSafeMemoryManager', 'MEMORY_MUTEX',
    'format_time', 'format_time_bracket', 'clean_text',
    'format_simple_text', 'format_diarized_text', 'extract_plain_text',
    'post_process_text', 'process_segment',
    'restore_capitalization', 'restore_punctuation',
    'convert_numbers', 'fix_asr_errors', 'clean_repetitions'
]
