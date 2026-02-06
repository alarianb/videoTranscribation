"""
Конфигурация и константы приложения
"""

import sys
import os
import warnings

# Отключаем предупреждения
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Константы приложения
APP_NAME = "Audio/Video Transcription Pro"
APP_VERSION = "8.0-ENHANCED"
AUTHOR = "Lebedev Nikolay"

# HuggingFace токен для pyannote (можно задать через переменную окружения)
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ============================================
# Настройки диаризации
# ============================================
# Backend: "pyannote" (лучшее качество), "heuristic" (без зависимостей)
DIARIZATION_BACKEND = os.environ.get("DIARIZATION_BACKEND", "auto")  # auto, pyannote, heuristic

# Модель pyannote для диаризации
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# ============================================
# Настройки Whisper
# ============================================
# Temperature fallback для сложных сегментов
WHISPER_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Промпты для разных языков (улучшают качество)
WHISPER_INITIAL_PROMPTS = {
    'ru': "Транскрипция русской речи с правильной пунктуацией.",
    'en': "Transcription of English speech with proper punctuation.",
    'uk': "Транскрипція української мови з правильною пунктуацією.",
    'de': "Transkription deutscher Sprache mit korrekter Interpunktion.",
    'fr': "Transcription de la parole française avec ponctuation correcte.",
}

# Поддерживаемые форматы файлов
SUPPORTED_FORMATS = [
    '.mp4', '.mkv', '.avi', '.mov', '.webm',
    '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'
]

# Цвета для спикеров
SPEAKER_COLORS = [
    '#667eea', '#e53e3e', '#38a169', '#d69e2e', '#9f7aea',
    '#ed8936', '#4299e1', '#48bb78', '#f56565', '#805ad5'
]

# Цвета для спикеров (для форматирования)
SPEAKER_COLORS_MAP = {
    1: '#667eea',  # Синий-фиолетовый
    2: '#ef4444',  # Красный
    3: '#10b981',  # Зеленый
    4: '#f59e0b',  # Оранжевый
    5: '#a78bfa',  # Светло-фиолетовый
    6: '#06b6d4',  # Голубой
    7: '#f97316',  # Оранжевый яркий
    8: '#84cc16',  # Лайм
    9: '#ec4899',  # Розовый
    10: '#8b5cf6', # Фиолетовый
}

# Проверка зависимостей
FASTER_WHISPER_AVAILABLE = False
DOCX_AVAILABLE = False
TORCH_AVAILABLE = False
DEVICE = "cpu"

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("ОШИБКА: faster_whisper не установлен!")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Безопасная инициализация GPU
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if sys.platform == "win32":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(f"PyTorch: {torch.__version__}, Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("PyTorch не установлен - CPU режим")

# Проверка pyannote-audio
PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    print("pyannote-audio: доступен")
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("pyannote-audio не установлен - используется эвристическая диаризация")

# Автоматический выбор backend диаризации
def get_diarization_backend():
    """Определяет лучший доступный backend для диаризации"""
    if DIARIZATION_BACKEND == "pyannote" and PYANNOTE_AVAILABLE:
        return "pyannote"
    elif DIARIZATION_BACKEND == "heuristic":
        return "heuristic"
    elif DIARIZATION_BACKEND == "auto":
        if PYANNOTE_AVAILABLE and DEFAULT_HF_TOKEN:
            return "pyannote"
        return "heuristic"
    return "heuristic"
