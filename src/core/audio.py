"""
Работа с аудио файлами
"""

import sys
import os
import subprocess
from pathlib import Path

# Профили аудио фильтров для разных сценариев
AUDIO_FILTER_PROFILES = {
    # Мягкий фильтр - сохраняет больше частот (рекомендуется по умолчанию)
    "soft": "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11",

    # Средний фильтр - баланс между качеством и подавлением шума
    "medium": "highpass=f=100,lowpass=f=7000,loudnorm=I=-16:TP=-1.5:LRA=11",

    # Жесткий фильтр - максимальное подавление шума (для шумных записей)
    "aggressive": "highpass=f=200,lowpass=f=3000,loudnorm=I=-16:TP=-1.5:LRA=11",

    # Без фильтров - только нормализация громкости
    "none": "loudnorm=I=-16:TP=-1.5:LRA=11",
}

DEFAULT_AUDIO_FILTER = "soft"


def find_ffmpeg():
    """Поиск FFmpeg в системе"""
    # Проверяем локальную копию
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys.executable).parent
    else:
        app_dir = Path(__file__).parent.parent.parent

    # Проверяем разные возможные места
    for ffmpeg_name in ['ffmpeg.exe', 'ffmpeg']:
        local_ffmpeg = app_dir / ffmpeg_name
        if local_ffmpeg.exists():
            return str(local_ffmpeg)

    # Проверяем системный PATH
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        if result.returncode == 0:
            return 'ffmpeg'
    except:
        pass

    return None


def extract_audio(input_path: str, output_path: str, log_func=None, filter_profile: str = None):
    """
    Извлечение аудио из видео/аудио файла

    Args:
        input_path: Путь к исходному файлу
        output_path: Путь для сохранения аудио (WAV)
        log_func: Функция для логирования
        filter_profile: Профиль аудио фильтров ("soft", "medium", "aggressive", "none")
                       По умолчанию "soft" - мягкий bandpass 80-8000 Hz

    Raises:
        RuntimeError: Если FFmpeg не найден или произошла ошибка
    """
    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    ffmpeg_exe = find_ffmpeg()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg не найден! Установите FFmpeg или поместите ffmpeg.exe в папку с программой")

    # Выбираем профиль фильтров
    profile = filter_profile or DEFAULT_AUDIO_FILTER
    audio_filter = AUDIO_FILTER_PROFILES.get(profile, AUDIO_FILTER_PROFILES[DEFAULT_AUDIO_FILTER])

    log(f"Используем профиль аудио: {profile}", "INFO")

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        "-vn",                      # Без видео
        "-acodec", "pcm_s16le",     # PCM 16-bit
        "-ar", "16000",             # 16 kHz (оптимально для Whisper)
        "-ac", "1",                 # Моно
        "-af", audio_filter,        # Аудио фильтры
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка FFmpeg: {result.stderr}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"Аудио извлечено: {size_mb:.1f} МБ (профиль: {profile})", "SUCCESS")


def check_ffmpeg() -> bool:
    """Проверка наличия FFmpeg"""
    return find_ffmpeg() is not None
