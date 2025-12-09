"""
Работа с аудио файлами
"""

import sys
import os
import subprocess
from pathlib import Path


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


def extract_audio(input_path: str, output_path: str, log_func=None):
    """
    Извлечение аудио из видео/аудио файла

    Args:
        input_path: Путь к исходному файлу
        output_path: Путь для сохранения аудио (WAV)
        log_func: Функция для логирования

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

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-af", "highpass=f=200,lowpass=f=3000",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка FFmpeg: {result.stderr}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"Аудио извлечено: {size_mb:.1f} МБ", "SUCCESS")


def check_ffmpeg() -> bool:
    """Проверка наличия FFmpeg"""
    return find_ffmpeg() is not None
