"""
Работа с моделями Whisper
"""

import sys
import os
from pathlib import Path

from ..utils import DEVICE, FASTER_WHISPER_AVAILABLE, CrashSafeMemoryManager

if FASTER_WHISPER_AVAILABLE:
    from faster_whisper import WhisperModel


def get_models_cache_dir() -> Path:
    """Получить директорию для кэша моделей"""
    if sys.platform == "win32":
        cache_dir = Path.home() / "AppData" / "Local" / "WhisperModels"
    else:
        cache_dir = Path.home() / ".cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_model(model_size: str, log_func=None):
    """
    Безопасная загрузка модели Whisper

    Args:
        model_size: Размер модели (tiny, base, small, medium, large-v3)
        log_func: Функция для логирования (опционально)

    Returns:
        WhisperModel или None при ошибке
    """
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper не установлен!")

    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    try:
        compute_type = "float16" if DEVICE == "cuda" else "int8"

        # Для больших моделей на слабых GPU
        if DEVICE == "cuda" and model_size in ['large', 'large-v2', 'large-v3']:
            try:
                import torch
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram < 10:
                    compute_type = "int8"
                    log(f"GPU память: {vram:.1f}GB. Используем int8", "WARNING")
            except:
                pass

        log(f"Создание модели с {compute_type}", "INFO")

        model = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=compute_type,
            cpu_threads=min(4, os.cpu_count() or 4),
            num_workers=1,
            download_root=get_models_cache_dir()
        )

        log(f"Модель загружена (compute_type: {compute_type})", "SUCCESS")
        return model

    except Exception as e:
        log(f"Ошибка загрузки модели: {e}", "ERROR")
        try:
            log("Пробуем базовую модель...", "WARNING")
            model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
                download_root=get_models_cache_dir()
            )
            log("Базовая модель загружена в безопасном режиме", "WARNING")
            return model
        except Exception as fallback_e:
            log(f"Критическая ошибка загрузки: {fallback_e}", "ERROR")
            raise fallback_e


def download_model(model_name: str, progress_callback=None, log_func=None):
    """
    Скачивание модели Whisper

    Args:
        model_name: Название модели
        progress_callback: Функция для отчета о прогрессе (value, message)
        log_func: Функция для логирования

    Returns:
        bool: True если успешно, False если ошибка
    """
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper не установлен!")

    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    def progress(value, message):
        if progress_callback:
            progress_callback(value, message)

    try:
        log(f"Скачивание модели {model_name}...", "INFO")
        progress(10, "Проверка модели...")

        progress(30, "Загрузка модели...")

        # Безопасное создание модели
        compute_type = "float16" if DEVICE == "cuda" else "int8"

        # Предварительная очистка памяти
        CrashSafeMemoryManager.safe_gpu_cleanup("before model download")

        model = WhisperModel(
            model_name,
            device=DEVICE,
            compute_type=compute_type,
            cpu_threads=min(4, os.cpu_count() or 4),
            num_workers=1,
            download_root=get_models_cache_dir()
        )

        progress(80, "Проверка работоспособности...")

        # Безопасное удаление модели
        del model
        CrashSafeMemoryManager.safe_gpu_cleanup("after model download")

        progress(100, "Модель готова!")
        return True

    except Exception as e:
        log(f"Ошибка загрузки модели: {str(e)}", "ERROR")
        CrashSafeMemoryManager.safe_gpu_cleanup("after model download error")
        return False
