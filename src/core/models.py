"""
Работа с моделями Whisper
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List

from ..utils import DEVICE, FASTER_WHISPER_AVAILABLE, CrashSafeMemoryManager

if FASTER_WHISPER_AVAILABLE:
    from faster_whisper import WhisperModel

# Mapping модели -> название в Hugging Face
MODEL_REPO_MAPPING = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# Примерные размеры моделей в MB (для информации)
MODEL_SIZES_MB = {
    "tiny": 75,
    "base": 145,
    "small": 465,
    "medium": 1500,
    "large": 3100,
    "large-v2": 3100,
    "large-v3": 3100,
}

# ============================================
# Кеширование модели в памяти
# ============================================
_cached_model = None
_cached_model_name = None
_cached_model_device = None


def get_cached_model():
    """Возвращает кешированную модель, если есть"""
    return _cached_model, _cached_model_name


def clear_model_cache(log_func=None):
    """Очищает кеш модели из памяти"""
    global _cached_model, _cached_model_name, _cached_model_device

    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)

    if _cached_model is not None:
        try:
            model_name = _cached_model_name
            del _cached_model
            _cached_model = None
            _cached_model_name = None
            _cached_model_device = None
            CrashSafeMemoryManager.safe_gpu_cleanup("after model cache clear")
            log(f"Модель {model_name} выгружена из кеша", "INFO")
        except Exception as e:
            log(f"Ошибка очистки кеша модели: {e}", "WARNING")


def get_models_cache_dir() -> Path:
    """
    Получить единую директорию для кэша моделей

    Все модели хранятся в одном месте:
    - Windows: %LOCALAPPDATA%/WhisperModels
    - Linux/Mac: ~/.cache/whisper

    Также устанавливает HF_HOME для huggingface_hub
    """
    if sys.platform == "win32":
        cache_dir = Path.home() / "AppData" / "Local" / "WhisperModels"
    else:
        cache_dir = Path.home() / ".cache" / "whisper"

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Устанавливаем единый путь для huggingface_hub
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)

    return cache_dir


def is_model_downloaded(model_name: str) -> bool:
    """
    Проверяет, скачана ли модель локально

    Args:
        model_name: Название модели (tiny, base, small, medium, large-v3)

    Returns:
        bool: True если модель уже скачана
    """
    cache_dir = get_models_cache_dir()

    # Получаем имя репозитория
    repo_name = MODEL_REPO_MAPPING.get(model_name, f"Systran/faster-whisper-{model_name}")

    # faster-whisper использует huggingface_hub для кеширования
    # Модели хранятся в формате: models--{org}--{name}
    hf_cache_name = repo_name.replace("/", "--")
    model_dir = cache_dir / f"models--{hf_cache_name}"

    if model_dir.exists():
        # Проверяем наличие основных файлов модели
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            # Проверяем наличие хотя бы одного снэпшота с model.bin
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    model_file = snapshot / "model.bin"
                    if model_file.exists():
                        return True

    # Также проверяем старый формат (прямое имя)
    old_format_dir = cache_dir / model_name
    if old_format_dir.exists():
        model_file = old_format_dir / "model.bin"
        if model_file.exists():
            return True

    return False


def get_downloaded_models() -> List[str]:
    """
    Возвращает список скачанных моделей

    Returns:
        List[str]: Список названий скачанных моделей
    """
    downloaded = []
    for model_name in MODEL_REPO_MAPPING.keys():
        if is_model_downloaded(model_name):
            downloaded.append(model_name)
    return downloaded


def get_model_info(model_name: str) -> Dict:
    """
    Получить информацию о модели

    Args:
        model_name: Название модели

    Returns:
        Dict с информацией: downloaded, size_mb, repo
    """
    return {
        "name": model_name,
        "downloaded": is_model_downloaded(model_name),
        "size_mb": MODEL_SIZES_MB.get(model_name, 0),
        "repo": MODEL_REPO_MAPPING.get(model_name, f"Systran/faster-whisper-{model_name}")
    }


def get_all_models_info() -> List[Dict]:
    """
    Получить информацию обо всех моделях

    Returns:
        List[Dict]: Список с информацией о каждой модели
    """
    return [get_model_info(name) for name in MODEL_REPO_MAPPING.keys()]


def load_model(model_size: str, log_func=None, use_cache: bool = True):
    """
    Безопасная загрузка модели Whisper с кешированием

    Args:
        model_size: Размер модели (tiny, base, small, medium, large-v3)
        log_func: Функция для логирования (опционально)
        use_cache: Использовать кеширование модели (по умолчанию True)

    Returns:
        WhisperModel или None при ошибке
    """
    global _cached_model, _cached_model_name, _cached_model_device

    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper не установлен!")

    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    # Проверяем кеш
    if use_cache and _cached_model is not None:
        if _cached_model_name == model_size and _cached_model_device == DEVICE:
            log(f"Используем кешированную модель {model_size}", "INFO")
            return _cached_model
        else:
            # Нужна другая модель - очищаем кеш
            log(f"Выгружаем модель {_cached_model_name} для загрузки {model_size}", "INFO")
            clear_model_cache(log_func)

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

        log(f"Загрузка модели {model_size} ({compute_type})...", "INFO")

        model = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=compute_type,
            cpu_threads=min(4, os.cpu_count() or 4),
            num_workers=1,
            download_root=get_models_cache_dir()
        )

        # Кешируем модель
        if use_cache:
            _cached_model = model
            _cached_model_name = model_size
            _cached_model_device = DEVICE
            log(f"Модель {model_size} загружена и закеширована", "SUCCESS")
        else:
            log(f"Модель {model_size} загружена (без кеширования)", "SUCCESS")

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


def download_model(model_name: str, progress_callback=None, log_func=None, force=False):
    """
    Скачивание модели Whisper

    Args:
        model_name: Название модели
        progress_callback: Функция для отчета о прогрессе (value, message)
        log_func: Функция для логирования
        force: Принудительно скачать, даже если модель уже есть

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
        progress(5, "Проверка наличия модели...")

        # Проверяем, скачана ли модель
        if not force and is_model_downloaded(model_name):
            model_info = get_model_info(model_name)
            log(f"Модель {model_name} ({model_info['size_mb']} MB) уже скачана", "INFO")
            progress(100, "Модель уже установлена!")
            return True

        size_mb = MODEL_SIZES_MB.get(model_name, 0)
        log(f"Скачивание модели {model_name} (~{size_mb} MB)...", "INFO")
        progress(10, f"Скачивание {model_name} (~{size_mb} MB)...")

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
        log(f"Модель {model_name} успешно загружена", "SUCCESS")
        return True

    except Exception as e:
        log(f"Ошибка загрузки модели: {str(e)}", "ERROR")
        CrashSafeMemoryManager.safe_gpu_cleanup("after model download error")
        return False
