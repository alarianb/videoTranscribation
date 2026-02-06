"""
Интеграция NVIDIA NeMo ASR (Conformer)
"""

import wave
from typing import List, Optional

from ..utils import DEVICE, clean_text, process_segment

_cached_model = None
_cached_model_name = None


def _get_wav_duration(audio_path: str) -> float:
    """Возвращает длительность WAV в секундах."""
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        if rate == 0:
            return 0.0
        return frames / float(rate)


def load_nemo_model(model_name: str, log_func=None, use_cache: bool = True):
    """
    Загрузка модели NeMo ASR с кешированием.

    Args:
        model_name: Имя модели из NGC/NeMo (например, stt_en_conformer_ctc_large)
        log_func: Функция логирования
        use_cache: Использовать кеш модели
    """
    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    global _cached_model, _cached_model_name

    if use_cache and _cached_model is not None and _cached_model_name == model_name:
        log(f"Используем кешированную NeMo модель {model_name}", "INFO")
        return _cached_model

    log(f"Загрузка NeMo модели: {model_name}", "INFO")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
    model.eval()

    if DEVICE == "cuda":
        model = model.to("cuda")
        log("NeMo модель переведена на CUDA", "INFO")

    _cached_model = model
    _cached_model_name = model_name
    log(f"NeMo модель {model_name} загружена", "SUCCESS")
    return model


def transcribe_nemo_audio(
    audio_path: str,
    model,
    settings: dict,
    log_func=None
) -> List[dict]:
    """
    Транскрибация аудио с помощью NeMo ASR.

    Возвращает список сегментов (одним блоком), поскольку NeMo CTC
    не предоставляет таймкодов по умолчанию.
    """
    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    language = settings.get("language", "auto")

    log("Запуск NeMo транскрибации...", "INFO")
    texts = model.transcribe([audio_path], batch_size=1)
    text = texts[0] if texts else ""
    text = clean_text(text.strip())
    text = process_segment(text, language=language)

    duration = _get_wav_duration(audio_path)

    if not text:
        return []

    return [{
        "start": 0.0,
        "end": float(duration),
        "text": text
    }]
