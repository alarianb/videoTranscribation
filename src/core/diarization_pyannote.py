"""
Диаризация с использованием pyannote-audio

Требует:
- pyannote-audio>=3.1.0
- HuggingFace токен с доступом к модели pyannote/speaker-diarization-3.1

ВНИМАНИЕ: pyannote требует ~2-4GB RAM на CPU. При нехватке памяти
используйте эвристическую диаризацию.
"""

import gc
from typing import List, Dict, Optional, Callable
from ..utils import (
    PYANNOTE_AVAILABLE, DEFAULT_HF_TOKEN, PYANNOTE_MODEL,
    SPEAKER_COLORS, CrashSafeMemoryManager
)

if PYANNOTE_AVAILABLE:
    from pyannote.audio import Pipeline
    import torch

# Кеш для пайплайна диаризации
_diarization_pipeline = None
_pipeline_device = None


def is_pyannote_ready() -> bool:
    """Проверяет, готов ли pyannote к использованию"""
    return PYANNOTE_AVAILABLE and bool(DEFAULT_HF_TOKEN)


def check_memory_available(required_gb: float = 2.0) -> bool:
    """Проверяет, достаточно ли памяти для pyannote"""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        return available_gb >= required_gb
    except:
        return True  # Если не можем проверить, пробуем


def load_diarization_pipeline(
    hf_token: str = None,
    device: str = None,
    log_func: Callable = None
) -> Optional[object]:
    """
    Загружает пайплайн диаризации pyannote

    Args:
        hf_token: HuggingFace токен (или используется DEFAULT_HF_TOKEN)
        device: Устройство (cuda/cpu, по умолчанию auto)
        log_func: Функция логирования

    Returns:
        Pipeline или None при ошибке
    """
    global _diarization_pipeline, _pipeline_device

    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    if not PYANNOTE_AVAILABLE:
        log("pyannote-audio не установлен!", "ERROR")
        return None

    token = hf_token or DEFAULT_HF_TOKEN
    if not token:
        log("HuggingFace токен не указан! Установите HF_TOKEN или передайте токен", "ERROR")
        return None

    # Проверяем доступную память
    if not check_memory_available(2.0):
        log("Недостаточно памяти для pyannote (нужно ~2GB). Используйте heuristic.", "ERROR")
        return None

    # Определяем устройство
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Используем кеш если модель уже загружена
    if _diarization_pipeline is not None and _pipeline_device == device:
        log("Используем кешированный пайплайн диаризации", "INFO")
        return _diarization_pipeline

    try:
        log(f"Загрузка pyannote модели {PYANNOTE_MODEL} ({device})...", "INFO")
        log("ВНИМАНИЕ: pyannote требует ~2-4GB RAM", "WARNING")

        # Очистка памяти перед загрузкой
        gc.collect()
        CrashSafeMemoryManager.safe_gpu_cleanup("before pyannote load")

        pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            use_auth_token=token
        )

        # Переносим на устройство
        pipeline.to(torch.device(device))

        _diarization_pipeline = pipeline
        _pipeline_device = device

        log(f"pyannote загружен на {device}", "SUCCESS")
        return pipeline

    except Exception as e:
        log(f"Ошибка загрузки pyannote: {e}", "ERROR")
        CrashSafeMemoryManager.safe_gpu_cleanup("after pyannote error")
        return None


def unload_diarization_pipeline(log_func: Callable = None):
    """Выгружает пайплайн диаризации из памяти"""
    global _diarization_pipeline, _pipeline_device

    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)

    if _diarization_pipeline is not None:
        try:
            del _diarization_pipeline
            _diarization_pipeline = None
            _pipeline_device = None
            CrashSafeMemoryManager.safe_gpu_cleanup("after pyannote unload")
            log("pyannote пайплайн выгружен", "INFO")
        except Exception as e:
            log(f"Ошибка выгрузки pyannote: {e}", "WARNING")


def diarize_audio(
    audio_path: str,
    min_speakers: int = None,
    max_speakers: int = None,
    hf_token: str = None,
    log_func: Callable = None,
    progress_callback: Callable = None
) -> List[Dict]:
    """
    Выполняет диаризацию аудио файла с помощью pyannote

    Args:
        audio_path: Путь к аудио файлу (WAV, 16kHz mono рекомендуется)
        min_speakers: Минимальное количество спикеров (None = auto)
        max_speakers: Максимальное количество спикеров (None = auto)
        hf_token: HuggingFace токен
        log_func: Функция логирования
        progress_callback: Callback для прогресса (value, message)

    Returns:
        List[Dict] с ключами: speaker, speaker_id, start, end
    """
    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    def progress(value, message):
        if progress_callback:
            progress_callback(value, message)

    try:
        progress(10, "Загрузка модели диаризации...")

        pipeline = load_diarization_pipeline(hf_token, log_func=log_func)
        if pipeline is None:
            return []

        progress(30, "Анализ аудио...")

        # Подготавливаем параметры
        diarize_params = {}
        if min_speakers is not None:
            diarize_params['min_speakers'] = min_speakers
        if max_speakers is not None:
            diarize_params['max_speakers'] = max_speakers

        log(f"Запуск диаризации (speakers: {min_speakers or 'auto'}-{max_speakers or 'auto'})...", "INFO")

        # Выполняем диаризацию
        if diarize_params:
            diarization = pipeline(audio_path, **diarize_params)
        else:
            diarization = pipeline(audio_path)

        progress(80, "Обработка результатов...")

        # Преобразуем результаты в наш формат
        segments = []
        speaker_map = {}  # Маппинг pyannote speaker -> наш speaker_id

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Создаем уникальный speaker_id
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map) + 1

            speaker_id = speaker_map[speaker]
            color_idx = (speaker_id - 1) % len(SPEAKER_COLORS)

            segments.append({
                'speaker': f"Спикер {speaker_id}",
                'speaker_id': speaker_id,
                'speaker_color': SPEAKER_COLORS[color_idx],
                'start': float(turn.start),
                'end': float(turn.end),
                'text': ''  # Текст будет добавлен позже
            })

        progress(100, "Диаризация завершена")

        unique_speakers = len(speaker_map)
        log(f"pyannote диаризация: {len(segments)} сегментов, {unique_speakers} спикеров", "SUCCESS")

        return segments

    except Exception as e:
        log(f"Ошибка pyannote диаризации: {e}", "ERROR")
        CrashSafeMemoryManager.safe_gpu_cleanup("after pyannote diarization error")
        return []


def align_transcription_with_diarization(
    transcription_segments: List[Dict],
    diarization_segments: List[Dict],
    log_func: Callable = None
) -> List[Dict]:
    """
    Совмещает результаты транскрипции Whisper с диаризацией pyannote

    Args:
        transcription_segments: Сегменты от Whisper (start, end, text)
        diarization_segments: Сегменты от pyannote (speaker, speaker_id, start, end)
        log_func: Функция логирования

    Returns:
        List[Dict]: Объединенные сегменты с текстом и спикерами
    """
    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)

    if not transcription_segments:
        return []

    if not diarization_segments:
        # Если диаризация пустая, возвращаем транскрипцию как есть
        return transcription_segments

    aligned = []

    for trans_seg in transcription_segments:
        trans_start = trans_seg['start']
        trans_end = trans_seg['end']
        trans_mid = (trans_start + trans_end) / 2

        # Находим спикера по максимальному перекрытию
        best_speaker = None
        best_overlap = 0

        for diar_seg in diarization_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']

            # Вычисляем перекрытие
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg

        # Если не нашли перекрытие, ищем ближайшего спикера по центру сегмента
        if best_speaker is None:
            min_distance = float('inf')
            for diar_seg in diarization_segments:
                diar_mid = (diar_seg['start'] + diar_seg['end']) / 2
                distance = abs(trans_mid - diar_mid)
                if distance < min_distance:
                    min_distance = distance
                    best_speaker = diar_seg

        # Создаем объединенный сегмент
        if best_speaker:
            aligned.append({
                'start': trans_start,
                'end': trans_end,
                'text': trans_seg.get('text', ''),
                'speaker': best_speaker['speaker'],
                'speaker_id': best_speaker['speaker_id'],
                'speaker_color': best_speaker.get('speaker_color', SPEAKER_COLORS[0])
            })
        else:
            # Fallback: назначаем первого спикера
            aligned.append({
                'start': trans_start,
                'end': trans_end,
                'text': trans_seg.get('text', ''),
                'speaker': 'Спикер 1',
                'speaker_id': 1,
                'speaker_color': SPEAKER_COLORS[0]
            })

    log(f"Alignment: {len(aligned)} сегментов с назначенными спикерами", "INFO")
    return aligned
