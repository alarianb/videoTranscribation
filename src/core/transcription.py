"""
Логика транскрибации
"""

from ..utils import CrashSafeMemoryManager, clean_text, format_time, process_segment, post_process_text


def transcribe_audio(audio_path: str, model, settings: dict,
                     log_func=None, segment_callback=None, running_check=None):
    """
    Транскрибация аудио файла

    Args:
        audio_path: Путь к аудио файлу
        model: Модель Whisper
        settings: Настройки транскрибации
        log_func: Функция логирования
        segment_callback: Callback для сегментов в реальном времени
        running_check: Функция проверки, нужно ли продолжать

    Returns:
        list: Список сегментов с text, start, end
    """
    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    def is_running():
        if running_check:
            return running_check()
        return True

    try:
        segments, info = model.transcribe(
            audio_path,
            language=settings['language'] if settings['language'] != 'auto' else None,
            task="transcribe",
            beam_size=5,
            best_of=5,
            patience=1,
            temperature=0.0,
            initial_prompt="Это транскрипция на русском языке." if settings['language'] == 'ru' else None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=float('inf'),
                min_silence_duration_ms=settings.get('min_silence', 1000),
                speech_pad_ms=400
            ),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        # Язык
        if hasattr(info, 'language'):
            log(f"Определен язык: {info.language} ({info.language_probability:.0%})", "INFO")

        # Определяем язык для постобработки
        detected_language = settings.get('language', 'auto')
        if hasattr(info, 'language'):
            detected_language = info.language

        # Безопасная обработка сегментов
        segments_list = []
        total_segments = 0

        for segment in segments:
            if not is_running():
                break

            try:
                # Первичная очистка
                text = clean_text(segment.text.strip())

                # Постобработка: исправление ошибок ASR, повторений
                text = process_segment(text, language=detected_language)

                if text and len(text) > 2:
                    segments_list.append({
                        'start': float(segment.start),
                        'end': float(segment.end),
                        'text': text
                    })

                    # Отправляем сегмент для отображения
                    if segment_callback:
                        segment_callback(f"[{format_time(segment.start)}] {text}")

                    total_segments += 1
                    if total_segments % 10 == 0:
                        # Периодическая очистка
                        if total_segments % 50 == 0:
                            CrashSafeMemoryManager.safe_gpu_cleanup("during transcription")

            except Exception as seg_error:
                log(f"Пропускаем проблемный сегмент: {seg_error}", "DEBUG")
                continue

        log(f"Распознано сегментов: {len(segments_list)}", "SUCCESS")
        return segments_list

    except Exception as e:
        log(f"Ошибка транскрибации: {e}", "ERROR")
        raise e


def validate_segment(seg, index=0):
    """Безопасная валидация сегмента"""
    try:
        if not seg or not isinstance(seg, dict):
            return False

        required_keys = ['start', 'end', 'text']
        if not all(key in seg for key in required_keys):
            return False

        start_time = float(seg['start'])
        end_time = float(seg['end'])

        if start_time < 0 or end_time < start_time or end_time - start_time > 600:
            return False

        text = str(seg['text']).strip()
        if not text or len(text) < 1:
            return False

        return True

    except Exception:
        return False


def finalize_transcription(segments: list, language: str = 'ru', options: dict = None) -> list:
    """
    Финальная постобработка всех сегментов

    Применяет полную постобработку:
    - Восстановление регистра
    - Восстановление пунктуации
    - Преобразование числительных в цифры
    - Исправление ошибок ASR

    Args:
        segments: Список сегментов с текстом
        language: Язык текста ('ru', 'en', 'auto')
        options: Опции постобработки (см. post_process_text)

    Returns:
        list: Обработанные сегменты
    """
    if not segments:
        return segments

    processed_segments = []

    for seg in segments:
        try:
            if not validate_segment(seg):
                continue

            text = seg.get('text', '')

            # Применяем полную постобработку
            processed_text = post_process_text(text, language=language, options=options)

            if processed_text and len(processed_text) > 1:
                processed_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': processed_text,
                    'speaker': seg.get('speaker'),
                    'speaker_id': seg.get('speaker_id')
                })

        except Exception:
            # При ошибке сохраняем оригинал
            processed_segments.append(seg)

    return processed_segments


def get_full_text(segments: list, language: str = 'ru') -> str:
    """
    Получить полный текст из сегментов с постобработкой

    Args:
        segments: Список сегментов
        language: Язык текста

    Returns:
        str: Полный обработанный текст
    """
    if not segments:
        return ""

    # Собираем текст из всех сегментов
    texts = [seg.get('text', '') for seg in segments if seg.get('text')]
    full_text = ' '.join(texts)

    # Применяем полную постобработку
    processed = post_process_text(full_text, language=language)

    return processed
