"""
Логика диаризации спикеров
"""

from ..utils import (
    CrashSafeMemoryManager, SPEAKER_COLORS,
    format_simple_text, format_diarized_text
)
from .transcription import validate_segment


def apply_diarization(segments: list, settings: dict, log_func=None, running_check=None):
    """
    Применение диаризации к сегментам

    Args:
        segments: Список сегментов с text, start, end
        settings: Настройки диаризации (min_pause, max_speakers, show_timestamps)
        log_func: Функция логирования
        running_check: Функция проверки, нужно ли продолжать

    Returns:
        str: Отформатированный текст с разделением по спикерам (HTML)
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
        log("Применение улучшенной диаризации...", "INFO")

        if not segments or len(segments) == 0:
            log("Нет сегментов для диаризации", "WARNING")
            return "Нет распознанной речи для обработки."

        # Предварительная очистка памяти
        CrashSafeMemoryManager.safe_gpu_cleanup("before diarization")

        min_pause = float(settings.get('min_pause', 2.0))
        max_speakers = int(settings.get('max_speakers', 5))

        log(f"Обрабатываем {len(segments)} сегментов (макс. {max_speakers} спикеров, пауза {min_pause}с)", "INFO")

        # Шаг 1: Извлекаем характеристики каждого сегмента
        segment_features = []
        for i, seg in enumerate(segments):
            if not validate_segment(seg, i):
                continue

            start_time = float(seg['start'])
            end_time = float(seg['end'])
            text = str(seg['text']).strip()[:500]

            if not text or len(text) < 2:
                continue

            duration = end_time - start_time
            words = len(text.split())
            chars = len(text)

            # Характеристики речи
            speech_rate = words / duration if duration > 0 else 0
            char_rate = chars / duration if duration > 0 else 0
            avg_word_len = chars / words if words > 0 else 0

            segment_features.append({
                'index': i,
                'start': start_time,
                'end': end_time,
                'text': text,
                'duration': duration,
                'speech_rate': speech_rate,
                'char_rate': char_rate,
                'avg_word_len': avg_word_len,
                'words': words
            })

        if not segment_features:
            return format_simple_text(segments, validate_segment)

        # Шаг 2: Умная диаризация с кластеризацией
        diarized_segments = _smart_diarization(
            segment_features, min_pause, max_speakers, log, is_running
        )

        # Очистка после диаризации
        CrashSafeMemoryManager.safe_gpu_cleanup("after diarization processing")

        if not diarized_segments:
            log("Диаризация не создала сегментов", "WARNING")
            return format_simple_text(segments, validate_segment)

        unique_speakers = len(set(seg['speaker'] for seg in diarized_segments))
        log(f"Диаризация завершена: {len(diarized_segments)} сегментов, {unique_speakers} спикеров", "SUCCESS")

        show_timestamps = settings.get('show_timestamps', False)
        return format_diarized_text(diarized_segments, show_timestamps)

    except Exception as e:
        log(f"Ошибка диаризации: {e}", "ERROR")
        CrashSafeMemoryManager.safe_gpu_cleanup("after diarization error")
        return format_simple_text(segments, validate_segment)


def _smart_diarization(segment_features, min_pause, max_speakers, log_func, is_running):
    """Умная диаризация с анализом характеристик речи"""
    def log(msg, level="INFO"):
        if log_func:
            log_func(msg, level)
        else:
            print(f"[{level}] {msg}")

    try:
        diarized = []
        speaker_profiles = {}
        current_speaker = 1
        last_end = 0.0

        for i, feat in enumerate(segment_features):
            try:
                if i % 25 == 0 and not is_running():
                    log("Диаризация прервана", "WARNING")
                    break

                start_time = feat['start']
                end_time = feat['end']
                text = feat['text']

                # Определяем, нужна ли смена спикера
                pause_duration = start_time - last_end if last_end > 0 else 0

                if pause_duration > min_pause:
                    # Есть значительная пауза - возможна смена спикера
                    best_speaker = _find_best_speaker(
                        feat, speaker_profiles, current_speaker, max_speakers
                    )

                    if best_speaker != current_speaker:
                        log(
                            f"Смена спикера: {current_speaker} -> {best_speaker} (пауза {pause_duration:.1f}с)",
                            "DEBUG"
                        )
                    current_speaker = best_speaker

                # Обновляем профиль текущего спикера
                _update_speaker_profile(speaker_profiles, current_speaker, feat)

                # Получаем цвет спикера
                color_idx = (current_speaker - 1) % len(SPEAKER_COLORS)

                diarized.append({
                    'speaker': f"Спикер {current_speaker}",
                    'speaker_id': current_speaker,
                    'speaker_color': SPEAKER_COLORS[color_idx],
                    'text': text,
                    'start': start_time,
                    'end': end_time
                })

                last_end = end_time

                # Периодическая очистка
                if i % 50 == 0 and i > 0:
                    CrashSafeMemoryManager.safe_gpu_cleanup("during smart diarization")

            except Exception as seg_error:
                log(f"Пропускаем сегмент {i}: {seg_error}", "DEBUG")
                continue

        return diarized

    except Exception as e:
        log(f"Ошибка умной диаризации: {e}", "ERROR")
        return []


def _find_best_speaker(feat, speaker_profiles, current_speaker, max_speakers):
    """Находит наиболее подходящего спикера для сегмента"""
    try:
        if not speaker_profiles:
            return 1

        # Характеристики текущего сегмента
        curr_rate = feat['speech_rate']
        curr_char_rate = feat['char_rate']
        curr_avg_word = feat['avg_word_len']

        best_speaker = current_speaker
        best_score = float('inf')

        # Сравниваем с профилями существующих спикеров
        for speaker_id, profile in speaker_profiles.items():
            if profile['count'] < 2:
                continue

            # Вычисляем расстояние до профиля спикера
            rate_diff = abs(curr_rate - profile['avg_speech_rate'])
            char_diff = abs(curr_char_rate - profile['avg_char_rate'])
            word_diff = abs(curr_avg_word - profile['avg_word_len'])

            # Нормализованный score (меньше = лучше)
            score = (rate_diff * 2.0) + (char_diff * 0.5) + (word_diff * 1.0)

            if score < best_score:
                best_score = score
                best_speaker = speaker_id

        # Если score слишком высокий и есть место для нового спикера
        threshold = 3.0
        num_speakers = len(speaker_profiles)

        if best_score > threshold and num_speakers < max_speakers:
            new_speaker = num_speakers + 1
            return new_speaker

        # Если текущий спикер имеет схожий профиль, оставляем его
        if current_speaker in speaker_profiles:
            current_profile = speaker_profiles[current_speaker]
            if current_profile['count'] >= 2:
                current_rate_diff = abs(curr_rate - current_profile['avg_speech_rate'])
                current_char_diff = abs(curr_char_rate - current_profile['avg_char_rate'])
                current_word_diff = abs(curr_avg_word - current_profile['avg_word_len'])
                current_score = (current_rate_diff * 2.0) + (current_char_diff * 0.5) + (current_word_diff * 1.0)

                if current_score < best_score * 1.3:
                    return current_speaker

        return best_speaker

    except Exception:
        return current_speaker


def _update_speaker_profile(profiles, speaker_id, feat):
    """Обновляет профиль спикера новыми данными"""
    try:
        if speaker_id not in profiles:
            profiles[speaker_id] = {
                'count': 0,
                'total_speech_rate': 0,
                'total_char_rate': 0,
                'total_word_len': 0,
                'avg_speech_rate': 0,
                'avg_char_rate': 0,
                'avg_word_len': 0
            }

        p = profiles[speaker_id]
        p['count'] += 1
        p['total_speech_rate'] += feat['speech_rate']
        p['total_char_rate'] += feat['char_rate']
        p['total_word_len'] += feat['avg_word_len']

        # Обновляем средние значения
        p['avg_speech_rate'] = p['total_speech_rate'] / p['count']
        p['avg_char_rate'] = p['total_char_rate'] / p['count']
        p['avg_word_len'] = p['total_word_len'] / p['count']

    except Exception:
        pass
