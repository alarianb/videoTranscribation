"""
Утилиты форматирования текста
"""

import re
from .config import SPEAKER_COLORS_MAP


def format_time(seconds):
    """Форматирование времени в читаемый формат MM:SS"""
    try:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    except:
        return "00:00"


def format_time_bracket(seconds):
    """Форматирование времени в формате [MM:SS]"""
    try:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"[{mins:02d}:{secs:02d}]"
    except:
        return ""


def clean_text(text):
    """Безопасная очистка текста от мусора и повторений"""
    if not text:
        return text

    try:
        # Удаляем повторения
        text = re.sub(r'([а-яА-Яa-zA-Z])\1{3,}', r'\1\1', text)
        text = re.sub(r'(\b\w{1,3}\b)[\s-]*(?:\1[\s-]*){3,}', r'\1', text)

        # Удаляем мусорные символы
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\"\']+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    except Exception:
        return str(text)[:500]  # Безопасный fallback


def format_simple_text(segments, validate_func=None):
    """Безопасное простое форматирование сегментов в текст"""
    if not segments or len(segments) == 0:
        return "Нет данных для форматирования."

    texts = []
    for i, seg in enumerate(segments):
        try:
            if validate_func and not validate_func(seg, i):
                continue

            text = str(seg.get('text', '')).strip()
            if text and len(text) > 1:
                texts.append(text)

        except Exception:
            continue

    if not texts:
        return "Не удалось извлечь текст из сегментов."

    result = " ".join(texts)
    result = result.replace("  ", " ").strip()

    if len(result) > 100000:
        result = result[:100000] + "\n\n[Результат обрезан для стабильности]"

    return result if result else "Пустой результат форматирования."


def format_diarized_text(segments, show_timestamps=False):
    """Безопасное форматирование диаризованного текста с HTML и цветами"""
    if not segments or len(segments) == 0:
        return "Нет сегментов для форматирования."

    formatted_parts = []
    current_speaker = None
    current_speaker_id = None
    current_texts = []
    current_start_time = None

    for i, seg in enumerate(segments):
        try:
            if not seg or not isinstance(seg, dict):
                continue

            speaker = str(seg.get('speaker', 'Неизвестный')).strip()[:50]
            speaker_id = seg.get('speaker_id', 1)
            text = str(seg.get('text', '')).strip()[:1000]
            start_time = seg.get('start', 0)

            if not speaker or not text:
                continue

            if speaker != current_speaker:
                if current_texts and current_speaker:
                    color = SPEAKER_COLORS_MAP.get(current_speaker_id, '#667eea')
                    time_html = ""
                    if show_timestamps and current_start_time is not None:
                        time_html = f'<span style="color: #64748b; font-size: 11px;">{format_time_bracket(current_start_time)} </span>'
                    speaker_html = f'<span style="color: {color}; font-weight: bold;">{current_speaker}:</span>'
                    text_html = f'<span style="color: #e2e8f0;"> {" ".join(current_texts)}</span>'
                    formatted_parts.append(f'<p style="margin: 10px 0;">{time_html}{speaker_html}{text_html}</p>')

                current_speaker = speaker
                current_speaker_id = speaker_id
                current_texts = [text]
                current_start_time = start_time
            else:
                current_texts.append(text)

                if len(current_texts) > 100:
                    current_texts = current_texts[-100:]

        except Exception:
            continue

    # Добавляем последний блок
    if current_texts and current_speaker:
        color = SPEAKER_COLORS_MAP.get(current_speaker_id, '#667eea')
        time_html = ""
        if show_timestamps and current_start_time is not None:
            time_html = f'<span style="color: #64748b; font-size: 11px;">{format_time_bracket(current_start_time)} </span>'
        speaker_html = f'<span style="color: {color}; font-weight: bold;">{current_speaker}:</span>'
        text_html = f'<span style="color: #e2e8f0;"> {" ".join(current_texts)}</span>'
        formatted_parts.append(f'<p style="margin: 10px 0;">{time_html}{speaker_html}{text_html}</p>')

    if not formatted_parts:
        return "Не удалось сформатировать диаризованный текст."

    # HTML обертка
    result = f'''<div style="font-family: 'Segoe UI', sans-serif; line-height: 1.6;">
        {"".join(formatted_parts)}
    </div>'''

    return result if result.strip() else "Пустой результат диаризации."


def extract_plain_text(segments):
    """Извлекает plain text из сегментов для экспорта"""
    try:
        formatted_parts = []
        current_speaker = None
        current_texts = []

        for seg in segments:
            if not seg or not isinstance(seg, dict):
                continue

            speaker = str(seg.get('speaker', 'Неизвестный')).strip()
            text = str(seg.get('text', '')).strip()

            if not speaker or not text:
                continue

            if speaker != current_speaker:
                if current_texts and current_speaker:
                    formatted_parts.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = speaker
                current_texts = [text]
            else:
                current_texts.append(text)

        if current_texts and current_speaker:
            formatted_parts.append(f"{current_speaker}: {' '.join(current_texts)}")

        return "\n\n".join(formatted_parts)
    except:
        return ""
