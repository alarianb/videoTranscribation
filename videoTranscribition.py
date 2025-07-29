import sys
import os
import subprocess
import tempfile
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QLineEdit, QTextEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QMessageBox, QStyleFactory, QProgressBar, QComboBox,
    QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from docx import Document
import time
import json
import gc
import warnings
import re
import numpy as np
import torch
import logging

# Подавляем предупреждения
warnings.filterwarnings("ignore")

# Хак для Windows
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYANNOTE_CACHE"] = os.path.expanduser("~/.cache/torch/pyannote")
    sys.modules['triton'] = None
# Импортируем библиотеки
try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import whisperx

    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
from packaging import version
# PyPI «packaging» :contentReference[oaicite:4]{index=4}
from importlib.metadata import version as pkg_version

try:
    WX_VER = pkg_version("whisperx")
except Exception:
    WX_VER = pkg_version("whisperx")
USE_ALIGNMENT = (
    pkg_version("whisperx") >= "3.4.0"   # поддерживает safetensors
    and os.getenv("NO_ALIGNMENT", "0") != "1"
    and torch.cuda.is_available()        # ускоряет только наGPU
)
ALIGN_MODELS = [
   "anton-l/wav2vec2-large-xlsr-53-russian",
   "pszemraj/wav2vec2-large-xlsr-53-russian-ru",
]



############################################################
# Улучшенная функция для очистки текста от галлюцинаций
############################################################

def clean_hallucinations(text):
    """Агрессивная очистка от галлюцинаций для русского языка"""
    if not text or len(text) < 3:
        print(f"[DEBUG] Отброшено как галлюцинация: {text[:60]}…")
        return True


    # Список паттернов галлюцинаций
    hallucination_patterns = [
        r'[оОаАэЭуУыЫиИеЕёЁюЮяЯ]{10,}',  # Длинные повторы гласных
        r'([а-яА-Я])\1{5,}',  # Повторение одной буквы более 5 раз
        r'(ой|ай|эй|уй|ий|ей|ёй|юй|яй)[-\s]?\1{3,}',  # Повторяющиеся междометия
        r'(\b\w{1,3}\b)(\s+\1){4,}',  # Короткие слова повторяются много раз
        r'[\-]{5,}',  # Много дефисов подряд
        r'[\.]{5,}',  # Много точек подряд
        r'(\s|^)([хХ]{3,}|[пП]{3,}|[фФ]{3,}|[сС]{3,}|[шШ]{3,})',  # Шипящие/свистящие
    ]
    hallucination_patterns.extend([
        r'(?:\b[оО]й[,\s]*){4,}',  # «ой, ой, ой…»
        r'(?:\b[аА]й[,\s]*){4,}',  # «ай, ай, ай…»
    ])

    # Применяем все паттерны
    for pattern in hallucination_patterns:
        text = re.sub(pattern, '', text)

    # Удаляем строки, которые состоят только из повторений
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Проверяем, не состоит ли строка из повторений короткого паттерна
        if re.match(r'^(.{1,5})\1{3,}$', line):
            continue

        # Проверяем соотношение уникальных символов
        if len(line) > 10:
            unique_chars = len(set(line.replace(' ', '')))
            total_chars = len(line.replace(' ', ''))
            if total_chars > 0 and unique_chars / total_chars < 0.2:  # Менее 20% уникальных
                continue

        # Проверяем на бессмысленные последовательности
        if re.search(r'[а-яА-Я]{20,}', line.replace(' ', '')):
            continue

        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Финальная очистка
    text = re.sub(r'\s+', ' ', text)  # Множественные пробелы
    text = re.sub(r'\n{3,}', '\n\n', text)  # Множественные переносы

    return text.strip()


def is_likely_hallucination(text):
    """Проверка, является ли текст вероятной галлюцинацией"""
    _log("DEBUG", f"Отброшено как галлюцинация: «{text[:60]}…»")
    if not text or len(text) < 3:
        return True
    text_lower = text.lower()  # ← объявляем сразу
    tokens = re.findall(r'\b[а-яА-Я]{1,4}\b', text_lower)
    if tokens and max(tokens.count(t) for t in set(tokens)) / len(tokens) > 0.4:
        return True

    # ≥50% одной буквы
    if any(text_lower.count(ch) / len(text_lower) > 0.5 for ch in set(text_lower)):
        return True

    vowels = 'аеёиоуыэюя'
    consonants = 'бвгджзйклмнпрстфхцчшщ'
    return not (any(c in vowels for c in text_lower) and any(c in consonants for c in text_lower))

    # Слишком много повторений одной буквы
    for char in text_lower:
        if text_lower.count(char) / len(text_lower) > 0.5:
            return True

    # Проверка на повторяющиеся слоги
    for i in range(1, min(4, len(text) // 2)):
        pattern = text[:i]
        if text == pattern * (len(text) // len(pattern)):
            return True

    # Проверка на отсутствие согласных или гласных
    vowels = 'аеёиоуыэюя'
    consonants = 'бвгджзйклмнпрстфхцчшщ'

    has_vowel = any(c in vowels for c in text_lower)
    has_consonant = any(c in consonants for c in text_lower)

    if not has_vowel or not has_consonant:
        return True

    return False


############################################################
# Оптимизированная транскрибация для русского языка
############################################################
def _log(level, msg):
    (logger or (lambda l, m: print(f"[{l}] {m}")))(level, msg)
def transcribe_russian_optimized(file_path, model_size="large-v3", device="auto", hf_token=None,logger=None):
    """Оптимизированная транскрибация специально для русского языка"""



    # Для русского языка лучше использовать large модель если есть GPU
    if device == "cuda" and model_size in ["base", "small"]:
        print("Рекомендуется использовать модель 'large-v3' для русского языка на GPU")
    # после определения device
    _log("INFO", f"Устройство: {device.upper()} "
                     f"{torch.cuda.get_device_name(0) if device == 'cuda' else ''}")

    if device == "cpu":
        _log("WARNING",
                 "CPU‑режим: скорость ~‑6× медленней GPU(см. общественные бенчмарки) "
                 "– ожидайте более длительного ожидания результата")
    else:
        _log("INFO", "GPU‑режим: ускорение x5‑x40 по сравнению с CPU "
                         "по данным открытых тестов")  # :contentReference[oaicite:1]{index=1}
    # Используем WhisperX если доступен
    if WHISPERX_AVAILABLE and device == "cuda":
        return transcribe_with_whisperx_russian(file_path, model_size, device, hf_token)
    else:
        return transcribe_with_whisper_russian(file_path, model_size, device)


def transcribe_with_whisper_russian(file_path, model_size="base", device="cpu"):
    """Faster Whisper оптимизированный для русского"""

    compute_type = "float16" if device == "cuda" else "int8"

    # Для русского языка используем специальные параметры
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=os.path.expanduser("~/.cache/whisper")
    )

    # Параметры оптимизированные для русского языка
    segments, info = model.transcribe(
        file_path,
        language="ru",
        task="transcribe",
        beam_size=15,
        best_of=5,
        patience=3,
        length_penalty=1,
        temperature=0.0,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-0.7,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
        initial_prompt="Это разговор на русском языке.",
        word_timestamps=True,
        prepend_punctuations="\"'¿([{-",
        append_punctuations="\"'.。,，!！?？:：",
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=9999,
            min_silence_duration_ms=800,
            window_size_samples=1024,
            speech_pad_ms=400
        )
    )

    all_segments = []
    for segment in segments:
        text = segment.text.strip()

        # Фильтруем галлюцинации
        if is_likely_hallucination(text):
            continue

        # Чистим текст
        cleaned_text = clean_hallucinations(text)

        if cleaned_text and len(cleaned_text) > 2:
            # Дополнительная проверка качества сегмента
            if hasattr(segment, 'avg_logprob') and segment.avg_logprob < -1.2:
                continue

            all_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': cleaned_text,
                'confidence': getattr(segment, 'avg_logprob', 0)
            })

    del model
    gc.collect()
    return all_segments


def transcribe_with_whisperx_russian(file_path, model_size="large-v3", device="cuda", hf_token=None):
    """WhisperX оптимизированный для русского языка"""

    batch_size = 16 if device == "cuda" else 4
    compute_type = "float16" if device == "cuda" else "int8"

    # Загружаем аудио
    audio = whisperx.load_audio(file_path)

    # Загружаем модель с параметрами для русского
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language="ru",
        asr_options={
            "beam_size": 10,
            "best_of": 5,
            "patience": 2,
            "length_penalty": 1,
            "temperatures": [0.0],
            "compression_ratio_threshold": 2.0,
            "log_prob_threshold": -0.7,
            "no_speech_threshold": 0.5,
            "condition_on_previous_text": False,
            "initial_prompt": "Это разговор на русском языке.",
            "suppress_tokens": [-1],
            "word_timestamps": True
        },
        vad_options={
            "vad_onset": 0.300,
            "vad_offset": 0.2
        }
    )

    # Транскрибация
    result = model.transcribe(audio, batch_size=batch_size, language="ru")
    del model
    gc.collect()

    # Для русского языка используем альтернативную модель выравнивания
    try:
        # Пробуем wav2vec2 модель для русского
        align_model_names = [
            "anton-l/wav2vec2-large-xlsr-53-russian",
            "pszemraj/wav2vec2-large-xlsr-53-russian-ru",
        ]

        aligned = False
        for model_name in (ALIGN_MODELS if USE_ALIGNMENT else []):
            try:
                print(f"Пробуем модель выравнивания: {model_name}")
                model_a, metadata = whisperx.load_align_model(
                    language_code="ru",
                    device=device,
                    model_name=model_name
                )

                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )

                del model_a
                aligned = True
                print(f"Выравнивание успешно с моделью: {model_name}")
                break

            except Exception as e:
                print(f"Модель {model_name} не сработала: {e}")
                continue

        if not aligned:
            print("Выравнивание пропущено - используем базовые временные метки")

    except Exception as e:
        print(f"Выравнивание недоступно: {e}")

    # Диаризация
    if hf_token and "segments" in result:
        try:
            from pyannote.audio import Pipeline
            import torch

            print("Загрузка модели диаризации...")
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            if device == "cuda":
                diarize_model.to(torch.device("cuda"))

            # Оптимизируем параметры для русского языка
            diarize_model.instantiate({
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 15,
                    "threshold": 0.7,
                },
                "segmentation": {
                    "min_duration_off": 0.5,
                }
            })

            print("Применение диаризации...")
            diarization = diarize_model(file_path)

            # Назначаем спикеров
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            # Сопоставляем с транскрипцией
            for seg in result["segments"]:
                seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2

                best_speaker = None
                best_overlap = 0

                for speaker_seg in speaker_segments:
                    overlap_start = max(seg.get("start", 0), speaker_seg["start"])
                    overlap_end = min(seg.get("end", 0), speaker_seg["end"])
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker_seg["speaker"]

                if best_speaker:
                    seg["speaker"] = best_speaker

            print("Диаризация применена успешно")

        except Exception as e:
            print(f"Диаризация не удалась: {e}")

    # Обработка и фильтрация результатов
    final_segments = []
    for seg in result.get("segments", []):
        text = seg.get('text', '').strip()

        # Фильтруем галлюцинации
        if is_likely_hallucination(text):
            continue

        cleaned_text = clean_hallucinations(text)

        if cleaned_text and len(cleaned_text) > 2:
            # Проверяем confidence если доступен
            if 'confidence' in seg and seg['confidence'] < 0.5:
                continue

            final_segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': cleaned_text,
                'speaker': seg.get('speaker'),
                'confidence': seg.get('confidence', 1.0)
            })

    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

    return final_segments


############################################################
# Улучшенная диаризация
############################################################

def apply_smart_diarization(segments, min_pause=1.5, merge_threshold=2.0):
    """Умная диаризация с объединением сегментов"""
    if not segments:
        return []

    # Проверяем, есть ли уже информация о спикерах
    has_speakers = any(seg.get('speaker') for seg in segments)

    if has_speakers:
        # Используем существующую диаризацию
        diarized_segments = []
        speaker_mapping = {}
        speaker_count = 1

        for seg in segments:
            original_speaker = seg.get('speaker', 'SPEAKER_00')

            if original_speaker not in speaker_mapping:
                speaker_mapping[original_speaker] = f"Спикер {speaker_count}"
                speaker_count += 1

            diarized_segments.append({
                'speaker': speaker_mapping[original_speaker],
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'confidence': seg.get('confidence', 1.0)
            })
    else:
        # Применяем эвристическую диаризацию
        diarized_segments = []
        current_speaker = 1
        last_end_time = 0

        for i, segment in enumerate(segments):
            pause = segment['start'] - last_end_time if i > 0 else 0

            # Меняем спикера при длинной паузе
            if i > 0 and pause > min_pause:
                # Дополнительная эвристика: если текст начинается с вопроса,
                # вероятно другой спикер
                if segment['text'].strip().endswith('?'):
                    current_speaker = 2 if current_speaker == 1 else 1
                elif pause > min_pause * 1.5:
                    current_speaker = 2 if current_speaker == 1 else 1

            diarized_segments.append({
                'speaker': f"Спикер {current_speaker}",
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'confidence': segment.get('confidence', 1.0)
            })
            last_end_time = segment['end']

    # Объединяем близкие сегменты одного спикера
    merged_segments = []
    current_segment = None

    for seg in diarized_segments:
        if current_segment is None:
            current_segment = seg.copy()
            current_segment['texts'] = [current_segment['text']]
        elif (seg['speaker'] == current_segment['speaker'] and
              seg['start'] - current_segment['end'] < merge_threshold):
            # Объединяем
            current_segment['end'] = seg['end']
            current_segment['texts'].append(seg['text'])
            # Берем минимальный confidence
            current_segment['confidence'] = min(
                current_segment.get('confidence', 1.0),
                seg.get('confidence', 1.0)
            )
        else:
            # Сохраняем текущий сегмент
            current_segment['text'] = ' '.join(current_segment['texts'])
            del current_segment['texts']
            merged_segments.append(current_segment)

            # Начинаем новый
            current_segment = seg.copy()
            current_segment['texts'] = [current_segment['text']]

    # Добавляем последний сегмент
    if current_segment:
        current_segment['text'] = ' '.join(current_segment['texts'])
        del current_segment['texts']
        merged_segments.append(current_segment)

    return merged_segments


############################################################
# Рабочий поток
############################################################

class TranscriptionWorker(QThread):
    progress_signal = Signal(int)
    status_signal = Signal(str, str)
    finished_signal = Signal(str, list)

    def __init__(self, video_file, language="ru", engine="whisper", model_size="base",
                 device="auto", min_pause=1.5, merge_threshold=2.0,
                 use_diarization=True, hf_token=None, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.language = language
        self.engine = engine
        self.model_size = model_size
        self.device = device
        self.min_pause = min_pause
        self.merge_threshold = merge_threshold
        self.use_diarization = use_diarization
        self.hf_token = hf_token
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_audio_file = os.path.join(self.temp_dir.name, "extracted_audio.wav")


    def run(self):
        device_type = "GPU" if torch.cuda.is_available() else "CPU"        _log("INFO", f"Устройство: {device_type} "
                         f"{torch.cuda.get_device_name(0) if device_type == 'GPU' else ''}")
        if device_type == "CPU":
            _log("WARNING", "CPU‑режим: распознавание будет значительно дольше.")
        try:
            start_time = time.time()
            _log("INFO", f"Обработка файла: {os.path.basename(self.video_file)}")

            # Проверка FFmpeg
            if not self.check_ffmpeg():
                raise RuntimeError("FFmpeg не найден. Поместите ffmpeg.exe в папку с программой.")

            # Конвертация в аудио
            _log("INFO", "Извлечение и подготовка аудио...")
            self.convert_audio(self.video_file, self.output_audio_file)
            self.progress_signal.emit(20)

            # Транскрибация
            segments = []

            if self.language == "ru":
                # Для русского используем оптимизированную функцию
                _log("INFO", f"Запуск оптимизированной транскрибации для русского языка...")
                segments = transcribe_russian_optimized(
                    self.output_audio_file,
                    self.model_size,
                    self.device,
                    self.hf_token if self.use_diarization else None,
                    logger=_log
                )
            else:
                # Для других языков используем стандартную
                if self.engine == "whisperx" and WHISPERX_AVAILABLE:
                    _log("INFO", f"Запуск WhisperX...")
                    segments = transcribe_with_whisperx_russian(
                        self.output_audio_file,
                        self.model_size,
                        self.device,
                        self.hf_token if self.use_diarization else None
                    )
                    if USE_ALIGNMENT:
                        for model_name in ALIGN_MODELS:
                            try:
                                model_a, metadata = whisperx.load_align_model(
                                    language_code="ru",
                                    device=self.device,
                                    model_name=model_name
                                )
                                segments = whisperx.align(
                                    segments, model_a, metadata,
                                    whisperx.load_audio(self.output_audio_file),
                                    self.device, return_char_alignments=False
                                )
                                _log("INFO", f"Word‑alignment «{model_name}» выполнен")
                                break
                            except Exception as e:
                                _log("WARNING", f"Word‑alignment «{model_name}» не сработал: {e}")
                    else:
                        _log("INFO", "Word‑alignment отключен")
                else:
                    _log("INFO", f"Запуск Whisper...")
                    segments = transcribe_with_whisper_russian(
                        self.output_audio_file,
                        self.model_size,
                        self.device
                    )

            self.progress_signal.emit(70)

            # Применяем диаризацию
            if self.use_diarization:
                _log("INFO", "Обработка спикеров...")
                diarized_segments = apply_smart_diarization(
                    segments,
                    self.min_pause,
                    self.merge_threshold
                )
            else:
                # Без диаризации - просто форматируем
                diarized_segments = [{
                    'speaker': 'Текст',
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'],
                    'confidence': seg.get('confidence', 1.0)
                } for seg in segments]

            self.progress_signal.emit(90)

            # Финальная фильтрация низкокачественных сегментов
            filtered_segments = []
            for seg in diarized_segments:
                if seg.get('confidence', 1.0) > 0.3:  # Фильтруем только очень плохие
                    filtered_segments.append(seg)

            # Форматирование результата
            formatted_text = self.format_diarized_text(filtered_segments)

            self.progress_signal.emit(95)

            # Статистика
            end_time = time.time()
            duration = end_time - start_time
            unique_speakers = len(set(seg['speaker'] for seg in filtered_segments)) if filtered_segments else 0

            _log("SUCCESS", f"Завершено за {duration:.1f} сек")
            _log("INFO", f"Распознано сегментов: {len(filtered_segments)}")
            if self.use_diarization:
                _log("INFO", f"Найдено спикеров: {unique_speakers}")

            self.progress_signal.emit(100)
            self.finished_signal.emit(formatted_text, filtered_segments)

        except Exception as e:
            import traceback
            error_msg = f"Ошибка: {str(e)}"
            _log("ERROR", error_msg)
            self.finished_signal.emit(error_msg, [])
            print(traceback.format_exc())
        finally:
            try:
                self.temp_dir.cleanup()
            except:
                pass

    def check_ffmpeg(self):
        """Проверка наличия FFmpeg"""
        try:
            # Проверяем в текущей директории
            if os.path.exists('ffmpeg.exe'):
                return True
            # Проверяем в PATH
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def convert_audio(self, input_file, output_file):
        """Конвертация с оптимизацией для распознавания речи"""
        ffmpeg_cmd = self.get_ffmpeg_path()

        # Параметры оптимизированные для русской речи
        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i", input_file,
            "-vn",  # Убираем видео
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16 kHz - оптимально для Whisper
            "-ac", "1",  # Моно
            "-af", "highpass=f=80,lowpass=f=8000,anlmdn=s=7:p=0.002",  # Фильтрация + шумоподавление
            output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg ошибка: {result.stderr}")

    def get_ffmpeg_path(self):
        """Получение пути к FFmpeg"""
        if os.path.exists('ffmpeg.exe'):
            return os.path.abspath('ffmpeg.exe')
        return 'ffmpeg'

    def format_diarized_text(self, segments):
        """Форматирование текста с диаризацией"""
        lines = []
        for segment in segments:
            speaker = segment['speaker']
            text = segment['text']
            lines.append(f"{speaker}: {text}")
        return "\n\n".join(lines)

    def log(self, level, message):
        """Отправка сообщения в UI"""
        self.status_signal.emit(level, message)


############################################################
# Главное окно
############################################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.transcribed_text = ""
        self.diarized_segments = []
        self.setWindowTitle("Транскрибация для русского языка v3.0")
        self.resize(1100, 800)
        self.setMinimumSize(900, 650)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.video_file_path = None
        self.transcription_thread = None

        self.init_ui(central_widget)
        self.check_requirements()

    def init_ui(self, central_widget):
        main_layout = QVBoxLayout(central_widget)

        # Группа выбора файла
        self.file_group = QGroupBox("1. Выбор файла")
        layout_file = QHBoxLayout()
        self.select_video_btn = QPushButton("Выбрать видео/аудио")
        self.video_label = QLabel("Файл не выбран")
        self.video_label.setWordWrap(True)
        layout_file.addWidget(self.select_video_btn)
        layout_file.addWidget(self.video_label, 1)
        self.file_group.setLayout(layout_file)

        # Группа настроек
        self.settings_group = QGroupBox("2. Настройки транскрибации")
        settings_layout = QVBoxLayout()

        # Основные настройки
        main_settings = QHBoxLayout()

        # Язык (только русский и английский)
        lang_label = QLabel("Язык:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Русский", "English"])

        # Движок
        engine_label = QLabel("Движок:")
        self.engine_combo = QComboBox()
        if WHISPER_AVAILABLE:
            self.engine_combo.addItem("Whisper", "whisper")
        if WHISPERX_AVAILABLE:
            self.engine_combo.addItem("WhisperX", "whisperx")

        # Модель
        model_label = QLabel("Модель:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tiny (39M)",
            "base (74M)",
            "small (244M)",
            "medium (769M)",
            "large-v3 (1550M)"
        ])
        self.model_combo.setCurrentText("small (244M)")

        # Устройство
        device_label = QLabel("Устройство:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Авто", "CPU", "GPU"])

        main_settings.addWidget(lang_label)
        main_settings.addWidget(self.language_combo)
        main_settings.addSpacing(20)
        main_settings.addWidget(engine_label)
        main_settings.addWidget(self.engine_combo)
        main_settings.addSpacing(20)
        main_settings.addWidget(model_label)
        main_settings.addWidget(self.model_combo)
        main_settings.addSpacing(20)
        main_settings.addWidget(device_label)
        main_settings.addWidget(self.device_combo)
        main_settings.addStretch()

        # Настройки диаризации
        diarization_layout = QHBoxLayout()

        self.use_diarization = QCheckBox("Разделять по спикерам")
        self.use_diarization.setChecked(True)

        pause_label = QLabel("Мин. пауза (сек):")
        self.pause_spin = QSpinBox()
        self.pause_spin.setMinimum(1)
        self.pause_spin.setMaximum(5)
        self.pause_spin.setValue(2)

        merge_label = QLabel("Объединять если < (сек):")
        self.merge_spin = QSpinBox()
        self.merge_spin.setMinimum(1)
        self.merge_spin.setMaximum(5)
        self.merge_spin.setValue(2)

        diarization_layout.addWidget(self.use_diarization)
        diarization_layout.addSpacing(20)
        diarization_layout.addWidget(pause_label)
        diarization_layout.addWidget(self.pause_spin)
        diarization_layout.addSpacing(20)
        diarization_layout.addWidget(merge_label)
        diarization_layout.addWidget(self.merge_spin)
        diarization_layout.addStretch()

        # Рекомендация для русского языка
        self.recommendation_label = QLabel(
            "💡 Для русского языка рекомендуется модель 'large-v3' на GPU"
        )
        self.recommendation_label.setStyleSheet("color: #1976d2; padding: 5px;")

        settings_layout.addLayout(main_settings)
        settings_layout.addLayout(diarization_layout)
        settings_layout.addWidget(self.recommendation_label)
        self.settings_group.setLayout(settings_layout)

        # Кнопка запуска
        self.transcribe_btn = QPushButton("3. Начать транскрибацию")
        self.transcribe_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 12px; }")

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()

        # Группа результатов
        self.result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        # Кнопки действий
        buttons_layout = QHBoxLayout()
        self.report_btn = QPushButton("Создать отчет")
        self.report_btn.setEnabled(False)
        self.save_btn = QPushButton("Сохранить результат")
        self.save_btn.setEnabled(False)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.report_btn)
        buttons_layout.addWidget(self.save_btn)

        result_layout.addWidget(self.result_text)
        result_layout.addLayout(buttons_layout)
        self.result_group.setLayout(result_layout)

        # Группа журнала
        _log_group = QGroupBox("Журнал")
        log_layout = QVBoxLayout()
        _log_text = QTextEdit()
        _log_text.setReadOnly(True)
        _log_text.setMaximumHeight(120)
        log_layout.addWidget(_log_text)
        _log_group.setLayout(log_layout)

        # Добавляем все в основной layout
        main_layout.addWidget(self.file_group)
        main_layout.addWidget(self.settings_group)
        main_layout.addWidget(self.transcribe_btn)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.result_group, 1)
        main_layout.addWidget(_log_group)

        # Подключение сигналов
        self.select_video_btn.clicked.connect(self.select_video_file)
        self.transcribe_btn.clicked.connect(self.transcribe_video)
        self.report_btn.clicked.connect(self.create_report)
        self.save_btn.clicked.connect(self.save_results)
        self.language_combo.currentTextChanged.connect(self.on_language_changed)
        self.device_combo.currentTextChanged.connect(self.on_device_changed)

        # Применяем стиль
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        self.setStyleSheet(self._build_stylesheet())

    def check_requirements(self):
        """Проверка установленных компонентов"""
        if not WHISPER_AVAILABLE and not WHISPERX_AVAILABLE:
            _log_text.append(
                '<div style="color: red;"><b>Ошибка!</b> Не установлен ни один движок транскрибации!</div>')
            self.transcribe_btn.setEnabled(False)
            return

        if WHISPER_AVAILABLE:
            _log_text.append('<div style="color: green;">✓ Faster Whisper установлен</div>')
        if WHISPERX_AVAILABLE:
            _log_text.append('<div style="color: green;">✓ WhisperX установлен</div>')

        # Проверяем GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                _log_text.append(f'<div style="color: green;">✓ GPU: {gpu_name}</div>')
                # Рекомендуем large модель для GPU
                self.model_combo.setCurrentText("large-v3 (1550M)")
            else:
                _log_text.append('<div style="color: orange;">⚠ GPU не найден, используется CPU</div>')
        except ImportError:
            _log_text.append('<div style="color: orange;">⚠ PyTorch не установлен для GPU</div>')

    def on_language_changed(self, text):
        """При изменении языка обновляем рекомендации"""
        if text == "Русский":
            self.recommendation_label.setText(
                "💡 Для русского языка рекомендуется модель 'large-v3' на GPU"
            )
            self.recommendation_label.show()
        else:
            self.recommendation_label.hide()

    def on_device_changed(self, text):
        """При изменении устройства обновляем рекомендации по модели"""
        if text == "CPU":
            self.model_combo.setCurrentText("base (74M)")
            _log_text.append(
                '<div style="color: orange;">⚠ На CPU рекомендуется использовать модель base или small</div>')

    @Slot()
    def select_video_file(self):
        """Выбор файла для транскрибации"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "",
            "Все медиа (*.mp4 *.mov *.avi *.mkv *.webm *.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus);;Видео (*.mp4 *.mov *.avi *.mkv *.webm);;Аудио (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus)"
        )
        if file_path:
            self.video_file_path = file_path
            self.video_label.setText(os.path.basename(file_path))

            # Показываем информацию о файле
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            duration = self.get_media_duration(file_path)

            info = f'<div>Файл: {os.path.basename(file_path)}</div>'
            info += f'<div>Размер: {size_mb:.1f} МБ</div>'
            if duration:
                info += f'<div>Длительность: {duration}</div>'

            _log_text.append(info)
            self.result_text.clear()
            self.report_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

    def get_media_duration(self, file_path):
        """Получение длительности медиафайла"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                duration_sec = float(result.stdout.strip())
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)
                return f"{minutes}:{seconds:02d}"
        except:
            pass
        return None

    @Slot()
    def transcribe_video(self):
        """Запуск транскрибации"""
        if not self.video_file_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл для транскрибации.")
            return

        # Очищаем предыдущие результаты
        self.result_text.clear()
        _log_text.clear()
        self.report_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        _log_text.append("<b>🚀 Запуск транскрибации...</b>")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.transcribe_btn.setEnabled(False)

        # Маппинг значений
        lang_map = {"Русский": "ru", "English": "en"}
        model_map = {
            "tiny (39M)": "tiny",
            "base (74M)": "base",
            "small (244M)": "small",
            "medium (769M)": "medium",
            "large-v3 (1550M)": "large-v3"
        }
        device_map = {"Авто": "auto", "CPU": "cpu", "GPU": "cuda"}

        # Создаем рабочий поток
        self.transcription_thread = TranscriptionWorker(
            self.video_file_path,
            language=lang_map.get(self.language_combo.currentText(), "ru"),
            engine=self.engine_combo.currentData() or "whisper",
            model_size=model_map.get(self.model_combo.currentText(), "base"),
            device=device_map.get(self.device_combo.currentText(), "auto"),
            min_pause=self.pause_spin.value(),
            merge_threshold=self.merge_spin.value(),
            use_diarization=self.use_diarization.isChecked(),
            hf_token=os.getenv("HF_TOKEN")
        )

        # Подключаем сигналы
        self.transcription_thread.progress_signal.connect(self.on_progress)
        self.transcription_thread.status_signal.connect(self.on_status)
        self.transcription_thread.finished_signal.connect(self.on_transcription_finished)

        # Запускаем
        self.transcription_thread.start()

    @Slot(int)
    def on_progress(self, value):
        """Обновление прогресс бара"""
        self.progress_bar.setValue(value)

    @Slot(str, str)
    def on_status(self, level, message):
        """Вывод статусных сообщений"""
        colors = {
            "ERROR": "#ffebee",
            "WARNING": "#fff3e0",
            "SUCCESS": "#e8f5e9",
            "INFO": "#e3f2fd"
        }
        color = colors.get(level, "#f5f5f5")

        icons = {
            "ERROR": "❌",
            "WARNING": "⚠️",
            "SUCCESS": "✅",
            "INFO": "ℹ️"
        }
        icon = icons.get(level, "")

        _log_text.append(
            f'<div style="background: {color}; color: #212121; padding: 4px 8px; '
            f'border-radius: 4px; margin: 2px 0;">'
            f'{icon} <b>[{level}]</b> {message}</div>'
        )

    @Slot(str, list)
    def on_transcription_finished(self, result_text, segments):
        """Обработка завершения транскрибации"""
        self.progress_bar.hide()
        self.transcribe_btn.setEnabled(True)

        if result_text.startswith("Ошибка:"):
            _log_text.append(f'<div style="color: red;"><b>{result_text}</b></div>')
            QMessageBox.critical(self, "Ошибка", "Произошла ошибка транскрибации. Подробности в журнале.")
        else:
            self.transcribed_text = result_text
            self.diarized_segments = segments
            self.result_text.setHtml(self._format_html_text(result_text))
            _log_text.append(
                '<div style="background: #e8f5e9; color: #212121; padding: 6px;">'
                '<b>✅ Транскрибация завершена успешно!</b></div>'
            )
            self.report_btn.setEnabled(True)
            self.save_btn.setEnabled(True)

        self.transcription_thread = None

    def _format_html_text(self, text):
        """Форматирование текста для отображения"""
        html_lines = []
        colors = {
            "Спикер 1": "#1976d2",
            "Спикер 2": "#388e3c",
            "Спикер 3": "#d32f2f",
            "Спикер 4": "#7b1fa2",
            "Спикер 5": "#f57c00",
            "Текст": "#424242"
        }

        for line in text.split('\n\n'):
            if ':' in line:
                try:
                    speaker, content = line.split(':', 1)
                    color = colors.get(speaker.strip(), "#424242")
                    html_lines.append(
                        f'<p style="margin: 10px 0; line-height: 1.5;">'
                        f'<b style="color:{color};">{speaker}:</b> {content.strip()}'
                        f'</p>'
                    )
                except ValueError:
                    html_lines.append(f'<p style="margin: 5px 0;">{line}</p>')
            elif line.strip():
                html_lines.append(f'<p style="margin: 5px 0;">{line}</p>')

        return "".join(html_lines)

    def create_report(self):
        """Создание отчета о транскрибации"""
        if not self.transcribed_text:
            return

        report = [
            "ОТЧЕТ О ТРАНСКРИБАЦИИ",
            "=" * 50,
            f"\nФайл: {os.path.basename(self.video_file_path)}",
            f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Движок: {self.engine_combo.currentText()}",
            f"Модель: {self.model_combo.currentText()}",
            f"Язык: {self.language_combo.currentText()}",
            f"\n{'=' * 50}",
            "СТАТИСТИКА:\n"
        ]

        if self.diarized_segments:
            speaker_stats = {}
            total_words = 0
            total_duration = 0
            total_confidence = 0
            confidence_count = 0

            for seg in self.diarized_segments:
                speaker = seg['speaker']
                words = len(seg['text'].split())
                duration = seg['end'] - seg['start']
                confidence = seg.get('confidence', 1.0)

                total_words += words
                total_duration = max(total_duration, seg['end'])
                if 'confidence' in seg:
                    total_confidence += confidence
                    confidence_count += 1

                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        'words': 0,
                        'segments': 0,
                        'duration': 0,
                        'confidence_sum': 0,
                        'confidence_count': 0
                    }

                speaker_stats[speaker]['words'] += words
                speaker_stats[speaker]['segments'] += 1
                speaker_stats[speaker]['duration'] += duration
                if 'confidence' in seg:
                    speaker_stats[speaker]['confidence_sum'] += confidence
                    speaker_stats[speaker]['confidence_count'] += 1

            # Статистика по спикерам
            for speaker, stats in sorted(speaker_stats.items()):
                percent_words = (stats['words'] / total_words * 100) if total_words > 0 else 0
                percent_time = (stats['duration'] / total_duration * 100) if total_duration > 0 else 0

                report.append(f"{speaker}:")
                report.append(f"  Реплик: {stats['segments']}")
                report.append(f"  Слов: {stats['words']} ({percent_words:.1f}%)")
                report.append(f"  Время речи: {stats['duration']:.1f} сек ({percent_time:.1f}%)")

                if stats['confidence_count'] > 0:
                    avg_confidence = stats['confidence_sum'] / stats['confidence_count']
                    report.append(f"  Средняя уверенность: {avg_confidence:.2%}")

                report.append("")

            # Общая статистика
            report.append("Общая статистика:")
            report.append(f"  Всего слов: {total_words}")
            report.append(f"  Спикеров: {len(speaker_stats)}")
            report.append(f"  Продолжительность: {total_duration:.1f} сек ({total_duration / 60:.1f} мин)")

            if confidence_count > 0:
                avg_total_confidence = total_confidence / confidence_count
                report.append(f"  Средняя уверенность: {avg_total_confidence:.2%}")

        report.extend(["\n" + "=" * 50, "ТРАНСКРИПЦИЯ:", "=" * 50 + "\n", self.transcribed_text])

        # Сохранение отчета
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчет",
            f"report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report))
                QMessageBox.information(self, "Успех", "Отчет сохранен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить отчет: {e}")

    def save_results(self):
        """Сохранение результатов в различных форматах"""
        if not self.transcribed_text:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат",
            f"transcription_{time.strftime('%Y%m%d_%H%M%S')}",
            "Word (*.docx);;Text (*.txt);;JSON (*.json);;SRT (*.srt);;VTT (*.vtt)"
        )

        if file_path:
            try:
                if file_path.endswith('.docx'):
                    self._save_docx(file_path)
                elif file_path.endswith('.json'):
                    self._save_json(file_path)
                elif file_path.endswith('.srt'):
                    self._save_srt(file_path)
                elif file_path.endswith('.vtt'):
                    self._save_vtt(file_path)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.transcribed_text)

                QMessageBox.information(self, "Успех", "Результат сохранен!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

    def _save_docx(self, path):
        """Сохранение в формате Word"""
        doc = Document()
        doc.add_heading('Транскрипция', 0)

        # Метаданные
        doc.add_paragraph(
            f'Файл: {os.path.basename(self.video_file_path)}\n'
            f'Дата: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'Модель: {self.model_combo.currentText()}\n'
        )

        doc.add_heading('Текст', level=1)

        # Транскрипция
        for line in self.transcribed_text.split('\n\n'):
            if ':' in line:
                try:
                    speaker, content = line.split(':', 1)
                    p = doc.add_paragraph()
                    p.add_run(f"{speaker}: ").bold = True
                    p.add_run(content.strip())
                except ValueError:
                    doc.add_paragraph(line)
            elif line.strip():
                doc.add_paragraph(line)

        doc.save(path)

    def _save_json(self, path):
        """Сохранение в формате JSON"""
        data = {
            'metadata': {
                'file': os.path.basename(self.video_file_path),
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'engine': self.engine_combo.currentText(),
                'model': self.model_combo.currentText(),
                'language': self.language_combo.currentText()
            },
            'segments': [
                {
                    'speaker': seg['speaker'],
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'],
                    'confidence': seg.get('confidence', 1.0)
                }
                for seg in self.diarized_segments
            ],
            'full_text': self.transcribed_text
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_srt(self, path):
        """Сохранение в формате SRT (субтитры)"""
        with open(path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(self.diarized_segments):
                start = self._time_to_srt(seg.get('start', 0))
                end = self._time_to_srt(seg.get('end', 0))
                text = f"{seg['speaker']}: {seg['text']}" if self.use_diarization.isChecked() else seg['text']
                f.write(f"{i + 1}\n{start} --> {end}\n{text}\n\n")

    def _save_vtt(self, path):
        """Сохранение в формате WebVTT"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(self.diarized_segments):
                start = self._time_to_vtt(seg.get('start', 0))
                end = self._time_to_vtt(seg.get('end', 0))
                text = f"{seg['speaker']}: {seg['text']}" if self.use_diarization.isChecked() else seg['text']
                f.write(f"{start} --> {end}\n{text}\n\n")

    def _time_to_srt(self, seconds):
        """Конвертация времени в формат SRT"""
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        ms = int((s - int(s)) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

    def _time_to_vtt(self, seconds):
        """Конвертация времени в формат VTT"""
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        ms = int((s - int(s)) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

    def _build_stylesheet(self):
        """Стиль интерфейса"""
        return """
            QMainWindow, QWidget {
                background: #fafafa;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #212121;
            }
            QGroupBox {
                font-weight: 600;
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 10px;
                color: #1976d2;
            }
            QLabel {
                color: #424242;
                padding-top: 5px;
            }
            QPushButton {
                background: #1976d2;
                color: white;
                border: none;
                padding: 10px 22px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #1565c0;
            }
            QPushButton:pressed {
                background: #0d47a1;
            }
            QPushButton:disabled {
                background: #bdbdbd;
            }
            QComboBox, QSpinBox {
                padding: 6px 10px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: white;
                color: #212121;
                min-width: 100px;
            }
            QComboBox:focus, QSpinBox:focus {
                border-color: #1976d2;
            }
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: white;
                padding: 8px;
                color: #212121;
                font-size: 13px;
                line-height: 1.4;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #e0e0e0;
                height: 20px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background: #1976d2;
                border-radius: 4px;
            }
            QCheckBox {
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """

    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        if self.transcription_thread and self.transcription_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Закрыть программу',
                'Транскрибация еще выполняется. Вы уверены, что хотите выйти?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.transcription_thread.quit()
                self.transcription_thread.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Transcription RU")
    app.setOrganizationName("WhisperRU")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
