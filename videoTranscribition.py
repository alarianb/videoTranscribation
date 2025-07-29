import sys
import os
import subprocess
import tempfile
import time
import json
import gc
import warnings
import re
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QLineEdit, QTextEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QMessageBox, QStyleFactory, QProgressBar, QComboBox,
    QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from docx import Document

# --- Начало конфигурации и проверок зависимостей ---

# Подавляем некоторые предупреждения для чистоты вывода
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
warnings.filterwarnings("ignore", category=UserWarning, module='pydub')

# Хак для Windows для решения конфликтов библиотек
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Проверка и импорт основных библиотек для транскрибации
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("[ERROR] PyTorch не установлен. Пожалуйста, установите его: pip install torch")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

# Модели для выравнивания текста (для WhisperX)
ALIGN_MODELS_RU = [
   "anton-l/wav2vec2-large-xlsr-53-russian",
   "pszemraj/wav2vec2-large-xlsr-53-russian-ru",
]

# --- Конец конфигурации и проверок зависимостей ---


# --- Начало настройки логирования ---

# Создаем специальный обработчик для перенаправления логов в QTextEdit
class QtLogHandler(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        # ИСПРАВЛЕНО: Используем метод append() вместо appendHtml()
        self.widget.append(msg)

# Форматтер для цветного вывода логов в HTML
class HtmlFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: ('#a0a0a0', 'DBUG'), # Серый
        logging.INFO: ('#2196f3', 'INFO'),  # Синий
        logging.WARNING: ('#ff9800', 'WARN'),# Оранжевый
        logging.ERROR: ('#f44336', 'ERR'),  # Красный
        logging.CRITICAL: ('#b71c1c', 'CRIT'),# Темно-красный
    }

    def format(self, record):
        color, level_name = self.FORMATS.get(record.levelno, ('black', record.levelname))
        # Используем f-string для большей читаемости
        return f'<div style="color: {color};"><b>[{level_name}]</b> {record.getMessage()}</div>'

# Глобальный логгер
log = logging.getLogger(__name__)

# --- Конец настройки логирования ---


# --- Начало утилит для обработки текста ---

def is_likely_hallucination(text, lang="ru"):
    """
    Проверяет, является ли сегмент текста вероятной галлюцинацией.
    Эта функция объединяет несколько эвристик для отсеивания бессмысленного текста.
    """
    if not text or len(text.strip()) < 3:
        return True

    text_lower = text.lower().strip()

    # 1. Проверка на повторяющиеся короткие слова
    tokens = re.findall(r'\b\w{1,4}\b', text_lower)
    if len(tokens) > 5 and max(tokens.count(t) for t in set(tokens)) / len(tokens) > 0.4:
        log.debug(f"Отброшено (повтор коротких слов): «{text[:60]}…»")
        return True

    # 2. Проверка на отсутствие гласных или согласных (для русского)
    if lang == "ru":
        vowels = 'аеёиоуыэюя'
        consonants = 'бвгджзйклмнпрстфхцчшщ'
        has_vowels = any(c in vowels for c in text_lower)
        has_consonants = any(c in consonants for c in text_lower)
        if not (has_vowels and has_consonants):
            log.debug(f"Отброшено (нет гласных/согласных): «{text[:60]}…»")
            return True

    # 3. Проверка на повторение одного символа
    if len(text_lower) > 10:
        for char in set(text_lower):
            if text_lower.count(char) / len(text_lower) > 0.5:
                log.debug(f"Отброшено (повтор одного символа): «{text[:60]}…»")
                return True

    # 4. Проверка на повторяющиеся паттерны (например, "ахахахах")
    if len(text_lower) > 8:
        for i in range(1, 5):
            pattern = text_lower[:i]
            if pattern * (len(text_lower) // len(pattern)) == text_lower:
                log.debug(f"Отброшено (повторяющийся паттерн): «{text[:60]}…»")
                return True

    return False

# --- Конец утилит для обработки текста ---


# --- Начало модулей транскрибации ---

def transcribe_with_faster_whisper(file_path, model_size, device, lang):
    """Транскрибация с использованием faster-whisper."""
    log.info(f"Загрузка модели faster-whisper: {model_size}...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    log.info("Транскрибация аудио...")
    segments, info = model.transcribe(
        file_path,
        language=lang,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    results = []
    for segment in segments:
        if not is_likely_hallucination(segment.text, lang=lang):
            results.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'confidence': segment.avg_logprob
            })

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def transcribe_with_whisperx(file_path, model_size, device, lang, hf_token):
    """Транскрибация с использованием whisperX."""
    batch_size = 16 if device == "cuda" else 4
    compute_type = "float16" if device == "cuda" else "int8"

    log.info("Загрузка аудио для whisperX...")
    audio = whisperx.load_audio(file_path)

    log.info(f"Загрузка модели whisperX: {model_size}...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=lang)

    log.info("Транскрибация аудио...")
    result = model.transcribe(audio, batch_size=batch_size)
    del model
    gc.collect()

    # Выравнивание (Word-level alignment)
    if lang == "ru" and device == "cuda":
        log.info("Запуск выравнивания текста...")
        try:
            align_model, metadata = whisperx.load_align_model(language_code=lang, device=device, model_name=ALIGN_MODELS_RU[0])
            result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
            del align_model
            gc.collect()
            log.info("Выравнивание успешно завершено.")
        except Exception as e:
            log.warning(f"Не удалось выполнить выравнивание: {e}")
    else:
        log.info("Выравнивание пропущено (доступно только для ru на GPU).")

    # Диаризация (разделение по спикерам)
    if hf_token:
        log.info("Запуск диаризации...")
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            log.info("Диаризация успешно завершена.")
        except Exception as e:
            log.warning(f"Не удалось выполнить диаризацию: {e}. Проверьте токен Hugging Face.")
    else:
        log.info("Диаризация пропущена (требуется токен Hugging Face).")

    # Формирование результата
    final_segments = []
    for seg in result.get("segments", []):
        text = seg.get('text', '').strip()
        if not is_likely_hallucination(text, lang=lang):
            final_segments.append({
                'start': seg.get('start'),
                'end': seg.get('end'),
                'text': text,
                'speaker': seg.get('speaker', 'SPEAKER_00'),
                'confidence': seg.get('avg_logprob', 0)
            })

    if device == "cuda":
        torch.cuda.empty_cache()
    return final_segments

# --- Конец модулей транскрибации ---


# --- Начало рабочего потока (QThread) ---

class TranscriptionWorker(QThread):
    progress_signal = Signal(int, str)
    finished_signal = Signal(str, list)
    log_signal = Signal(int, str)

    def __init__(self, file_path, settings, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.settings = settings
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_audio_file = os.path.join(self.temp_dir.name, "extracted_audio.wav")

    def run(self):
        try:
            start_time = time.time()
            self.log_signal.emit(logging.INFO, f"Начало обработки файла: {os.path.basename(self.file_path)}")
            self.progress_signal.emit(5, "Проверка FFmpeg...")

            if not self.check_ffmpeg():
                raise RuntimeError("FFmpeg не найден. Убедитесь, что ffmpeg.exe находится в папке с программой или в системном PATH.")

            self.progress_signal.emit(10, "Извлечение аудио...")
            self.convert_audio(self.file_path, self.output_audio_file)
            self.progress_signal.emit(25, "Подготовка к транскрибации...")

            segments = []
            engine = self.settings['engine']
            
            if engine == "whisperx" and WHISPERX_AVAILABLE:
                self.log_signal.emit(logging.INFO, "Используется движок WhisperX.")
                segments = transcribe_with_whisperx(
                    self.output_audio_file, self.settings['model_size'], self.settings['device'],
                    self.settings['language'], self.settings.get('hf_token')
                )
            elif FASTER_WHISPER_AVAILABLE:
                self.log_signal.emit(logging.INFO, "Используется движок Faster Whisper.")
                segments = transcribe_with_faster_whisper(
                    self.output_audio_file, self.settings['model_size'], self.settings['device'],
                    self.settings['language']
                )
            else:
                raise RuntimeError("Нет доступных движков для транскрибации.")

            self.progress_signal.emit(80, "Обработка результатов...")
            
            # Если у сегментов нет спикеров после whisperx, применяем эвристику
            if not any('speaker' in seg for seg in segments):
                 segments = self.heuristic_diarization(segments)

            # Объединение сегментов
            merged_segments = self.merge_segments(segments)

            self.progress_signal.emit(95, "Форматирование текста...")
            formatted_text = self.format_diarized_text(merged_segments)

            end_time = time.time()
            duration = end_time - start_time
            self.log_signal.emit(logging.INFO, f"Завершено за {duration:.1f} сек.")
            
            self.progress_signal.emit(100, "Готово!")
            self.finished_signal.emit(formatted_text, merged_segments)

        except Exception as e:
            self.log_signal.emit(logging.CRITICAL, f"Критическая ошибка: {e}")
            import traceback
            self.log_signal.emit(logging.DEBUG, traceback.format_exc())
            self.finished_signal.emit(f"Ошибка: {e}", [])
        finally:
            self.temp_dir.cleanup()

    def heuristic_diarization(self, segments, min_pause=1.5):
        """Простая эвристическая диаризация по паузам."""
        if not segments: return []
        
        diarized = []
        current_speaker = "Спикер 1"
        last_end = 0
        for seg in segments:
            pause = seg['start'] - last_end if last_end > 0 else 0
            if pause > min_pause:
                current_speaker = "Спикер 2" if current_speaker == "Спикер 1" else "Спикер 1"
            
            new_seg = seg.copy()
            new_seg['speaker'] = current_speaker
            diarized.append(new_seg)
            last_end = seg['end']
        return diarized
        
    def merge_segments(self, segments, merge_threshold=2.0):
        """Объединение близких сегментов одного спикера."""
        if not segments: return []

        merged = []
        current_segment = segments[0].copy()

        for next_segment in segments[1:]:
            is_same_speaker = current_segment.get('speaker') == next_segment.get('speaker')
            is_close = next_segment['start'] - current_segment['end'] < merge_threshold
            
            if is_same_speaker and is_close:
                current_segment['text'] += ' ' + next_segment['text']
                current_segment['end'] = next_segment['end']
            else:
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        merged.append(current_segment)
        return merged

    def check_ffmpeg(self):
        """Проверка наличия FFmpeg."""
        try:
            ffmpeg_path = 'ffmpeg.exe' if sys.platform == "win32" and os.path.exists('ffmpeg.exe') else 'ffmpeg'
            subprocess.run([ffmpeg_path, '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def convert_audio(self, input_file, output_file):
        """Конвертация медиафайла в WAV 16kHz моно."""
        ffmpeg_path = 'ffmpeg.exe' if sys.platform == "win32" and os.path.exists('ffmpeg.exe') else 'ffmpeg'
        cmd = [
            ffmpeg_path, "-y", "-i", input_file, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_file
        ]
        self.log_signal.emit(logging.INFO, f"Команда FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg ошибка: {result.stderr}")

    def format_diarized_text(self, segments):
        """Форматирование текста с диаризацией."""
        return "\n\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)

# --- Конец рабочего потока (QThread) ---


# --- Начало UI (Главное окно) ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.transcribed_text = ""
        self.diarized_segments = []
        self.video_file_path = None
        self.transcription_thread = None
        self.setWindowTitle("Video Transcription App v3.1 (Refactored)")
        self.resize(1100, 800)
        self.setMinimumSize(900, 650)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.init_ui(central_widget)
        self.init_logging()
        self.check_requirements()

    def init_logging(self):
        """Инициализация системы логирования."""
        handler = QtLogHandler(self.log_text_edit)
        handler.setFormatter(HtmlFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    def init_ui(self, central_widget):
        main_layout = QVBoxLayout(central_widget)

        # --- Группа 1: Выбор файла ---
        file_group = QGroupBox("1. Выбор файла")
        layout_file = QHBoxLayout()
        self.select_file_btn = QPushButton("Выбрать видео/аудио")
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setWordWrap(True)
        layout_file.addWidget(self.select_file_btn)
        layout_file.addWidget(self.file_label, 1)
        file_group.setLayout(layout_file)

        # --- Группа 2: Настройки ---
        settings_group = QGroupBox("2. Настройки транскрибации")
        settings_layout = QVBoxLayout()
        
        main_settings = QHBoxLayout()
        # Язык
        main_settings.addWidget(QLabel("Язык:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ru", "en"])
        main_settings.addWidget(self.language_combo)
        main_settings.addSpacing(20)
        
        # Движок
        main_settings.addWidget(QLabel("Движок:"))
        self.engine_combo = QComboBox()
        if FASTER_WHISPER_AVAILABLE: self.engine_combo.addItem("Faster Whisper", "faster-whisper")
        if WHISPERX_AVAILABLE: self.engine_combo.addItem("WhisperX", "whisperx")
        main_settings.addWidget(self.engine_combo)
        main_settings.addSpacing(20)

        # Модель
        main_settings.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText("small")
        main_settings.addWidget(self.model_combo)
        main_settings.addStretch()
        
        settings_layout.addLayout(main_settings)
        settings_group.setLayout(settings_layout)

        # --- Кнопка запуска ---
        self.transcribe_btn = QPushButton("3. Начать транскрибацию")
        
        # --- Прогресс ---
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.status_label = QLabel("")
        self.status_label.hide()

        # --- Группа 3: Результат ---
        result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        
        buttons_layout = QHBoxLayout()
        self.save_btn = QPushButton("Сохранить результат")
        self.save_btn.setEnabled(False)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.save_btn)
        
        result_layout.addWidget(self.result_text)
        result_layout.addLayout(buttons_layout)
        result_group.setLayout(result_layout)

        # --- Группа 4: Журнал ---
        log_group = QGroupBox("Журнал")
        log_layout = QVBoxLayout()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMaximumHeight(150)
        log_layout.addWidget(self.log_text_edit)
        log_group.setLayout(log_layout)

        # --- Сборка UI ---
        main_layout.addWidget(file_group)
        main_layout.addWidget(settings_group)
        main_layout.addWidget(self.transcribe_btn)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(result_group, 1)
        main_layout.addWidget(log_group)

        # --- Подключение сигналов ---
        self.select_file_btn.clicked.connect(self.select_file)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.save_btn.clicked.connect(self.save_results)

    def check_requirements(self):
        """Проверка системных требований и вывод информации в лог."""
        log.info("Проверка системных требований...")
        if not FASTER_WHISPER_AVAILABLE and not WHISPERX_AVAILABLE:
            log.critical("Не найдены движки транскрибации (Faster Whisper или WhisperX).")
            self.transcribe_btn.setEnabled(False)
            QMessageBox.critical(self, "Ошибка", "Необходимые библиотеки не установлены!")
            return

        if FASTER_WHISPER_AVAILABLE: log.info("✓ Faster Whisper доступен.")
        if WHISPERX_AVAILABLE: log.info("✓ WhisperX доступен.")
        
        if TORCH_AVAILABLE:
            if DEVICE == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                log.info(f"✓ Найдено GPU: {gpu_name}. Рекомендуется модель 'large-v3'.")
                self.model_combo.setCurrentText("large-v3")
            else:
                log.warning("GPU не найдено, используется CPU. Транскрибация будет медленной.")
                self.model_combo.setCurrentText("base")
        else:
            log.error("PyTorch не найден. Работа невозможна.")
            self.transcribe_btn.setEnabled(False)

    @Slot()
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Медиафайлы (*.mp4 *.mkv *.mp3 *.wav)")
        if file_path:
            self.video_file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            log.info(f"Выбран файл: {file_path}")
            self.result_text.clear()
            self.save_btn.setEnabled(False)

    @Slot()
    def start_transcription(self):
        if not self.video_file_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл.")
            return

        self.log_text_edit.clear()
        self.result_text.clear()
        self.save_btn.setEnabled(False)
        self.transcribe_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.show()
        
        settings = {
            'language': self.language_combo.currentText(),
            'engine': self.engine_combo.currentData(),
            'model_size': self.model_combo.currentText(),
            'device': DEVICE,
            'hf_token': os.getenv("HF_TOKEN") # Для диаризации в WhisperX
        }

        self.transcription_thread = TranscriptionWorker(self.video_file_path, settings)
        self.transcription_thread.progress_signal.connect(self.on_progress)
        self.transcription_thread.finished_signal.connect(self.on_finished)
        self.transcription_thread.log_signal.connect(log.log) # Перенаправляем лог из потока
        self.transcription_thread.start()

    @Slot(int, str)
    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    @Slot(str, list)
    def on_finished(self, result_text, segments):
        self.progress_bar.hide()
        self.status_label.hide()
        self.transcribe_btn.setEnabled(True)

        if result_text.startswith("Ошибка:"):
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка транскрибации.\n{result_text}")
        else:
            self.transcribed_text = result_text
            self.diarized_segments = segments
            self.result_text.setText(result_text)
            self.save_btn.setEnabled(True)
            log.info("Транскрибация успешно завершена!")

        self.transcription_thread = None

    def save_results(self):
        """Сохранение результатов в различных форматах."""
        if not self.transcribed_text: return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Сохранить результат", f"transcription_{time.strftime('%Y%m%d')}",
            "Word (*.docx);;Text (*.txt);;JSON (*.json)"
        )

        if not file_path: return

        try:
            if selected_filter.startswith('Word'):
                self._save_docx(file_path)
            elif selected_filter.startswith('JSON'):
                self._save_json(file_path)
            else: # Text
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.transcribed_text)
            
            QMessageBox.information(self, "Успех", "Результат сохранен!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")
            log.error(f"Ошибка сохранения файла: {e}")

    def _save_docx(self, path):
        """Сохранение в формате Word."""
        doc = Document()
        doc.add_heading('Транскрипция', 0)
        doc.add_paragraph(f'Файл: {os.path.basename(self.video_file_path)}\nДата: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        doc.add_heading('Текст', level=1)
        doc.add_paragraph(self.transcribed_text)
        doc.save(path)

    def _save_json(self, path):
        """Сохранение в формате JSON."""
        data = {
            'metadata': {
                'file': os.path.basename(self.video_file_path),
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'segments': self.diarized_segments,
            'full_text': self.transcribed_text
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def closeEvent(self, event):
        """Обработка закрытия приложения."""
        if self.transcription_thread and self.transcription_thread.isRunning():
            reply = QMessageBox.question(self, 'Закрыть программу',
                'Транскрибация еще выполняется. Вы уверены, что хотите выйти?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.transcription_thread.quit()
                self.transcription_thread.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# --- Конец UI ---


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
