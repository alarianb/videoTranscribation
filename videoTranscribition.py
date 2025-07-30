"""
Профессиональная программа транскрибации с диаризацией
Оптимизированная версия с автоматическим скачиванием моделей
"""

import sys
import os
import gc
import re
import time
import json
import subprocess
import tempfile
import warnings
import hashlib
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from memory_utils import force_gc
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QTextEdit, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QMessageBox, QStyleFactory, QProgressBar, QComboBox,
    QCheckBox, QSpinBox, QTabWidget, QTextBrowser,
    QSplitter, QFrame, QStyle, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QFont, QPalette, QColor, QTextCharFormat, QTextCursor, QPixmap

# Отключаем предупреждения
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Константы
APP_NAME = "Audio/Video Transcription Pro"
APP_VERSION = "5.1"
AUTHOR = "Lebedev Nikolay"
DEFAULT_HF_TOKEN = "YOUR TOKEN HERE"

# Проверка зависимостей при импорте
FASTER_WHISPER_AVAILABLE = False
DOCX_AVAILABLE = False
TORCH_AVAILABLE = False
DEVICE = "cpu"

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if sys.platform == "win32":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
except ImportError:
    pass


class ModelDownloader(QThread):
    """Поток для скачивания и проверки моделей"""

    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)
    log_signal = Signal(str, str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.should_stop = False

    def run(self):
        try:
            self.log_signal.emit(f"Проверка модели {self.model_name}...", "INFO")
            self.progress_signal.emit(10, "Проверка наличия модели...")

            # Пытаемся создать модель - если нет, то faster-whisper скачает автоматически
            if self.should_stop:
                return

            self.progress_signal.emit(30, "Загрузка модели...")

            # Создаем временную модель для проверки и загрузки
            compute_type = "float16" if DEVICE == "cuda" else "int8"

            model = WhisperModel(
                self.model_name,
                device=DEVICE,
                compute_type=compute_type,
                cpu_threads=min(4, os.cpu_count()),
                num_workers=1,
                download_root=self.get_models_cache_dir()
            )

            self.progress_signal.emit(80, "Проверка работоспособности...")

            # Удаляем модель из памяти
            del model
            gc.collect()
            if DEVICE == "cuda" and TORCH_AVAILABLE:
                torch.cuda.empty_cache()

            self.progress_signal.emit(100, "Модель готова!")
            self.finished_signal.emit(True, "Модель успешно загружена")

        except Exception as e:
            self.log_signal.emit(f"Ошибка загрузки модели: {str(e)}", "ERROR")
            self.finished_signal.emit(False, str(e))

    def stop(self):
        self.should_stop = True
        self.quit()
        self.wait()

    @staticmethod
    def get_models_cache_dir() -> Path:
        """Получить директорию для кэша моделей"""
        if sys.platform == "win32":
            cache_dir = Path.home() / "AppData" / "Local" / "WhisperModels"
        else:
            cache_dir = Path.home() / ".cache" / "whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


class AboutDialog(QDialog):
    """Диалог О программе"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе")
        self.setFixedSize(400, 300)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel(APP_NAME)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #58a6ff;
                margin: 10px;
            }
        """)

        # Версия
        version = QLabel(f"Версия {APP_VERSION}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("font-size: 14px; color: #8b949e; margin: 5px;")

        # Автор
        author = QLabel(f"Разработчик: {AUTHOR}")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author.setStyleSheet("font-size: 12px; color: #d4d4d4; margin: 5px;")

        # Описание
        description = QLabel(
            "Профессиональная программа для транскрибации\n"
            "аудио и видео файлов с поддержкой\n"
            "диаризации спикеров на базе Whisper AI"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 11px; color: #8b949e; margin: 15px;")

        # Технологии
        tech = QLabel("Использует: OpenAI Whisper, PyAnnote, PyQt6")
        tech.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tech.setStyleSheet("font-size: 10px; color: #6b7280; margin: 10px;")

        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background: #667eea;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #5a67d8;
            }
        """)

        layout.addWidget(title)
        layout.addWidget(version)
        layout.addWidget(author)
        layout.addWidget(description)
        layout.addWidget(tech)
        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)


class LogWidget(QTextBrowser):
    """Красивый виджет для логов с цветовой подсветкой"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setOpenExternalLinks(False)
        self.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #7a7a7a;
            }
        """)

    def log(self, message, level="INFO"):
        """Добавить сообщение в лог с цветовой подсветкой"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "INFO": "#58a6ff",
            "SUCCESS": "#56d364",
            "WARNING": "#f0883e",
            "ERROR": "#f85149",
            "DEBUG": "#8b949e"
        }

        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }

        color = colors.get(level, "#d4d4d4")
        icon = icons.get(level, "•")

        html = f'<span style="color: #8b949e">[{timestamp}]</span> '
        html += f'<span style="color: {color}">{icon} <b>{level}</b>:</span> '
        html += f'<span style="color: #d4d4d4">{message}</span>'

        self.append(html)

        # Автоскролл к последнему сообщению
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)


class TranscriptionWorker(QThread):
    """Рабочий поток для транскрибации с улучшенной очисткой памяти"""

    progress_signal = Signal(int, str)
    log_signal = Signal(str, str)
    finished_signal = Signal(str)
    segment_signal = Signal(str)
    stats_signal = Signal(dict)

    def __init__(self, file_path, settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings
        self.temp_dir = None
        self._is_running = True
        self.start_time = None

    def run(self):
        """Основной процесс транскрибации"""
        try:
            self.start_time = time.time()

            # Создаем временную директорию
            self.temp_dir = tempfile.TemporaryDirectory()
            output_audio = os.path.join(self.temp_dir.name, "audio.wav")

            # Этап 1: Извлечение аудио
            self.progress_signal.emit(10, "📼 Извлечение аудио...")
            self.log_signal.emit("Начало извлечения аудио из файла", "INFO")
            self.extract_audio(output_audio)

            if not self._is_running:
                return

            # Этап 2: Загрузка модели
            self.progress_signal.emit(20, "🤖 Загрузка модели Whisper...")
            self.log_signal.emit(f"Загрузка модели: {self.settings['model_size']} на {DEVICE}", "INFO")

            model = self.load_model()

            # Этап 3: Транскрибация
            self.progress_signal.emit(30, "🎙️ Распознавание речи...")
            self.log_signal.emit("Начало транскрибации. Это может занять время...", "INFO")

            segments = self.transcribe_audio(output_audio, model)

            # Очистка модели из памяти
            del model
            gc.collect()
            if DEVICE == "cuda" and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                self.log_signal.emit("Память GPU очищена", "DEBUG")

            if not self._is_running:
                return

            # Этап 4: Диаризация
            formatted_text = ""
            if self.settings.get('use_diarization'):
                self.progress_signal.emit(70, "👥 Разделение по спикерам...")
                formatted_text = self.apply_diarization(output_audio, segments)
            else:
                formatted_text = self.format_simple_text(segments)

            # Статистика
            self.send_statistics(segments, formatted_text)

            self.progress_signal.emit(100, "✨ Готово!")
            self.log_signal.emit("Транскрибация успешно завершена!", "SUCCESS")
            self.finished_signal.emit(formatted_text)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.log_signal.emit(f"Критическая ошибка: {str(e)}", "ERROR")
            self.finished_signal.emit(f"Ошибка: {str(e)}")
        finally:
            self.cleanup()

    def stop(self):
        """Остановка процесса"""
        self._is_running = False
        self.log_signal.emit("Остановка процесса...", "WARNING")
        self.quit()
        self.wait()

    def extract_audio(self, output_path):
        """Извлечение аудио из видео"""
        ffmpeg_exe = self.find_ffmpeg()
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg не найден! Установите FFmpeg или поместите ffmpeg.exe в папку с программой")

        cmd = [
            ffmpeg_exe, "-y",
            "-i", self.file_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-af", "highpass=f=200,lowpass=f=3000",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка FFmpeg: {result.stderr}")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        self.log_signal.emit(f"Аудио извлечено: {size_mb:.1f} МБ", "SUCCESS")

    def find_ffmpeg(self):
        """Поиск FFmpeg"""
        # Сначала пробуем в папке с программой
        if getattr(sys, 'frozen', False):
            # Если запущено из exe
            app_dir = Path(sys.executable).parent
        else:
            # Если запущено из скрипта
            app_dir = Path(__file__).parent

        local_ffmpeg = app_dir / "ffmpeg.exe"
        if local_ffmpeg.exists():
            return str(local_ffmpeg)

        # Пробуем системный FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode == 0:
                return 'ffmpeg'
        except:
            pass

        return None

    def load_model(self):
        """Загрузка модели с оптимальными параметрами"""
        compute_type = "float16" if DEVICE == "cuda" else "int8"

        # Для больших моделей на слабых GPU
        if DEVICE == "cuda" and self.settings['model_size'] in ['large', 'large-v2', 'large-v3']:
            try:
                import torch
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram < 10:
                    compute_type = "int8"
                    self.log_signal.emit(f"GPU память: {vram:.1f}GB. Используем int8", "WARNING")
            except:
                pass

        model = WhisperModel(
            self.settings['model_size'],
            device=DEVICE,
            compute_type=compute_type,
            cpu_threads=min(4, os.cpu_count()),
            num_workers=1,
            download_root=ModelDownloader.get_models_cache_dir()
        )

        self.log_signal.emit(f"Модель загружена (compute_type: {compute_type})", "SUCCESS")
        return model

    def transcribe_audio(self, audio_path, model):
        """Транскрибация с отображением прогресса"""
        segments, info = model.transcribe(
            audio_path,
            language=self.settings['language'] if self.settings['language'] != 'auto' else None,
            task="transcribe",
            beam_size=5,
            best_of=5,
            patience=1,
            temperature=0.0,
            initial_prompt="Это транскрипция на русском языке." if self.settings['language'] == 'ru' else None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=float('inf'),
                min_silence_duration_ms=self.settings.get('min_silence', 1000),
                speech_pad_ms=400
            ),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        # Язык
        if hasattr(info, 'language'):
            self.log_signal.emit(f"Определен язык: {info.language} ({info.language_probability:.0%})", "INFO")

        # Обработка сегментов
        segments_list = []
        total_segments = 0

        for segment in segments:
            if not self._is_running:
                break

            text = self.clean_text(segment.text.strip())
            if text and len(text) > 2:
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text
                })

                # Отправляем сегмент для отображения
                self.segment_signal.emit(f"[{self.format_time(segment.start)}] {text}")

                total_segments += 1
                if total_segments % 10 == 0:
                    progress = min(30 + int(40 * segment.end / (info.duration or 100)), 70)
                    self.progress_signal.emit(progress, f"🎙️ Обработано сегментов: {total_segments}")

        self.log_signal.emit(f"Распознано сегментов: {len(segments_list)}", "SUCCESS")
        return segments_list

    def clean_text(self, text):
        """Очистка текста от галлюцинаций"""
        if not text:
            return text

        # Удаляем повторения
        text = re.sub(r'([а-яА-Яa-zA-Z])\1{3,}', r'\1\1', text)
        text = re.sub(r'(\b\w{1,3}\b)[\s-]*(?:\1[\s-]*){3,}', r'\1', text)

        # Удаляем мусорные символы
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\"\']+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def apply_diarization(self, audio_path, segments):
        """Применение диаризации"""
        try:
            # Простая диаризация без pyannote для уменьшения зависимостей
            self.log_signal.emit("Применяем простую диаризацию по паузам", "INFO")
            return self.simple_diarization(segments)

        except Exception as e:
            self.log_signal.emit(f"Ошибка диаризации: {e}", "ERROR")
            return self.simple_diarization(segments)

    def simple_diarization(self, segments):
        """Простая диаризация по паузам"""
        if not segments:
            return ""

        min_pause = self.settings.get('min_pause', 2.0)
        diarized_segments = []
        current_speaker = 1
        last_end = 0

        for seg in segments:
            if last_end > 0 and seg['start'] - last_end > min_pause:
                current_speaker = 2 if current_speaker == 1 else 1

            diarized_segments.append({
                'speaker': f"Спикер {current_speaker}",
                'text': seg['text'],
                'start': seg['start'],
                'end': seg['end']
            })

            last_end = seg['end']

        return self.format_diarized_text(diarized_segments)

    def format_simple_text(self, segments):
        """Простое форматирование без диаризации"""
        return " ".join(seg['text'] for seg in segments)

    def format_diarized_text(self, segments):
        """Форматирование с диаризацией"""
        formatted_parts = []
        current_speaker = None
        current_texts = []

        for seg in segments:
            if seg['speaker'] != current_speaker:
                if current_texts:
                    formatted_parts.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = seg['speaker']
                current_texts = [seg['text']]
            else:
                current_texts.append(seg['text'])

        if current_texts:
            formatted_parts.append(f"{current_speaker}: {' '.join(current_texts)}")

        return "\n\n".join(formatted_parts)

    def format_time(self, seconds):
        """Форматирование времени"""
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def send_statistics(self, segments, text):
        """Отправка статистики"""
        elapsed = time.time() - self.start_time
        stats = {
            'duration': elapsed,
            'segments': len(segments),
            'words': len(text.split()),
            'chars': len(text),
            'speed': len(segments) / elapsed if elapsed > 0 else 0
        }
        self.stats_signal.emit(stats)

    def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.temp_dir:
                self.temp_dir.cleanup()
                self.log_signal.emit("Временные файлы удалены", "DEBUG")
        except:
            pass

        gc.collect()
        if DEVICE == "cuda" and TORCH_AVAILABLE:
            try:
                import torch
                torch.cuda.empty_cache()
                self.log_signal.emit("Память GPU полностью очищена", "DEBUG")
            except:
                pass


class MainWindow(QMainWindow):
    """Главное окно с улучшенным дизайном"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)

        # Темная тема
        self.setStyleSheet(self.get_dark_theme())

        # Проверка зависимостей при запуске
        self.check_dependencies()

        self.init_ui()
        self.video_file_path = None
        self.transcription_thread = None
        self.transcribed_text = ""
        self.model_downloader = None

    def check_dependencies(self):
        """Проверка зависимостей при запуске"""
        missing = []

        if not FASTER_WHISPER_AVAILABLE:
            missing.append("faster-whisper")

        if missing:
            msg = f"Отсутствуют необходимые библиотеки:\n\n"
            for lib in missing:
                msg += f"• {lib}\n"
            msg += f"\nДля установки выполните:\npip install {' '.join(missing)}"

            QMessageBox.critical(self, "Ошибка зависимостей", msg)
            sys.exit(1)

    def init_ui(self):
        """Создание интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
        header_container = QWidget()
        header_layout = QVBoxLayout(header_container)
        header_layout.setSpacing(0)
        header_layout.setContentsMargins(0,0,0,0)

        title_label = QLabel("🎙️ Транскрибация аудио и видео")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
                padding-top: 10px;
                padding-bottom: 15px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 10px;
            }
        """)

        header_layout.addWidget(title_label)
        main_layout.addWidget(header_container)

        # Основной контент в табах
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background: #2d2d2d;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #3c3c3c;
                color: #d4d4d4;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #667eea;
                color: white;
            }
            QTabBar::tab:hover {
                background: #4c5cda;
            }
        """)

        # Вкладки
        transcription_tab = self.create_transcription_tab()
        tabs.addTab(transcription_tab, "📝 Транскрибация")

        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "⚙️ Настройки")

        logs_tab = self.create_logs_tab()
        tabs.addTab(logs_tab, "📊 Журнал")

        main_layout.addWidget(tabs)

        # Меню
        self.create_menu()

        # Статус бар
        self.create_status_bar()

    def create_menu(self):
        """Создание меню"""
        menubar = self.menuBar()

        # Меню Помощь
        help_menu = menubar.addMenu("Помощь")

        about_action = help_menu.addAction("О программе")
        about_action.triggered.connect(self.show_about)

    def show_about(self):
        """Показать диалог О программе"""
        dialog = AboutDialog(self)
        dialog.exec()

    def create_transcription_tab(self):
        """Создание вкладки транскрибации"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Выбор файла
        file_group = QGroupBox("📁 Выбор файла")
        file_layout = QHBoxLayout()

        self.select_file_btn = QPushButton("Выбрать файл")
        self.select_file_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.select_file_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
        """)

        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("color: #8b949e; font-style: italic;")

        file_layout.addWidget(self.select_file_btn)
        file_layout.addWidget(self.file_label, 1)
        file_group.setLayout(file_layout)

        # Контролы
        control_layout = QHBoxLayout()

        self.transcribe_btn = QPushButton("▶️ Начать транскрибацию")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5a67d8, stop:1 #6b46c1);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4c51bf, stop:1 #553c9a);
            }
            QPushButton:disabled {
                background: #4a5568;
            }
        """)

        self.stop_btn = QPushButton("⏹️ Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #e53e3e;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 25px;
            }
            QPushButton:hover {
                background: #c53030;
            }
            QPushButton:disabled {
                background: #4a5568;
            }
        """)

        self.download_model_btn = QPushButton("📥 Скачать модель")
        self.download_model_btn.setStyleSheet("""
            QPushButton {
                background: #38a169;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 25px;
            }
            QPushButton:hover {
                background: #2f855a;
            }
            QPushButton:disabled {
                background: #4a5568;
            }
        """)

        control_layout.addWidget(self.transcribe_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.download_model_btn)
        control_layout.addStretch()

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3c3c3c;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                background-color: #2d2d2d;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
            }
        """)
        self.progress_bar.hide()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8b949e; font-size: 12px;")
        self.status_label.hide()

        # Splitter для результатов
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Результат
        result_group = QGroupBox("📄 Результат")
        result_layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Здесь появится расшифровка...")

        # Кнопки сохранения
        save_layout = QHBoxLayout()
        self.save_txt_btn = self.create_save_button("💾 TXT", "#48bb78")
        self.save_docx_btn = self.create_save_button("📄 DOCX", "#4299e1")
        self.save_json_btn = self.create_save_button("📊 JSON", "#ed8936")

        save_layout.addStretch()
        save_layout.addWidget(self.save_txt_btn)
        save_layout.addWidget(self.save_docx_btn)
        save_layout.addWidget(self.save_json_btn)

        result_layout.addWidget(self.result_text)
        result_layout.addLayout(save_layout)
        result_group.setLayout(result_layout)

        # Живая транскрипция
        live_group = QGroupBox("🔴 Живая транскрипция")
        live_layout = QVBoxLayout()

        self.live_text = QTextEdit()
        self.live_text.setReadOnly(True)
        self.live_text.setMaximumHeight(150)
        self.live_text.setPlaceholderText("Сегменты будут появляться здесь по мере распознавания...")

        live_layout.addWidget(self.live_text)
        live_group.setLayout(live_layout)

        splitter.addWidget(result_group)
        splitter.addWidget(live_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Компоновка
        layout.addWidget(file_group)
        layout.addLayout(control_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(splitter)

        # Подключение сигналов
        self.select_file_btn.clicked.connect(self.select_file)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.stop_btn.clicked.connect(self.stop_transcription)
        self.download_model_btn.clicked.connect(self.download_model)
        self.save_txt_btn.clicked.connect(lambda: self.save_results('txt'))
        self.save_docx_btn.clicked.connect(lambda: self.save_results('docx'))
        self.save_json_btn.clicked.connect(lambda: self.save_results('json'))

        return widget

    def create_settings_tab(self):
        """Создание вкладки настроек"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Основные настройки
        basic_group = QGroupBox("🎯 Основные настройки")
        basic_layout = QVBoxLayout()

        # Язык
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Язык:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ru - Русский", "en - English", "auto - Автоопределение"])
        self.language_combo.setCurrentIndex(0)
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()

        # Модель
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        models = [
            "tiny - Очень быстро (39 MB)",
            "base - Быстро (74 MB)",
            "small - Баланс (244 MB)",
            "medium - Качественно (769 MB)",
            "large-v3 - Максимум (1.5 GB)"
        ]
        self.model_combo.addItems(models)
        self.model_combo.setCurrentIndex(2)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()

        basic_layout.addLayout(lang_layout)
        basic_layout.addLayout(model_layout)
        basic_group.setLayout(basic_layout)

        # Диаризация
        diarization_group = QGroupBox("👥 Диаризация (разделение по спикерам)")
        diarization_layout = QVBoxLayout()

        self.diarization_checkbox = QCheckBox("Включить простую диаризацию по паузам")
        self.diarization_checkbox.setChecked(True)

        # Параметры диаризации
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Мин. пауза между спикерами (сек):"))
        self.min_pause_spin = QSpinBox()
        self.min_pause_spin.setMinimum(1)
        self.min_pause_spin.setMaximum(10)
        self.min_pause_spin.setValue(2)
        params_layout.addWidget(self.min_pause_spin)

        params_layout.addWidget(QLabel("Мин. тишина (мс):"))
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setMinimum(500)
        self.min_silence_spin.setMaximum(3000)
        self.min_silence_spin.setSingleStep(100)
        self.min_silence_spin.setValue(1000)
        params_layout.addWidget(self.min_silence_spin)

        params_layout.addStretch()

        diarization_layout.addWidget(self.diarization_checkbox)
        diarization_layout.addLayout(params_layout)
        diarization_group.setLayout(diarization_layout)

        # Информация о системе
        info_group = QGroupBox("💻 Информация о системе")
        info_layout = QVBoxLayout()

        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMaximumHeight(150)
        self.update_system_info()

        info_layout.addWidget(self.system_info)
        info_group.setLayout(info_layout)

        # Компоновка
        layout.addWidget(basic_group)
        layout.addWidget(diarization_group)
        layout.addWidget(info_group)
        layout.addStretch()

        return widget

    def create_logs_tab(self):
        """Создание вкладки логов"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Логи
        self.log_widget = LogWidget()

        # Кнопки управления логами
        controls_layout = QHBoxLayout()

        clear_btn = QPushButton("🗑️ Очистить")
        clear_btn.clicked.connect(self.log_widget.clear)

        export_btn = QPushButton("💾 Экспорт")
        export_btn.clicked.connect(self.export_logs)

        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)
        controls_layout.addWidget(export_btn)

        # Статистика
        stats_group = QGroupBox("📊 Статистика последней транскрибации")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlaceholderText("Статистика появится после транскрибации...")

        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)

        # Компоновка
        layout.addWidget(self.log_widget)
        layout.addLayout(controls_layout)
        layout.addWidget(stats_group)

        return widget

    def create_status_bar(self):
        """Создание статус бара"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background: #1e1e1e;
                color: #8b949e;
                border-top: 1px solid #3c3c3c;
            }
        """)

        # Виджеты для статус бара
        self.memory_label = QLabel("💾 Память: --")
        self.gpu_label = QLabel("🎮 GPU: --")
        self.time_label = QLabel("⏱️ Время: --")

        status_bar.addPermanentWidget(self.memory_label)
        status_bar.addPermanentWidget(self.gpu_label)
        status_bar.addPermanentWidget(self.time_label)

        # Таймер для обновления статистики
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

    def create_save_button(self, text, color):
        """Создание красивой кнопки сохранения"""
        btn = QPushButton(text)
        btn.setEnabled(False)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                border-radius: 20px;
            }}
            QPushButton:hover {{
                background: {color}dd;
            }}
            QPushButton:pressed {{
                background: {color}bb;
            }}
            QPushButton:disabled {{
                background: #4a5568;
                color: #718096;
            }}
        """)
        return btn

    def get_dark_theme(self):
        """Темная тема"""
        return """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2d2d2d;
                color: #d4d4d4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 2px solid #3c3c3c;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #58a6ff;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #484848;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #484848;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #484848;
                padding: 6px;
                border-radius: 6px;
            }
            QComboBox:drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #d4d4d4;
                margin-right: 5px;
            }
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #484848;
                padding: 6px;
                border-radius: 6px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
            }
            QProgressBar {
                text-align: center;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #7a7a7a;
            }
            QSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #484848;
                padding: 4px;
                border-radius: 6px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #484848;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #667eea;
                border-color: #667eea;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border-bottom: 1px solid #3c3c3c;
            }
            QMenuBar::item {
                padding: 4px 8px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background-color: #667eea;
            }
            QMenu {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                color: #d4d4d4;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background-color: #667eea;
            }
        """

    def update_system_info(self):
        """Обновление информации о системе"""
        info = []

        # Python
        info.append(f"🐍 Python: {sys.version.split()[0]}")

        # GPU
        if TORCH_AVAILABLE:
            if DEVICE == "cuda":
                try:
                    import torch
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    info.append(f"🎮 GPU: {gpu_name} ({vram:.1f} GB)")
                except:
                    info.append("🎮 GPU: Обнаружен")
            else:
                info.append("💻 Режим: CPU")
        else:
            info.append("⚠️ PyTorch не установлен")

        # FFmpeg
        ffmpeg_found = self.check_ffmpeg()
        info.append(f"🎬 FFmpeg: {'✅ Найден' if ffmpeg_found else '❌ Не найден'}")

        # Кэш моделей
        cache_dir = ModelDownloader.get_models_cache_dir()
        if cache_dir.exists():
            models = list(cache_dir.glob("*"))
            info.append(f"📦 Кэш моделей: {len(models)} файлов")
            info.append(f"📁 Путь: {cache_dir}")

        self.system_info.setPlainText("\n".join(info))

    def check_ffmpeg(self):
        """Проверка FFmpeg"""
        try:
            # Проверяем локальный FFmpeg
            if getattr(sys, 'frozen', False):
                app_dir = Path(sys.executable).parent
            else:
                app_dir = Path(__file__).parent

            local_ffmpeg = app_dir / "ffmpeg.exe"
            if local_ffmpeg.exists():
                return True

            # Проверяем системный FFmpeg
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            return result.returncode == 0
        except:
            return False

    def update_status(self):
        """Обновление статус бара"""
        # Память
        try:
            import psutil
            memory = psutil.Process().memory_info().rss / (1024 ** 3)
            self.memory_label.setText(f"💾 Память: {memory:.1f} GB")
        except:
            pass

        # GPU
        if DEVICE == "cuda" and TORCH_AVAILABLE:
            try:
                import torch
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                self.gpu_label.setText(f"🎮 GPU: {allocated:.1f}/{reserved:.1f} GB")
            except:
                pass

        # Время
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(f"⏱️ {current_time}")

    def select_file(self):
        """Выбор файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео или аудио файл",
            "",
            "Медиа файлы (*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav *.m4a *.aac *.flac *.ogg);;Все файлы (*.*)"
        )

        if file_path:
            self.video_file_path = file_path
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 ** 2)

            self.file_label.setText(f"📄 {file_name} ({file_size:.1f} MB)")
            self.file_label.setStyleSheet("color: #58a6ff; font-style: normal;")

            self.log_widget.log(f"Выбран файл: {file_name}", "INFO")
            self.log_widget.log(f"Размер: {file_size:.1f} MB", "DEBUG")

            # Очистка
            self.result_text.clear()
            self.live_text.clear()
            self.transcribed_text = ""

            # Активация кнопки
            self.transcribe_btn.setEnabled(True)

            # Деактивация кнопок сохранения
            for btn in [self.save_txt_btn, self.save_docx_btn, self.save_json_btn]:
                btn.setEnabled(False)

    def download_model(self):
        """Скачивание модели"""
        model_name = self.model_combo.currentText().split(' - ')[0]

        reply = QMessageBox.question(
            self,
            "Скачивание модели",
            f"Скачать модель {model_name}?\n\nЭто может занять время в зависимости от размера модели и скорости интернета.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.download_model_btn.setEnabled(False)
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            self.status_label.show()

            self.model_downloader = ModelDownloader(model_name)
            self.model_downloader.progress_signal.connect(self.on_download_progress)
            self.model_downloader.finished_signal.connect(self.on_download_finished)
            self.model_downloader.log_signal.connect(self.log_widget.log)
            self.model_downloader.start()

    @Slot(int, str)
    def on_download_progress(self, value, message):
        """Прогресс скачивания модели"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    @Slot(bool, str)
    def on_download_finished(self, success, message):
        """Завершение скачивания модели"""
        self.progress_bar.hide()
        self.status_label.hide()
        self.download_model_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Успех", "Модель успешно загружена!")
            self.log_widget.log("Модель готова к использованию", "SUCCESS")
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{message}")

        self.model_downloader = None

    def start_transcription(self):
        """Запуск транскрибации"""
        if not self.video_file_path:
            return

        # UI
        self.transcribe_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.download_model_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.status_label.show()
        self.live_text.clear()
        self.result_text.clear()

        # Настройки
        lang = self.language_combo.currentText().split(' - ')[0]
        model = self.model_combo.currentText().split(' - ')[0]

        settings = {
            'language': lang,
            'model_size': model,
            'use_diarization': self.diarization_checkbox.isChecked(),
            'min_pause': self.min_pause_spin.value(),
            'min_silence': self.min_silence_spin.value()
        }

        self.log_widget.log("=" * 50, "INFO")
        self.log_widget.log("Начало транскрибации", "INFO")
        self.log_widget.log(f"Настройки: {json.dumps(settings, ensure_ascii=False)}", "DEBUG")

        # Запуск потока
        self.transcription_thread = TranscriptionWorker(self.video_file_path, settings)
        self.transcription_thread.progress_signal.connect(self.on_progress)
        self.transcription_thread.log_signal.connect(self.log_widget.log)
        self.transcription_thread.finished_signal.connect(self.on_finished)
        self.transcription_thread.segment_signal.connect(self.on_segment)
        self.transcription_thread.stats_signal.connect(self.on_stats)
        self.transcription_thread.start()

    def stop_transcription(self):
        """Остановка транскрибации"""
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.log_widget.log("Остановка транскрибации...", "WARNING")
            self.transcription_thread.stop()
            self.on_finished("Транскрибация остановлена пользователем")

    @Slot(int, str)
    def on_progress(self, value, message):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    @Slot(str)
    def on_segment(self, segment):
        """Отображение сегмента в реальном времени"""
        self.live_text.append(segment)

        # Автоскролл
        cursor = self.live_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.live_text.setTextCursor(cursor)

    @Slot(dict)
    def on_stats(self, stats):
        """Отображение статистики"""
        text = f"""
📊 Статистика транскрибации:
⏱️ Время обработки: {stats['duration']:.1f} сек
📝 Сегментов: {stats['segments']}
💬 Слов: {stats['words']}
📄 Символов: {stats['chars']}
⚡ Скорость: {stats['speed']:.1f} сегм/сек
"""
        self.stats_text.setPlainText(text.strip())

    @Slot(str)
    def on_finished(self, result):
        """Завершение транскрибации"""
        self.progress_bar.hide()
        self.status_label.hide()
        self.transcribe_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.download_model_btn.setEnabled(True)

        if result.startswith("Ошибка:"):
            QMessageBox.critical(self, "Ошибка", result)
            self.log_widget.log(result, "ERROR")
        else:
            self.transcribed_text = result
            self.result_text.setPlainText(result)

            # Активация кнопок сохранения
            for btn in [self.save_txt_btn, self.save_docx_btn, self.save_json_btn]:
                btn.setEnabled(True)

            # Скролл к началу
            cursor = self.result_text.textCursor()
            cursor.setPosition(0)
            self.result_text.setTextCursor(cursor)

            self.log_widget.log("Транскрибация завершена успешно!", "SUCCESS")
            self.log_widget.log("=" * 50, "INFO")

        self.transcription_thread = None

    def save_results(self, format_type):
        """Сохранение результатов"""
        if not self.transcribed_text:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = Path(self.video_file_path).stem

        if format_type == 'txt':
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить как текст",
                f"{base_name}_{timestamp}.txt",
                "Текстовые файлы (*.txt)"
            )
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Транскрипция: {os.path.basename(self.video_file_path)}\n")
                        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Программа: {APP_NAME} v{APP_VERSION}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(self.transcribed_text)

                    self.log_widget.log(f"Файл сохранен: {file_path}", "SUCCESS")
                    QMessageBox.information(self, "Успех", "Файл успешно сохранен!")
                except Exception as e:
                    self.log_widget.log(f"Ошибка сохранения: {e}", "ERROR")
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

        elif format_type == 'docx':
            if not DOCX_AVAILABLE:
                QMessageBox.warning(self, "Внимание", "python-docx не установлен!\nУстановите: pip install python-docx")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить как Word",
                f"{base_name}_{timestamp}.docx",
                "Документы Word (*.docx)"
            )
            if file_path:
                try:
                    doc = Document()
                    doc.add_heading('Транскрипция', 0)

                    # Метаданные
                    doc.add_paragraph(f'Файл: {os.path.basename(self.video_file_path)}')
                    doc.add_paragraph(f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    doc.add_paragraph(f'Программа: {APP_NAME} v{APP_VERSION}')
                    doc.add_paragraph(f'Модель: {self.model_combo.currentText()}')
                    doc.add_paragraph(f'Язык: {self.language_combo.currentText()}')

                    doc.add_heading('Текст', level=1)

                    # Форматирование для спикеров
                    if "Спикер" in self.transcribed_text:
                        for para in self.transcribed_text.split('\n\n'):
                            if para.strip():
                                p = doc.add_paragraph()
                                if para.startswith("Спикер"):
                                    parts = para.split(":", 1)
                                    if len(parts) == 2:
                                        p.add_run(parts[0] + ":").bold = True
                                        p.add_run(" " + parts[1])
                                else:
                                    p.add_run(para)
                    else:
                        doc.add_paragraph(self.transcribed_text)

                    doc.save(file_path)
                    self.log_widget.log(f"Документ сохранен: {file_path}", "SUCCESS")
                    QMessageBox.information(self, "Успех", "Документ успешно сохранен!")
                except Exception as e:
                    self.log_widget.log(f"Ошибка сохранения: {e}", "ERROR")
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

        elif format_type == 'json':
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить как JSON",
                f"{base_name}_{timestamp}.json",
                "JSON файлы (*.json)"
            )
            if file_path:
                try:
                    data = {
                        'metadata': {
                            'file': os.path.basename(self.video_file_path),
                            'date': datetime.now().isoformat(),
                            'program': f"{APP_NAME} v{APP_VERSION}",
                            'model': self.model_combo.currentText(),
                            'language': self.language_combo.currentText()
                        },
                        'text': self.transcribed_text,
                        'statistics': self.stats_text.toPlainText() if self.stats_text.toPlainText() else None
                    }

                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    self.log_widget.log(f"JSON сохранен: {file_path}", "SUCCESS")
                    QMessageBox.information(self, "Успех", "JSON успешно сохранен!")
                except Exception as e:
                    self.log_widget.log(f"Ошибка сохранения: {e}", "ERROR")
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

    def export_logs(self):
        """Экспорт логов"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт логов",
            f"logs_{timestamp}.txt",
            "Текстовые файлы (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Логи {APP_NAME} v{APP_VERSION}\n")
                    f.write(f"Экспорт: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(self.log_widget.toPlainText())
                QMessageBox.information(self, "Успех", "Логи экспортированы!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать: {e}")

    def closeEvent(self, event):
        """Закрытие приложения"""
        active_threads = []

        if self.transcription_thread and self.transcription_thread.isRunning():
            active_threads.append("транскрибация")

        if self.model_downloader and self.model_downloader.isRunning():
            active_threads.append("скачивание модели")

        if active_threads:
            reply = QMessageBox.question(
                self,
                'Подтверждение',
                f'Выполняется: {", ".join(active_threads)}.\nВы уверены, что хотите выйти?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.log_widget.log("Принудительное закрытие...", "WARNING")

                if self.transcription_thread:
                    self.transcription_thread.stop()
                if self.model_downloader:
                    self.model_downloader.stop()

                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Проверка критических зависимостей
    if not FASTER_WHISPER_AVAILABLE:
        QMessageBox.critical(
            None, "Критическая ошибка",
            "Не установлен faster-whisper!\n\n"
            "Для установки выполните:\n"
            "pip install faster-whisper\n\n"
            "Дополнительно рекомендуется:\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
            "pip install python-docx"
        )
        sys.exit(1)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
