"""
Профессиональная программа транскрибации с диаризацией
Автор: Lebedev Nikolay
Версия: 6.0-PROFESSIONAL с улучшенной диаризацией и современным UI
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
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QTextEdit, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QMessageBox, QStyleFactory, QProgressBar, QComboBox,
    QCheckBox, QSpinBox, QTabWidget, QTextBrowser,
    QSplitter, QFrame, QStyle, QDialog, QListWidget,
    QListWidgetItem, QAbstractItemView, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve, QMutex, QMutexLocker, QMimeData, QUrl
from PySide6.QtGui import QIcon, QFont, QPalette, QColor, QTextCharFormat, QTextCursor, QPixmap, QDragEnterEvent, QDropEvent

# Отключаем предупреждения
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Константы
APP_NAME = "Audio/Video Transcription Pro"
APP_VERSION = "6.0-PROFESSIONAL"
AUTHOR = "Lebedev Nikolay"
DEFAULT_HF_TOKEN = "" # СЮДА HF ТОКЕН

# Глобальный мьютекс для безопасности памяти
MEMORY_MUTEX = QMutex()

# Проверка зависимостей
FASTER_WHISPER_AVAILABLE = False
DOCX_AVAILABLE = False
TORCH_AVAILABLE = False
DEVICE = "cpu"

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("ОШИБКА: faster_whisper не установлен!")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Безопасная инициализация GPU
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if sys.platform == "win32":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(f"PyTorch: {torch.__version__}, Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    print("PyTorch не установлен - CPU режим")


class FileQueueItem:
    """Элемент очереди файлов"""
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.size = os.path.getsize(path) / (1024 ** 2)  # MB
        self.status = "В очереди"
        self.result = ""


class CrashSafeMemoryManager:
    """Безопасный менеджер памяти для предотвращения крашей"""

    _cleanup_in_progress = False
    _cleanup_lock = threading.Lock()

    @staticmethod
    def safe_gpu_cleanup(stage="unknown"):
        """Ультра-безопасная очистка GPU памяти"""
        # Предотвращаем множественные одновременные вызовы
        with CrashSafeMemoryManager._cleanup_lock:
            if CrashSafeMemoryManager._cleanup_in_progress:
                return

            CrashSafeMemoryManager._cleanup_in_progress = True

        try:
            # Мягкая сборка мусора Python
            try:
                gc.collect()
            except:
                pass

            # Очень осторожная работа с CUDA
            if DEVICE == "cuda" and TORCH_AVAILABLE:
                try:
                    import torch

                    # Множественные проверки безопасности
                    if not torch.cuda.is_available():
                        return

                    if torch.cuda.device_count() == 0:
                        return

                    # Проверяем, что есть инициализированный контекст
                    try:
                        current_device = torch.cuda.current_device()
                    except:
                        return

                    # Только мягкая очистка кэша, без синхронизации
                    try:
                        allocated_before = torch.cuda.memory_allocated(current_device) / (1024**2)
                        torch.cuda.empty_cache()
                        allocated_after = torch.cuda.memory_allocated(current_device) / (1024**2)

                        freed = allocated_before - allocated_after
                        if freed > 1:
                            print(f"GPU cleanup {stage}: freed {freed:.1f}MB")

                    except Exception as cache_error:
                        print(f"Cache cleanup warning: {cache_error}")

                except Exception as cuda_error:
                    print(f"CUDA cleanup warning: {cuda_error}")

        except Exception as e:
            print(f"Memory cleanup error: {e}")
        finally:
            CrashSafeMemoryManager._cleanup_in_progress = False

    @staticmethod
    def reset_for_new_transcription():
        """Полный сброс состояния для новой транскрибации"""
        try:
            # Принудительная сборка мусора
            for _ in range(3):
                gc.collect()

            # Очистка GPU памяти
            CrashSafeMemoryManager.safe_gpu_cleanup("reset for new transcription")

            # Небольшая пауза для стабилизации
            time.sleep(0.1)

        except Exception as e:
            print(f"Reset error: {e}")


class AboutDialog(QDialog):
    """Диалог О программе"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе")
        self.setFixedSize(450, 350)
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
            "аудио и видео файлов с диаризацией\n"
            "спикеров на базе Whisper AI\n\n"
            "✨ Поддержка множественных файлов\n"
            "✨ Drag & Drop интерфейс\n"
            "✨ Пакетная обработка"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 11px; color: #8b949e; margin: 15px;")

        # Технологии
        tech = QLabel("Использует: OpenAI Whisper, PyAnnote, PySide6, PyTorch")
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
    """Виджет для логов с цветовой подсветкой"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setOpenExternalLinks(False)
        self.setStyleSheet("""
            QTextBrowser {
                background-color: #0a0a14;
                color: #e2e8f0;
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 10px;
                padding: 12px;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 11px;
                line-height: 1.5;
            }
            QScrollBar:vertical {
                background: #0f0f1a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(y1:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(y1:0, y2:1,
                    stop:0 #7c8ff0, stop:1 #8b5fcf);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def log(self, message, level="INFO"):
        """Добавить сообщение в лог с цветовой подсветкой"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "INFO": "#60a5fa",
            "SUCCESS": "#34d399",
            "WARNING": "#fbbf24",
            "ERROR": "#f87171",
            "DEBUG": "#94a3b8"
        }

        icons = {
            "INFO": "●",
            "SUCCESS": "✓",
            "WARNING": "⚡",
            "ERROR": "✕",
            "DEBUG": "○"
        }

        color = colors.get(level, "#e2e8f0")
        icon = icons.get(level, "•")

        html = f'<span style="color: #64748b">[{timestamp}]</span> '
        html += f'<span style="color: {color}"><b>{icon} {level}</b></span> '
        html += f'<span style="color: #cbd5e1">{message}</span>'

        self.append(html)

        # Автоскролл к последнему сообщению
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)


class DropZoneWidget(QWidget):
    """Виджет зоны для перетаскивания файлов - вся область принимает файлы"""

    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self._is_dragging = False

        # Основной layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Подсказка для drag&drop
        self.hint_label = QLabel("Перетащите файлы сюда или используйте кнопку добавления")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-style: italic;
                padding: 8px;
                background: transparent;
            }
        """)

        # Список файлов
        self.file_list = QListWidget()
        self.file_list.setAcceptDrops(False)  # Отключаем DnD у списка, т.к. обрабатываем на уровне контейнера
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #0a0a14;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }
            QListWidget::item {
                background-color: #1a1a2e;
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 8px;
                padding: 10px 14px;
                margin: 3px;
                color: #e2e8f0;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-color: #667eea;
                color: white;
            }
            QListWidget::item:hover:!selected {
                background-color: #2d2d4a;
                border-color: rgba(102, 126, 234, 0.5);
            }
            QScrollBar:vertical {
                background: #0f0f1a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(y1:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 4px;
            }
        """)

        layout.addWidget(self.hint_label)
        layout.addWidget(self.file_list)

        self._update_style(False)

    def _update_style(self, is_dragging):
        """Обновление стиля при перетаскивании"""
        if is_dragging:
            self.setStyleSheet("""
                DropZoneWidget {
                    background-color: #0a0a14;
                    border: 2px solid #667eea;
                    border-radius: 12px;
                }
            """)
        else:
            self.setStyleSheet("""
                DropZoneWidget {
                    background-color: #0a0a14;
                    border: 2px dashed rgba(102, 126, 234, 0.4);
                    border-radius: 12px;
                }
            """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Обработка входа перетаскивания"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._is_dragging = True
            self._update_style(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Обработка движения при перетаскивании"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Обработка выхода перетаскивания"""
        self._is_dragging = False
        self._update_style(False)

    def dropEvent(self, event: QDropEvent):
        """Обработка сброса файлов"""
        self._is_dragging = False
        self._update_style(False)

        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    supported_formats = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
                    if ext in supported_formats:
                        files.append(file_path)

            if files:
                self.files_dropped.emit(files)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()


class FileListWidget(QListWidget):
    """Виджет списка файлов с поддержкой Drag&Drop"""

    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setStyleSheet("""
            QListWidget {
                background-color: #0a0a14;
                border: 2px dashed rgba(102, 126, 234, 0.4);
                border-radius: 12px;
                padding: 12px;
                font-size: 12px;
            }
            QListWidget::item {
                background-color: #1a1a2e;
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 8px;
                padding: 10px 14px;
                margin: 3px;
                color: #e2e8f0;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-color: #667eea;
                color: white;
            }
            QListWidget::item:hover:!selected {
                background-color: #2d2d4a;
                border-color: rgba(102, 126, 234, 0.5);
            }
            QScrollBar:vertical {
                background: #0f0f1a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(y1:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 4px;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Обработка входа перетаскивания"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace(
                "border: 2px dashed rgba(102, 126, 234, 0.4)",
                "border: 2px solid #667eea"
            ))
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Обработка выхода перетаскивания"""
        self.setStyleSheet(self.styleSheet().replace(
            "border: 2px solid #667eea",
            "border: 2px dashed rgba(102, 126, 234, 0.4)"
        ))

    def dropEvent(self, event: QDropEvent):
        """Обработка сброса файлов"""
        self.setStyleSheet(self.styleSheet().replace(
            "border: 2px solid #667eea",
            "border: 2px dashed rgba(102, 126, 234, 0.4)"
        ))

        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    # Проверяем расширение
                    ext = os.path.splitext(file_path)[1].lower()
                    supported_formats = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
                    if ext in supported_formats:
                        files.append(file_path)

            if files:
                self.files_dropped.emit(files)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()


class ModelDownloader(QThread):
    """Поток для безопасного скачивания моделей"""

    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)
    log_signal = Signal(str, str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.should_stop = False

    def run(self):
        try:
            self.log_signal.emit(f"Скачивание модели {self.model_name}...", "INFO")
            self.progress_signal.emit(10, "Проверка модели...")

            if self.should_stop:
                return

            self.progress_signal.emit(30, "Загрузка модели...")

            # Безопасное создание модели
            compute_type = "float16" if DEVICE == "cuda" else "int8"

            # Предварительная очистка памяти
            CrashSafeMemoryManager.safe_gpu_cleanup("before model download")

            model = WhisperModel(
                self.model_name,
                device=DEVICE,
                compute_type=compute_type,
                cpu_threads=min(4, os.cpu_count()),
                num_workers=1,
                download_root=self.get_models_cache_dir()
            )

            self.progress_signal.emit(80, "Проверка работоспособности...")

            # Безопасное удаление модели
            del model
            CrashSafeMemoryManager.safe_gpu_cleanup("after model download")

            self.progress_signal.emit(100, "Модель готова!")
            self.finished_signal.emit(True, "Модель успешно загружена")

        except Exception as e:
            self.log_signal.emit(f"Ошибка загрузки модели: {str(e)}", "ERROR")
            CrashSafeMemoryManager.safe_gpu_cleanup("after model download error")
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


class ProfessionalTranscriptionWorker(QThread):
    """Рабочий поток транскрибации с защитой от крашей"""

    progress_signal = Signal(int, str)
    log_signal = Signal(str, str)
    finished_signal = Signal(str)
    segment_signal = Signal(str)
    stats_signal = Signal(dict)
    file_completed_signal = Signal(str, str)  # file_path, result

    def __init__(self, file_paths: List[str], settings):
        super().__init__()
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.settings = settings
        self.temp_dir = None
        self._is_running = True
        self.start_time = None
        self.current_file_index = 0

    def run(self):
        """Процесс транскрибации с защитой от крашей для множественных файлов"""
        try:
            self.start_time = time.time()
            self.log_signal.emit(f"Начало пакетной транскрибации: {len(self.file_paths)} файлов", "INFO")

            all_results = []

            for index, file_path in enumerate(self.file_paths):
                if not self._is_running:
                    break

                self.current_file_index = index
                file_name = os.path.basename(file_path)
                self.log_signal.emit(f"[{index+1}/{len(self.file_paths)}] Обработка: {file_name}", "INFO")

                # Обрабатываем один файл
                result = self.process_single_file(file_path)

                if result:
                    all_results.append(f"=== {file_name} ===\n{result}\n")
                    self.file_completed_signal.emit(file_path, result)

                # Очистка памяти между файлами
                CrashSafeMemoryManager.safe_gpu_cleanup("between files")

                if not self._is_running:
                    break

            # Объединяем все результаты
            if all_results:
                combined_result = "\n\n".join(all_results)
                self.finished_signal.emit(combined_result)
                self.log_signal.emit(f"Пакетная транскрибация завершена: {len(all_results)} файлов обработано", "SUCCESS")
            else:
                self.finished_signal.emit("Транскрибация не дала результатов")

        except Exception as e:
            self.log_signal.emit(f"Ошибка пакетной транскрибации: {str(e)}", "ERROR")
            self.finished_signal.emit(f"Ошибка: {str(e)}")
        finally:
            self.cleanup()

    def process_single_file(self, file_path):
        """Обработка одного файла"""
        model = None
        try:
            # Создаем временную директорию
            self.temp_dir = tempfile.TemporaryDirectory()
            output_audio = os.path.join(self.temp_dir.name, "audio.wav")

            # Этап 1: Извлечение аудио
            base_progress = (self.current_file_index * 100) // len(self.file_paths)
            step_progress = 100 // len(self.file_paths)

            self.progress_signal.emit(base_progress + step_progress * 10 // 100, f"[{self.current_file_index+1}/{len(self.file_paths)}] Извлечение аудио...")
            self.extract_audio(file_path, output_audio)

            if not self._is_running:
                return None

            # Предварительная очистка памяти перед загрузкой модели
            CrashSafeMemoryManager.safe_gpu_cleanup("before model loading")

            # Этап 2: Загрузка модели
            self.progress_signal.emit(base_progress + step_progress * 20 // 100, f"[{self.current_file_index+1}/{len(self.file_paths)}] Загрузка модели...")
            model = self.load_model_safely()

            if not self._is_running:
                if model:
                    del model
                    model = None
                    CrashSafeMemoryManager.safe_gpu_cleanup("after early stop")
                return None

            # Этап 3: Транскрибация
            self.progress_signal.emit(base_progress + step_progress * 30 // 100, f"[{self.current_file_index+1}/{len(self.file_paths)}] Распознавание речи...")
            segments = self.transcribe_audio_safely(output_audio, model)

            # Безопасно освобождаем модель
            if model:
                try:
                    del model
                    model = None
                except Exception as model_cleanup_error:
                    self.log_signal.emit(f"Предупреждение при удалении модели: {model_cleanup_error}", "WARNING")

            # Очистка памяти после транскрибации
            CrashSafeMemoryManager.safe_gpu_cleanup("after transcription")

            if not self._is_running:
                return None

            # Проверяем валидность сегментов
            if not segments:
                self.log_signal.emit("Сегменты не получены, возможно аудио слишком тихое", "WARNING")
                return "Не удалось получить сегменты из аудио. Проверьте качество записи."

            # Этап 4: Диаризация
            formatted_text = ""
            if self.settings.get('use_diarization'):
                self.progress_signal.emit(base_progress + step_progress * 70 // 100, f"[{self.current_file_index+1}/{len(self.file_paths)}] Диаризация...")
                formatted_text = self.apply_crash_safe_diarization(segments)
            else:
                formatted_text = self.format_simple_text_safely(segments)

            # Очистка памяти после диаризации
            CrashSafeMemoryManager.safe_gpu_cleanup("after diarization")

            if not formatted_text or formatted_text.strip() == "":
                formatted_text = "Транскрибация завершена, но результат пуст. Проверьте аудио файл."

            # Статистика
            try:
                self.send_statistics(segments, formatted_text)
            except Exception as stats_error:
                self.log_signal.emit(f"Ошибка статистики: {stats_error}", "WARNING")

            self.progress_signal.emit(base_progress + step_progress, f"[{self.current_file_index+1}/{len(self.file_paths)}] Готово")

            return formatted_text

        except Exception as e:
            self.log_signal.emit(f"Ошибка обработки файла: {str(e)}", "ERROR")

            # Защита от краша при ошибке
            try:
                if model:
                    del model
                    model = None
                CrashSafeMemoryManager.safe_gpu_cleanup("emergency cleanup")
            except Exception as cleanup_critical_error:
                self.log_signal.emit(f"Критическая ошибка защиты: {cleanup_critical_error}", "ERROR")

            return f"Ошибка обработки: {str(e)}"
        finally:
            # Гарантированная финальная очистка
            try:
                if model:
                    del model
                    model = None
            except:
                pass

            # Очистка временной директории
            if self.temp_dir:
                try:
                    self.temp_dir.cleanup()
                except:
                    pass

    def extract_audio(self, input_path, output_path):
        """Извлечение аудио с защитой"""
        ffmpeg_exe = self.find_ffmpeg()
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg не найден! Установите FFmpeg или поместите ffmpeg.exe в папку с программой")

        cmd = [
            ffmpeg_exe, "-y",
            "-i", input_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-af", "highpass=f=200,lowpass=f=3000",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка FFmpeg: {result.stderr}")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        self.log_signal.emit(f"Аудио извлечено: {size_mb:.1f} МБ", "SUCCESS")

    def find_ffmpeg(self):
        """Поиск FFmpeg"""
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).parent

        local_ffmpeg = app_dir / "ffmpeg.exe"
        if local_ffmpeg.exists():
            return str(local_ffmpeg)

        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode == 0:
                return 'ffmpeg'
        except:
            pass

        return None

    def load_model_safely(self):
        """Безопасная загрузка модели"""
        try:
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

            self.log_signal.emit(f"Создание модели с {compute_type}", "INFO")

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

        except Exception as e:
            self.log_signal.emit(f"Ошибка загрузки модели: {e}", "ERROR")
            try:
                self.log_signal.emit("Пробуем базовую модель...", "WARNING")
                model = WhisperModel(
                    "base",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=2,
                    num_workers=1,
                    download_root=ModelDownloader.get_models_cache_dir()
                )
                self.log_signal.emit("Базовая модель загружена в безопасном режиме", "WARNING")
                return model
            except Exception as fallback_e:
                self.log_signal.emit(f"Критическая ошибка загрузки: {fallback_e}", "ERROR")
                raise fallback_e

    def transcribe_audio_safely(self, audio_path, model):
        """Безопасная транскрибация"""
        try:
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

            # Безопасная обработка сегментов
            segments_list = []
            total_segments = 0

            for segment in segments:
                if not self._is_running:
                    break

                try:
                    text = self.clean_text_safely(segment.text.strip())
                    if text and len(text) > 2:
                        segments_list.append({
                            'start': float(segment.start),
                            'end': float(segment.end),
                            'text': text
                        })

                        # Отправляем сегмент для отображения
                        self.segment_signal.emit(f"[{self.format_time(segment.start)}] {text}")

                        total_segments += 1
                        if total_segments % 10 == 0:
                            # Периодическая очистка
                            if total_segments % 50 == 0:
                                CrashSafeMemoryManager.safe_gpu_cleanup("during transcription")

                except Exception as seg_error:
                    self.log_signal.emit(f"Пропускаем проблемный сегмент: {seg_error}", "DEBUG")
                    continue

            self.log_signal.emit(f"Распознано сегментов: {len(segments_list)}", "SUCCESS")
            return segments_list

        except Exception as e:
            self.log_signal.emit(f"Ошибка транскрибации: {e}", "ERROR")
            raise e

    def clean_text_safely(self, text):
        """Безопасная очистка текста"""
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

    def apply_crash_safe_diarization(self, segments):
        """Улучшенная диаризация с поддержкой множественных спикеров"""
        try:
            self.log_signal.emit("Применение улучшенной диаризации...", "INFO")

            if not segments or len(segments) == 0:
                self.log_signal.emit("Нет сегментов для диаризации", "WARNING")
                return "Нет распознанной речи для обработки."

            # Предварительная очистка памяти
            CrashSafeMemoryManager.safe_gpu_cleanup("before diarization")

            min_pause = float(self.settings.get('min_pause', 2.0))
            max_speakers = int(self.settings.get('max_speakers', 5))

            self.log_signal.emit(f"Обрабатываем {len(segments)} сегментов (макс. {max_speakers} спикеров, пауза {min_pause}с)", "INFO")

            # Шаг 1: Извлекаем характеристики каждого сегмента
            segment_features = []
            for i, seg in enumerate(segments):
                if not self.validate_segment_safely(seg, i):
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
                speech_rate = words / duration if duration > 0 else 0  # слов в секунду
                char_rate = chars / duration if duration > 0 else 0  # символов в секунду
                avg_word_len = chars / words if words > 0 else 0  # средняя длина слова

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
                return self.format_simple_text_safely(segments)

            # Шаг 2: Умная диаризация с кластеризацией
            diarized_segments = self._smart_diarization(
                segment_features, min_pause, max_speakers
            )

            # Очистка после диаризации
            CrashSafeMemoryManager.safe_gpu_cleanup("after diarization processing")

            if not diarized_segments:
                self.log_signal.emit("Диаризация не создала сегментов", "WARNING")
                return self.format_simple_text_safely(segments)

            unique_speakers = len(set(seg['speaker'] for seg in diarized_segments))
            self.log_signal.emit(f"Диаризация завершена: {len(diarized_segments)} сегментов, {unique_speakers} спикеров", "SUCCESS")

            return self.format_diarized_text_safely(diarized_segments)

        except Exception as e:
            self.log_signal.emit(f"Ошибка диаризации: {e}", "ERROR")
            CrashSafeMemoryManager.safe_gpu_cleanup("after diarization error")
            return self.format_simple_text_safely(segments)

    def _smart_diarization(self, segment_features, min_pause, max_speakers):
        """Умная диаризация с анализом характеристик речи"""
        try:
            diarized = []
            speaker_profiles = {}  # Профили спикеров
            current_speaker = 1
            last_end = 0.0

            # Цвета для спикеров (для UI)
            speaker_colors = ['#667eea', '#e53e3e', '#38a169', '#d69e2e', '#9f7aea',
                              '#ed8936', '#4299e1', '#48bb78', '#f56565', '#805ad5']

            for i, feat in enumerate(segment_features):
                try:
                    if i % 25 == 0 and not self._is_running:
                        self.log_signal.emit("Диаризация прервана", "WARNING")
                        break

                    start_time = feat['start']
                    end_time = feat['end']
                    text = feat['text']

                    # Определяем, нужна ли смена спикера
                    pause_duration = start_time - last_end if last_end > 0 else 0

                    if pause_duration > min_pause:
                        # Есть значительная пауза - возможна смена спикера
                        best_speaker = self._find_best_speaker(
                            feat, speaker_profiles, current_speaker, max_speakers
                        )

                        if best_speaker != current_speaker:
                            self.log_signal.emit(
                                f"Смена спикера: {current_speaker} -> {best_speaker} (пауза {pause_duration:.1f}с)",
                                "DEBUG"
                            )
                        current_speaker = best_speaker

                    # Обновляем профиль текущего спикера
                    self._update_speaker_profile(speaker_profiles, current_speaker, feat)

                    # Получаем цвет спикера
                    color_idx = (current_speaker - 1) % len(speaker_colors)

                    diarized.append({
                        'speaker': f"Спикер {current_speaker}",
                        'speaker_id': current_speaker,
                        'speaker_color': speaker_colors[color_idx],
                        'text': text,
                        'start': start_time,
                        'end': end_time
                    })

                    last_end = end_time

                    # Периодическая очистка
                    if i % 50 == 0 and i > 0:
                        CrashSafeMemoryManager.safe_gpu_cleanup("during smart diarization")

                except Exception as seg_error:
                    self.log_signal.emit(f"Пропускаем сегмент {i}: {seg_error}", "DEBUG")
                    continue

            return diarized

        except Exception as e:
            self.log_signal.emit(f"Ошибка умной диаризации: {e}", "ERROR")
            return []

    def _find_best_speaker(self, feat, speaker_profiles, current_speaker, max_speakers):
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
            # создаем нового спикера
            threshold = 3.0  # Порог для создания нового спикера
            num_speakers = len(speaker_profiles)

            if best_score > threshold and num_speakers < max_speakers:
                new_speaker = num_speakers + 1
                self.log_signal.emit(f"Обнаружен новый спикер {new_speaker} (score={best_score:.2f})", "DEBUG")
                return new_speaker

            # Если текущий спикер имеет схожий профиль, оставляем его
            if current_speaker in speaker_profiles:
                current_profile = speaker_profiles[current_speaker]
                if current_profile['count'] >= 2:
                    current_rate_diff = abs(curr_rate - current_profile['avg_speech_rate'])
                    current_char_diff = abs(curr_char_rate - current_profile['avg_char_rate'])
                    current_word_diff = abs(curr_avg_word - current_profile['avg_word_len'])
                    current_score = (current_rate_diff * 2.0) + (current_char_diff * 0.5) + (current_word_diff * 1.0)

                    # Если текущий спикер почти так же хорош, оставляем его
                    if current_score < best_score * 1.3:
                        return current_speaker

            return best_speaker

        except Exception as e:
            self.log_signal.emit(f"Ошибка поиска спикера: {e}", "DEBUG")
            return current_speaker

    def _update_speaker_profile(self, profiles, speaker_id, feat):
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

        except Exception as e:
            pass  # Молча игнорируем ошибки обновления профиля

    def validate_segment_safely(self, seg, index):
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

    def format_simple_text_safely(self, segments):
        """Безопасное простое форматирование"""
        try:
            if not segments or len(segments) == 0:
                return "Нет данных для форматирования."

            texts = []
            for i, seg in enumerate(segments):
                try:
                    if not self.validate_segment_safely(seg, i):
                        continue

                    text = str(seg['text']).strip()
                    if text and len(text) > 1:
                        texts.append(text)

                    if i % 100 == 0 and not self._is_running:
                        break

                except Exception:
                    continue

            if not texts:
                return "Не удалось извлечь текст из сегментов."

            result = " ".join(texts)
            result = result.replace("  ", " ").strip()

            if len(result) > 100000:
                result = result[:100000] + "\n\n[Результат обрезан для стабильности]"

            return result if result else "Пустой результат форматирования."

        except Exception as e:
            self.log_signal.emit(f"Ошибка простого форматирования: {e}", "ERROR")
            return "Ошибка при форматировании результата."

    def _format_time(self, seconds):
        """Форматирование времени в читаемый формат [MM:SS]"""
        try:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"[{mins:02d}:{secs:02d}]"
        except:
            return ""

    def format_diarized_text_safely(self, segments):
        """Безопасное форматирование диаризованного текста с HTML и цветами"""
        try:
            if not segments or len(segments) == 0:
                return "Нет сегментов для форматирования."

            # Проверяем, нужно ли показывать временные метки
            show_timestamps = self.settings.get('show_timestamps', False)

            # Цвета для спикеров
            speaker_colors = {
                1: '#667eea',  # Синий-фиолетовый
                2: '#ef4444',  # Красный
                3: '#10b981',  # Зеленый
                4: '#f59e0b',  # Оранжевый
                5: '#a78bfa',  # Светло-фиолетовый
                6: '#06b6d4',  # Голубой
                7: '#f97316',  # Оранжевый яркий
                8: '#84cc16',  # Лайм
                9: '#ec4899',  # Розовый
                10: '#8b5cf6', # Фиолетовый
            }

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
                            color = speaker_colors.get(current_speaker_id, '#667eea')
                            time_html = ""
                            if show_timestamps and current_start_time is not None:
                                time_html = f'<span style="color: #64748b; font-size: 11px;">{self._format_time(current_start_time)} </span>'
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
                color = speaker_colors.get(current_speaker_id, '#667eea')
                time_html = ""
                if show_timestamps and current_start_time is not None:
                    time_html = f'<span style="color: #64748b; font-size: 11px;">{self._format_time(current_start_time)} </span>'
                speaker_html = f'<span style="color: {color}; font-weight: bold;">{current_speaker}:</span>'
                text_html = f'<span style="color: #e2e8f0;"> {" ".join(current_texts)}</span>'
                formatted_parts.append(f'<p style="margin: 10px 0;">{time_html}{speaker_html}{text_html}</p>')

            if not formatted_parts:
                return "Не удалось сформатировать диаризованный текст."

            # HTML обертка
            result = f'''<div style="font-family: 'Segoe UI', sans-serif; line-height: 1.6;">
                {"".join(formatted_parts)}
            </div>'''

            # Также сохраняем plain text версию для экспорта
            self._plain_text_result = self._extract_plain_text(segments)

            return result if result.strip() else "Пустой результат диаризации."

        except Exception as e:
            self.log_signal.emit(f"Ошибка форматирования диаризации: {e}", "ERROR")
            return "Ошибка при форматировании диаризованного текста."

    def _extract_plain_text(self, segments):
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

    def format_time(self, seconds):
        """Форматирование времени"""
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def send_statistics(self, segments, text):
        """Отправка статистики"""
        try:
            elapsed = time.time() - self.start_time
            stats = {
                'duration': elapsed,
                'segments': len(segments),
                'words': len(text.split()),
                'chars': len(text),
                'speed': len(segments) / elapsed if elapsed > 0 else 0,
                'files_count': len(self.file_paths)
            }
            self.stats_signal.emit(stats)
        except Exception as e:
            self.log_signal.emit(f"Ошибка статистики: {e}", "WARNING")

    def stop(self):
        """Остановка процесса"""
        self._is_running = False
        self.log_signal.emit("Получен сигнал остановки...", "WARNING")

        # Даем время на корректное завершение
        try:
            if self.isRunning():
                self.quit()
                if not self.wait(3000):  # Ждем 3 секунды
                    self.log_signal.emit("Принудительное завершение...", "WARNING")
                    self.terminate()
                    self.wait(1000)  # Еще секунда
        except Exception as e:
            self.log_signal.emit(f"Ошибка остановки: {e}", "ERROR")

    def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.temp_dir:
                self.temp_dir.cleanup()
                self.log_signal.emit("Временные файлы удалены", "DEBUG")
        except Exception as e:
            self.log_signal.emit(f"Предупреждение при удалении временных файлов: {e}", "WARNING")

        # Очень мягкая финальная очистка памяти
        try:
            gc.collect()
        except:
            pass


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)

        # Включаем drag&drop для главного окна
        self.setAcceptDrops(True)

        # Темная тема
        self.setStyleSheet(self.get_dark_theme())

        if not FASTER_WHISPER_AVAILABLE:
            QMessageBox.critical(
                self, "Критическая ошибка",
                "faster_whisper не установлен!\n\n"
                "Установите: pip install faster-whisper"
            )
            sys.exit(1)

        self.init_ui()
        self.file_queue = []  # Список FileQueueItem
        self.transcription_thread = None
        self.transcribed_text = ""
        self.model_downloader = None

    def init_ui(self):
        """Создание интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок - единый контейнер с градиентом для исключения видимых швов
        header_container = QWidget()
        header_container.setStyleSheet("""
            QWidget#header_container {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #a78bfa);
                border-radius: 15px;
            }
        """)
        header_container.setObjectName("header_container")
        header_layout = QVBoxLayout(header_container)
        header_layout.setSpacing(0)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("AUDIO/VIDEO TRANSCRIPTION PRO")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: 700;
                color: #ffffff;
                padding-top: 15px;
                padding-bottom: 5px;
                letter-spacing: 2px;
                background: transparent;
            }
        """)

        author_label = QLabel(f"v{APP_VERSION} by {AUTHOR}  |  Multi-File  |  Drag & Drop  |  Smart Diarization")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: rgba(255, 255, 255, 0.85);
                padding-bottom: 12px;
                padding-top: 2px;
                background: transparent;
            }
        """)

        header_layout.addWidget(title_label)
        header_layout.addWidget(author_label)
        main_layout.addWidget(header_container)

        # Основной контент в табах
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(102, 126, 234, 0.3);
                background: #16162a;
                border-radius: 12px;
                padding: 5px;
            }
            QTabBar::tab {
                background: #1a1a2e;
                color: #a0aec0;
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 600;
                border: 1px solid transparent;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #2d2d4a;
                border-color: rgba(102, 126, 234, 0.4);
            }
        """)

        # Вкладка транскрибации
        transcription_tab = self.create_transcription_tab()
        tabs.addTab(transcription_tab, "🎙️ ТРАНСКРИБАЦИЯ")

        # Вкладка настроек
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "⚙️ НАСТРОЙКИ")

        # Вкладка логов
        logs_tab = self.create_logs_tab()
        tabs.addTab(logs_tab, "📊 ЛОГИ")

        main_layout.addWidget(tabs)

        # Статус бар
        self.create_status_bar()

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Обработка входа перетаскивания для главного окна"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Обработка сброса файлов на главное окно"""
        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    # Проверяем расширение
                    ext = os.path.splitext(file_path)[1].lower()
                    supported_formats = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
                    if ext in supported_formats:
                        files.append(file_path)

            if files:
                self.add_files_to_queue(files)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def create_transcription_tab(self):
        """Создание вкладки транскрибации"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Выбор файлов
        file_group = QGroupBox("📁 Выбор файлов")
        file_layout = QVBoxLayout()

        # Кнопки управления файлами
        file_buttons_layout = QHBoxLayout()

        self.select_file_btn = QPushButton("+ Добавить файлы")
        self.select_file_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #047857);
            }
        """)

        self.clear_queue_btn = QPushButton("Очистить")
        self.clear_queue_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ef4444, stop:1 #dc2626);
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc2626, stop:1 #b91c1c);
            }
        """)

        self.remove_selected_btn = QPushButton("Удалить выбранные")
        self.remove_selected_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f59e0b, stop:1 #d97706);
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d97706, stop:1 #b45309);
            }
        """)

        file_buttons_layout.addWidget(self.select_file_btn)
        file_buttons_layout.addWidget(self.remove_selected_btn)
        file_buttons_layout.addWidget(self.clear_queue_btn)
        file_buttons_layout.addStretch()

        # Зона для drag&drop файлов - вся область принимает файлы
        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self.add_files_to_queue)
        # Для совместимости используем file_list_widget как ссылку на внутренний список
        self.file_list_widget = self.drop_zone.file_list

        file_layout.addLayout(file_buttons_layout)
        file_layout.addWidget(self.drop_zone)
        file_group.setLayout(file_layout)

        # Контролы
        control_layout = QHBoxLayout()

        self.transcribe_btn = QPushButton("НАЧАТЬ ТРАНСКРИБАЦИЮ")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #a78bfa);
                color: white;
                border: none;
                padding: 14px 35px;
                font-size: 15px;
                font-weight: 700;
                border-radius: 12px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5a67d8, stop:0.5 #6b46c1, stop:1 #9061f9);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4c51bf, stop:0.5 #553c9a, stop:1 #7c3aed);
            }
            QPushButton:disabled {
                background: #1a1a2e;
                color: #4a5568;
            }
        """)

        self.stop_btn = QPushButton("ОСТАНОВИТЬ")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ef4444, stop:1 #dc2626);
                color: white;
                border: none;
                padding: 14px 35px;
                font-size: 15px;
                font-weight: 700;
                border-radius: 12px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc2626, stop:1 #b91c1c);
            }
            QPushButton:disabled {
                background: #1a1a2e;
                color: #4a5568;
            }
        """)

        self.download_model_btn = QPushButton("Скачать модель")
        self.download_model_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                color: white;
                border: none;
                padding: 14px 25px;
                font-size: 13px;
                font-weight: 600;
                border-radius: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #047857);
            }
            QPushButton:disabled {
                background: #1a1a2e;
                color: #4a5568;
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
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
                text-align: center;
                font-weight: 600;
                font-size: 12px;
                background-color: #0f0f1a;
                height: 28px;
                color: #e2e8f0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #a78bfa);
                border-radius: 11px;
            }
        """)
        self.progress_bar.hide()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                font-size: 12px;
                padding: 5px;
                background: transparent;
            }
        """)
        self.status_label.hide()

        # Splitter для результатов
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Результат
        result_group = QGroupBox("РЕЗУЛЬТАТ")
        result_layout = QVBoxLayout()

        self.result_text = QTextBrowser()
        self.result_text.setOpenExternalLinks(False)
        self.result_text.setStyleSheet("""
            QTextBrowser {
                background-color: #0a0a14;
                color: #e2e8f0;
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 10px;
                padding: 15px;
                font-size: 13px;
                line-height: 1.6;
            }
        """)

        # Кнопки сохранения в прокручиваемом контейнере
        save_scroll_area = QScrollArea()
        save_scroll_area.setWidgetResizable(True)
        save_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        save_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        save_scroll_area.setMaximumHeight(55)
        save_scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:horizontal {
                background: #1a1a2e;
                height: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal {
                background: #667eea;
                border-radius: 3px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

        save_widget = QWidget()
        save_widget.setStyleSheet("background: transparent;")
        save_layout = QHBoxLayout(save_widget)
        save_layout.setContentsMargins(0, 0, 0, 0)

        self.save_txt_btn = self.create_save_button("💾 TXT", "#48bb78")
        self.save_docx_btn = self.create_save_button("📄 DOCX", "#4299e1")
        self.save_json_btn = self.create_save_button("📊 JSON", "#ed8936")
        self.save_all_btn = self.create_save_button("📦 Сохранить все", "#9f7aea")

        save_layout.addStretch()
        save_layout.addWidget(self.save_txt_btn)
        save_layout.addWidget(self.save_docx_btn)
        save_layout.addWidget(self.save_json_btn)
        save_layout.addWidget(self.save_all_btn)

        save_scroll_area.setWidget(save_widget)

        result_layout.addWidget(self.result_text)
        result_layout.addWidget(save_scroll_area)
        result_group.setLayout(result_layout)

        # Живая транскрипция
        live_group = QGroupBox("🔴 ЖИВАЯ ТРАНСКРИПЦИЯ")
        live_layout = QVBoxLayout()

        self.live_text = QTextEdit()
        self.live_text.setReadOnly(True)
        self.live_text.setMaximumHeight(150)
        self.live_text.setPlaceholderText("Сегменты будут появляться здесь в реальном времени...")

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
        self.select_file_btn.clicked.connect(self.select_files)
        self.clear_queue_btn.clicked.connect(self.clear_file_queue)
        self.remove_selected_btn.clicked.connect(self.remove_selected_files)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.stop_btn.clicked.connect(self.stop_transcription)
        self.download_model_btn.clicked.connect(self.download_model)
        self.save_txt_btn.clicked.connect(lambda: self.save_results('txt'))
        self.save_docx_btn.clicked.connect(lambda: self.save_results('docx'))
        self.save_json_btn.clicked.connect(lambda: self.save_results('json'))
        self.save_all_btn.clicked.connect(self.save_all_results)

        return widget

    def create_settings_tab(self):
        """Создание вкладки настроек"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Основные настройки
        basic_group = QGroupBox("⚙️ ОСНОВНЫЕ НАСТРОЙКИ")
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
        self.model_combo.setCurrentIndex(2)  # small по умолчанию
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()

        basic_layout.addLayout(lang_layout)
        basic_layout.addLayout(model_layout)
        basic_group.setLayout(basic_layout)

        # Диаризация
        diarization_group = QGroupBox("👥 ДИАРИЗАЦИЯ СПИКЕРОВ")
        diarization_layout = QVBoxLayout()

        self.diarization_checkbox = QCheckBox("Включить диаризацию спикеров")
        self.diarization_checkbox.setChecked(True)

        # Информация о диаризации
        info_label = QLabel("Автоматическое определение спикеров (до 10)")
        info_label.setStyleSheet("color: #56d364; font-size: 11px; margin: 5px 0px;")
        info_label.setWordWrap(True)

        # Чекбокс для отображения времени реплик
        self.show_timestamps_checkbox = QCheckBox("Показывать время каждой реплики")
        self.show_timestamps_checkbox.setChecked(False)
        self.show_timestamps_checkbox.setToolTip("Добавлять временные метки к каждой реплике в результате")

        # Параметры диаризации
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Мин. пауза (сек):"))
        self.min_pause_spin = QSpinBox()
        self.min_pause_spin.setMinimum(1)
        self.min_pause_spin.setMaximum(10)
        self.min_pause_spin.setValue(2)
        self.min_pause_spin.setToolTip("Минимальная пауза для возможной смены спикера")
        params_layout.addWidget(self.min_pause_spin)

        params_layout.addWidget(QLabel("Мин. тишина (мс):"))
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setMinimum(500)
        self.min_silence_spin.setMaximum(3000)
        self.min_silence_spin.setSingleStep(100)
        self.min_silence_spin.setValue(1000)
        self.min_silence_spin.setToolTip("Минимальная длительность тишины для VAD")
        params_layout.addWidget(self.min_silence_spin)

        params_layout.addStretch()

        diarization_layout.addWidget(self.diarization_checkbox)
        diarization_layout.addWidget(info_label)
        diarization_layout.addWidget(self.show_timestamps_checkbox)
        diarization_layout.addLayout(params_layout)
        diarization_group.setLayout(diarization_layout)

        # Подключаем обработчик
        self.diarization_checkbox.stateChanged.connect(self.on_diarization_changed)

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
                background: #0f0f1a;
                color: #94a3b8;
                border-top: 1px solid rgba(102, 126, 234, 0.2);
                padding: 4px 10px;
                font-size: 11px;
            }
            QStatusBar::item {
                border: none;
            }
        """)

        label_style = """
            QLabel {
                color: #94a3b8;
                padding: 0 15px;
                background: transparent;
            }
        """

        self.memory_label = QLabel("RAM: --")
        self.memory_label.setStyleSheet(label_style)
        self.gpu_label = QLabel("GPU: --")
        self.gpu_label.setStyleSheet(label_style)
        self.time_label = QLabel("--:--:--")
        self.time_label.setStyleSheet(label_style)
        self.queue_label = QLabel("Очередь: 0")
        self.queue_label.setStyleSheet(label_style)

        status_bar.addPermanentWidget(self.queue_label)
        status_bar.addPermanentWidget(self.memory_label)
        status_bar.addPermanentWidget(self.gpu_label)
        status_bar.addPermanentWidget(self.time_label)

        # Таймер для обновления статистики
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

    def create_save_button(self, text, color):
        """Создание кнопки сохранения с современным стилем"""
        btn = QPushButton(text)
        btn.setEnabled(False)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                padding: 10px 22px;
                font-weight: 600;
                font-size: 12px;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background: {color}dd;
                transform: translateY(-1px);
            }}
            QPushButton:pressed {{
                background: {color}bb;
            }}
            QPushButton:disabled {{
                background: #1a1a2e;
                color: #4a5568;
            }}
        """)
        return btn

    def get_dark_theme(self):
        """Современная темная тема с улучшенным дизайном"""
        return """
            QMainWindow {
                background-color: #0f0f1a;
            }
            QWidget {
                background-color: #16162a;
                color: #e2e8f0;
                font-family: 'Segoe UI', 'SF Pro Display', -apple-system, Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 12px;
                margin-top: 12px;
                padding: 15px;
                padding-top: 25px;
                font-weight: 600;
                font-size: 14px;
                background-color: rgba(22, 22, 42, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 12px;
                color: #a78bfa;
                background-color: #16162a;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #2d2d4a;
                border: 1px solid rgba(102, 126, 234, 0.4);
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 600;
                color: #e2e8f0;
            }
            QPushButton:hover {
                background-color: #3d3d5c;
                border-color: rgba(102, 126, 234, 0.7);
            }
            QPushButton:pressed {
                background-color: #1e1e3a;
            }
            QPushButton:disabled {
                background-color: #1a1a2e;
                color: #4a5568;
                border-color: rgba(74, 85, 104, 0.3);
            }
            QComboBox {
                background-color: #2d2d4a;
                border: 1px solid rgba(102, 126, 234, 0.3);
                padding: 8px 12px;
                border-radius: 8px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: rgba(102, 126, 234, 0.6);
            }
            QComboBox:drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #a78bfa;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d4a;
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                selection-background-color: #667eea;
                padding: 4px;
            }
            QLineEdit {
                background-color: #2d2d4a;
                border: 1px solid rgba(102, 126, 234, 0.3);
                padding: 8px 12px;
                border-radius: 8px;
            }
            QLineEdit:focus {
                border-color: #667eea;
            }
            QTextEdit {
                background-color: #0f0f1a;
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 10px;
                padding: 12px;
                selection-background-color: #667eea;
            }
            QTextEdit:focus {
                border-color: rgba(102, 126, 234, 0.5);
            }
            QProgressBar {
                text-align: center;
                border: none;
                border-radius: 10px;
                background-color: #1a1a2e;
                height: 20px;
            }
            QProgressBar::chunk {
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #a78bfa);
            }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(y1:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(y1:0, y2:1,
                    stop:0 #7c8ff0, stop:1 #8b5fcf);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #1a1a2e;
                height: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal {
                background: qlineargradient(x1:0, x2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 5px;
                min-width: 30px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QSpinBox {
                background-color: #2d2d4a;
                border: 1px solid rgba(102, 126, 234, 0.3);
                padding: 6px 10px;
                border-radius: 8px;
                min-width: 70px;
            }
            QSpinBox:hover {
                border-color: rgba(102, 126, 234, 0.6);
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: transparent;
                border: none;
                width: 16px;
            }
            QSpinBox::up-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #a78bfa;
            }
            QSpinBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #a78bfa;
            }
            QCheckBox {
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 2px solid rgba(102, 126, 234, 0.4);
                background-color: #1a1a2e;
            }
            QCheckBox::indicator:hover {
                border-color: rgba(102, 126, 234, 0.7);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-color: #667eea;
            }
            QMenuBar {
                background-color: #16162a;
                color: #e2e8f0;
                border-bottom: 1px solid rgba(102, 126, 234, 0.2);
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                border-radius: 6px;
            }
            QMenuBar::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
            }
            QMenu {
                background-color: #2d2d4a;
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
            }
            QLabel {
                color: #e2e8f0;
            }
            QToolTip {
                background-color: #2d2d4a;
                color: #e2e8f0;
                border: 1px solid rgba(102, 126, 234, 0.4);
                border-radius: 6px;
                padding: 6px 10px;
            }
            QSplitter::handle {
                background: rgba(102, 126, 234, 0.3);
                height: 3px;
            }
            QSplitter::handle:hover {
                background: #667eea;
            }
        """

    def on_diarization_changed(self, state):
        """Обработчик изменения чекбокса диаризации"""
        enabled = state == Qt.CheckState.Checked

        # Включаем/выключаем параметры
        self.min_pause_spin.setEnabled(enabled)
        self.min_silence_spin.setEnabled(enabled)
        self.show_timestamps_checkbox.setEnabled(enabled)

        self.log_widget.log(f"Диаризация {'включена' if enabled else 'отключена'}", "INFO")

    def update_system_info(self):
        """Обновление информации о системе"""
        info = []

        # Python
        info.append(f"Python: {sys.version.split()[0]}")

        # GPU
        if TORCH_AVAILABLE:
            if DEVICE == "cuda":
                try:
                    import torch
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    info.append(f"GPU: {gpu_name} ({vram:.1f} GB)")
                except:
                    info.append("GPU: Обнаружен")
            else:
                info.append("Режим: CPU")
        else:
            info.append("PyTorch не установлен")

        # FFmpeg
        ffmpeg_found = os.path.exists('ffmpeg.exe') or self.check_ffmpeg()
        info.append(f"FFmpeg: {'✅ Найден' if ffmpeg_found else '❌ Не найден'}")

        # Модели
        cache_dir = ModelDownloader.get_models_cache_dir()
        if cache_dir.exists():
            models = list(cache_dir.glob("*"))
            info.append(f"Кэш моделей: {len(models)} файлов")

        info.append("Защита от крашей: активна")
        info.append("Множественные файлы: поддерживается")
        info.append("Drag & Drop: активен")

        self.system_info.setPlainText("\n".join(info))

    def check_ffmpeg(self):
        """Проверка FFmpeg"""
        try:
            if getattr(sys, 'frozen', False):
                app_dir = Path(sys.executable).parent
            else:
                app_dir = Path(__file__).parent

            local_ffmpeg = app_dir / "ffmpeg.exe"
            if local_ffmpeg.exists():
                return True

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
            self.memory_label.setText(f"RAM: {memory:.1f} GB")
        except:
            pass

        # GPU
        if DEVICE == "cuda" and TORCH_AVAILABLE:
            try:
                import torch
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                self.gpu_label.setText(f"VRAM: {allocated:.1f}/{reserved:.1f} GB")
            except:
                pass
        else:
            self.gpu_label.setText("CPU")

        # Время
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)

        # Очередь
        self.queue_label.setText(f"Очередь: {len(self.file_queue)}")

    def select_files(self):
        """Выбор файлов (множественный)"""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Выберите видео или аудио файлы",
                "",
                "Медиа файлы (*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav *.m4a *.aac *.flac *.ogg);;Все файлы (*.*)"
            )

            if file_paths:
                self.add_files_to_queue(file_paths)

        except Exception as e:
            self.log_widget.log(f"Ошибка выбора файлов: {e}", "ERROR")
            QMessageBox.critical(self, "Ошибка", f"Не удалось выбрать файлы: {e}")

    def add_files_to_queue(self, file_paths):
        """Добавление файлов в очередь"""
        added_count = 0

        for file_path in file_paths:
            # Проверяем, не добавлен ли уже файл
            if any(item.path == file_path for item in self.file_queue):
                self.log_widget.log(f"Файл уже в очереди: {os.path.basename(file_path)}", "WARNING")
                continue

            # Создаем элемент очереди
            file_item = FileQueueItem(file_path)
            self.file_queue.append(file_item)

            # Добавляем в список UI
            list_item = QListWidgetItem(f"📄 {file_item.name} ({file_item.size:.1f} MB)")
            self.file_list_widget.addItem(list_item)

            self.log_widget.log(f"Добавлен в очередь: {file_item.name}", "SUCCESS")
            added_count += 1

        if added_count > 0:
            self.log_widget.log(f"Добавлено файлов: {added_count}", "INFO")
            self.transcribe_btn.setEnabled(True)

            # Очистка предыдущих результатов
            self.result_text.clear()
            self.live_text.clear()
            self.stats_text.clear()

    def remove_selected_files(self):
        """Удаление выбранных файлов из очереди"""
        selected_items = self.file_list_widget.selectedItems()

        if not selected_items:
            return

        for item in selected_items:
            row = self.file_list_widget.row(item)
            self.file_list_widget.takeItem(row)

            # Удаляем из очереди
            if row < len(self.file_queue):
                removed_file = self.file_queue.pop(row)
                self.log_widget.log(f"Удален из очереди: {removed_file.name}", "INFO")

        # Отключаем кнопку транскрибации если очередь пуста
        if not self.file_queue:
            self.transcribe_btn.setEnabled(False)

    def clear_file_queue(self):
        """Очистка всей очереди файлов"""
        if not self.file_queue:
            return

        reply = QMessageBox.question(
            self,
            "Подтверждение",
            f"Очистить очередь из {len(self.file_queue)} файлов?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.file_list_widget.clear()
            self.file_queue.clear()
            self.transcribe_btn.setEnabled(False)
            self.log_widget.log("Очередь файлов очищена", "INFO")

    def download_model(self):
        """Скачивание модели"""
        model_name = self.model_combo.currentText().split(' - ')[0]

        reply = QMessageBox.question(
            self,
            "Скачивание модели",
            f"Скачать модель {model_name}?\n\nЭто может занять время в зависимости от размера модели.",
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
        if not self.file_queue:
            QMessageBox.warning(self, "Предупреждение", "Сначала добавьте файлы для транскрибации")
            return

        # UI
        self.transcribe_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.download_model_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        self.clear_queue_btn.setEnabled(False)
        self.remove_selected_btn.setEnabled(False)

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
            'min_pause': float(self.min_pause_spin.value()),
            'min_silence': int(self.min_silence_spin.value()),
            'max_speakers': 10,  # Автоопределение - до 10 спикеров
            'show_timestamps': self.show_timestamps_checkbox.isChecked()
        }

        # Список путей файлов
        file_paths = [item.path for item in self.file_queue]

        self.log_widget.log("=" * 50, "INFO")
        self.log_widget.log(f"Начало пакетной транскрибации: {len(file_paths)} файлов", "INFO")
        self.log_widget.log(f"Настройки: {json.dumps(settings, ensure_ascii=False)}", "DEBUG")

        # Мягкая предварительная очистка памяти
        try:
            gc.collect()
        except:
            pass

        # Запуск потока
        try:
            self.transcription_thread = ProfessionalTranscriptionWorker(file_paths, settings)
            self.transcription_thread.progress_signal.connect(self.on_progress)
            self.transcription_thread.log_signal.connect(self.log_widget.log)
            self.transcription_thread.finished_signal.connect(self.on_finished)
            self.transcription_thread.segment_signal.connect(self.on_segment)
            self.transcription_thread.stats_signal.connect(self.on_stats)
            self.transcription_thread.file_completed_signal.connect(self.on_file_completed)
            self.transcription_thread.start()

            self.log_widget.log("Рабочий поток запущен", "SUCCESS")

        except Exception as thread_error:
            self.log_widget.log(f"Ошибка запуска потока: {thread_error}", "ERROR")
            self.on_finished(f"Ошибка запуска: {thread_error}")

    def stop_transcription(self):
        """Остановка транскрибации"""
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.log_widget.log("Остановка транскрибации...", "WARNING")

            try:
                self.transcription_thread.stop()

                # Ждем корректного завершения
                if not self.transcription_thread.wait(5000):  # 5 секунд
                    self.log_widget.log("Принудительное завершение потока...", "WARNING")
                    self.transcription_thread.terminate()
                    self.transcription_thread.wait(2000)  # Еще 2 секунды

                self.log_widget.log("Поток остановлен", "INFO")

            except Exception as e:
                self.log_widget.log(f"Ошибка остановки потока: {e}", "ERROR")
            finally:
                # В любом случае вызываем завершение
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
СТАТИСТИКА ТРАНСКРИБАЦИИ:
⏱️ Время обработки: {stats['duration']:.1f} сек
📂 Файлов обработано: {stats.get('files_count', 1)}
📝 Сегментов: {stats['segments']}
💬 Слов: {stats['words']}
📄 Символов: {stats['chars']}
⚡ Скорость: {stats['speed']:.1f} сегм/сек
"""
        self.stats_text.setPlainText(text.strip())

    @Slot(str, str)
    def on_file_completed(self, file_path, result):
        """Обработка завершения транскрибации одного файла"""
        file_name = os.path.basename(file_path)
        self.log_widget.log(f"Файл обработан: {file_name}", "SUCCESS")

        # Обновляем статус файла в очереди
        for i, item in enumerate(self.file_queue):
            if item.path == file_path:
                item.status = "Готово"
                item.result = result
                # Обновляем отображение в списке
                list_item = self.file_list_widget.item(i)
                if list_item:
                    list_item.setText(f"✅ {item.name} ({item.size:.1f} MB) - Готово")
                break

    @Slot(str)
    def on_finished(self, result):
        """Завершение транскрибации"""
        try:
            self.progress_bar.hide()
            self.status_label.hide()
            self.transcribe_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.download_model_btn.setEnabled(True)
            self.select_file_btn.setEnabled(True)
            self.clear_queue_btn.setEnabled(True)
            self.remove_selected_btn.setEnabled(True)

            if result.startswith("Ошибка"):
                QMessageBox.critical(self, "Ошибка", result)
                self.log_widget.log(result, "ERROR")
            else:
                # Отображаем результат
                if '<div' in result or '<span' in result:
                    # Это HTML - отображаем с цветами
                    self.result_text.setHtml(result)
                    # Для экспорта используем plain text из QTextBrowser
                    self.transcribed_text = self.result_text.toPlainText()
                else:
                    self.transcribed_text = result
                    self.result_text.setPlainText(result)

                # Активация кнопок сохранения
                for btn in [self.save_txt_btn, self.save_docx_btn, self.save_json_btn, self.save_all_btn]:
                    btn.setEnabled(True)

                # Скролл к началу
                cursor = self.result_text.textCursor()
                cursor.setPosition(0)
                self.result_text.setTextCursor(cursor)

                self.log_widget.log(f"Транскрибация завершена успешно: {len(self.file_queue)} файлов", "SUCCESS")
                self.log_widget.log("=" * 50, "INFO")

            # ВАЖНО: Очистка потока и состояния
            if self.transcription_thread:
                try:
                    if self.transcription_thread.isRunning():
                        self.transcription_thread.wait(1000)  # Ждем до 1 секунды
                except:
                    pass
                finally:
                    self.transcription_thread = None

            # Мягкая очистка памяти для подготовки к следующей транскрибации
            try:
                gc.collect()
                time.sleep(0.1)  # Даем время на стабилизацию
            except:
                pass

            self.log_widget.log("Система готова к новой транскрибации", "INFO")

        except Exception as e:
            self.log_widget.log(f"Ошибка завершения транскрибации: {e}", "ERROR")
            # Принудительная очистка при ошибке
            self.transcription_thread = None

    def save_results(self, format_type):
        """Сохранение результатов"""
        if not self.transcribed_text:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if len(self.file_queue) == 1:
            base_name = Path(self.file_queue[0].path).stem
        else:
            base_name = f"batch_{len(self.file_queue)}_files"

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
                        f.write(f"ПРОФЕССИОНАЛЬНАЯ ТРАНСКРИБАЦИЯ\n")
                        f.write(f"Файлов обработано: {len(self.file_queue)}\n")
                        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Программа: {APP_NAME} v{APP_VERSION}\n")
                        f.write(f"Автор: {AUTHOR}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(self.transcribed_text)

                    self.log_widget.log(f"Результат сохранен: {file_path}", "SUCCESS")
                    QMessageBox.information(self, "Успех", "Результат успешно сохранен!")
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
                    doc.add_heading('ПРОФЕССИОНАЛЬНАЯ ТРАНСКРИБАЦИЯ', 0)

                    # Метаданные
                    doc.add_paragraph(f'Файлов обработано: {len(self.file_queue)}')
                    for item in self.file_queue:
                        doc.add_paragraph(f'• {item.name} ({item.size:.1f} MB)')
                    doc.add_paragraph(f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    doc.add_paragraph(f'Программа: {APP_NAME} v{APP_VERSION}')
                    doc.add_paragraph(f'Автор: {AUTHOR}')
                    doc.add_paragraph(f'Модель: {self.model_combo.currentText()}')
                    doc.add_paragraph(f'Язык: {self.language_combo.currentText()}')

                    doc.add_heading('Результат', level=1)

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
                            'files': [{'name': item.name, 'path': item.path, 'size_mb': item.size} for item in self.file_queue],
                            'files_count': len(self.file_queue),
                            'date': datetime.now().isoformat(),
                            'program': f"{APP_NAME} v{APP_VERSION}",
                            'author': AUTHOR,
                            'model': self.model_combo.currentText(),
                            'language': self.language_combo.currentText(),
                            'professional_edition': True
                        },
                        'text': self.transcribed_text,
                        'individual_results': {},
                        'statistics': self.stats_text.toPlainText() if self.stats_text.toPlainText() else None
                    }

                    # Добавляем индивидуальные результаты
                    for item in self.file_queue:
                        if item.result:
                            data['individual_results'][item.name] = item.result

                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    self.log_widget.log(f"JSON сохранен: {file_path}", "SUCCESS")
                    QMessageBox.information(self, "Успех", "JSON успешно сохранен!")
                except Exception as e:
                    self.log_widget.log(f"Ошибка сохранения: {e}", "ERROR")
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

    def save_all_results(self):
        """Сохранение всех результатов по отдельным файлам"""
        if not self.file_queue or not any(item.result for item in self.file_queue):
            QMessageBox.warning(self, "Внимание", "Нет результатов для сохранения")
            return

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для сохранения результатов"
        )

        if not folder_path:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_count = 0

        try:
            for item in self.file_queue:
                if not item.result:
                    continue

                base_name = Path(item.path).stem

                # Сохраняем TXT
                txt_path = os.path.join(folder_path, f"{base_name}_{timestamp}.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"ТРАНСКРИБАЦИЯ: {item.name}\n")
                    f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Размер файла: {item.size:.1f} MB\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(item.result)

                saved_count += 1
                self.log_widget.log(f"Сохранен: {txt_path}", "SUCCESS")

            # Сохраняем общий отчет
            report_path = os.path.join(folder_path, f"report_{timestamp}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"ОТЧЕТ О ПАКЕТНОЙ ТРАНСКРИБАЦИИ\n")
                f.write(f"Программа: {APP_NAME} v{APP_VERSION}\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Файлов обработано: {len(self.file_queue)}\n")
                f.write(f"Успешно транскрибировано: {saved_count}\n")
                f.write("=" * 50 + "\n\n")
                f.write("СПИСОК ФАЙЛОВ:\n")
                for item in self.file_queue:
                    status = "✅ Готово" if item.result else "❌ Ошибка"
                    f.write(f"{status} - {item.name} ({item.size:.1f} MB)\n")

            self.log_widget.log(f"Сохранен отчет: {report_path}", "SUCCESS")

            QMessageBox.information(
                self,
                "Успех",
                f"Результаты сохранены!\n\n"
                f"Сохранено файлов: {saved_count}\n"
                f"Папка: {folder_path}"
            )

        except Exception as e:
            self.log_widget.log(f"Ошибка сохранения результатов: {e}", "ERROR")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты: {e}")

    def export_logs(self):
        """Экспорт логов"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт логов",
            f"professional_logs_{timestamp}.txt",
            "Текстовые файлы (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"ЛОГИ {APP_NAME} v{APP_VERSION}\n")
                    f.write(f"Автор: {AUTHOR}\n")
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
                self.log_widget.log("Принудительное закрытие программы...", "WARNING")

                # Остановка потоков без агрессивной очистки
                try:
                    if self.transcription_thread:
                        self.transcription_thread.stop()
                    if self.model_downloader:
                        self.model_downloader.stop()
                except:
                    pass

                event.accept()
            else:
                event.ignore()
        else:
            # Обычное закрытие без принудительной очистки GPU
            event.accept()


def main():
    """Главная функция"""
    try:
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create("Fusion"))
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(APP_VERSION)

        # Проверка критических зависимостей
        if not FASTER_WHISPER_AVAILABLE:
            QMessageBox.critical(
                None, "Критическая ошибка",
                "faster-whisper не установлен!\n\n"
                "Установите командой:\n"
                "pip install faster-whisper"
            )
            sys.exit(1)

        # Создаем меню О программе
        app.aboutToQuit.connect(lambda: None)  # Убираем агрессивную очистку при выходе

        window = MainWindow()

        # Добавляем меню
        menubar = window.menuBar()
        help_menu = menubar.addMenu("Помощь")
        about_action = help_menu.addAction("О программе")
        about_action.triggered.connect(lambda: AboutDialog(window).exec())

        window.show()

        window.log_widget.log("=" * 50, "INFO")
        window.log_widget.log(f"{APP_NAME} v{APP_VERSION} запущен", "SUCCESS")
        window.log_widget.log(f"Автор: {AUTHOR}", "INFO")
        window.log_widget.log("✨ Поддержка множественных файлов активна", "INFO")
        window.log_widget.log("✨ Drag & Drop активен", "INFO")
        window.log_widget.log("✨ Защита от крашей активирована", "INFO")
        window.log_widget.log("=" * 50, "INFO")

        sys.exit(app.exec())

    except Exception as e:
        print(f"Критическая ошибка запуска: {e}")
        import traceback
        traceback.print_exc()
        try:
            QMessageBox.critical(None, "Критическая ошибка", f"Не удалось запустить программу:\n{e}")
        except:
            pass


if __name__ == "__main__":
    main()
