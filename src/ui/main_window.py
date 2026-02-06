"""
Главное окно приложения
"""

import os
import gc
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QTextEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QMessageBox, QProgressBar, QComboBox, QCheckBox,
    QSpinBox, QTabWidget, QTextBrowser, QSplitter, QLineEdit,
    QListWidgetItem, QScrollArea, QTimer
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QTextCursor, QDragEnterEvent, QDropEvent

from ..utils import (
    APP_NAME, APP_VERSION, AUTHOR, SUPPORTED_FORMATS,
    FASTER_WHISPER_AVAILABLE, DOCX_AVAILABLE, TORCH_AVAILABLE, DEVICE,
    PYANNOTE_AVAILABLE, DEFAULT_HF_TOKEN, NEMO_AVAILABLE,
    NEMO_DEFAULT_MODEL, NEMO_CONFORMER_MODELS
)
from ..core import get_models_cache_dir, check_ffmpeg
from ..workers import ModelDownloader, TranscriptionWorker
from .widgets import LogWidget, DropZoneWidget
from .styles import get_dark_theme
from .dialogs import AboutDialog


class FileQueueItem:
    """Элемент очереди файлов"""
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.size = os.path.getsize(path) / (1024 ** 2)  # MB
        self.status = "В очереди"
        self.result = ""


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
        self.setStyleSheet(get_dark_theme())

        if not FASTER_WHISPER_AVAILABLE and not NEMO_AVAILABLE:
            QMessageBox.critical(
                self, "Критическая ошибка",
                "Не найдено ни одного ASR движка!\n\n"
                "Установите хотя бы один из них:\n"
                "pip install faster-whisper\n"
                "pip install nemo_toolkit[asr]"
            )
            sys.exit(1)

        self._init_ui()
        self.file_queue = []
        self.transcription_thread = None
        self.transcribed_text = ""
        self.model_downloader = None

    def _init_ui(self):
        """Создание интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
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

        # Табы
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

        tabs.addTab(self._create_transcription_tab(), "🎙️ ТРАНСКРИБАЦИЯ")
        tabs.addTab(self._create_settings_tab(), "⚙️ НАСТРОЙКИ")
        tabs.addTab(self._create_logs_tab(), "📊 ЛОГИ")

        main_layout.addWidget(tabs)
        self._create_status_bar()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in SUPPORTED_FORMATS:
                        files.append(file_path)
            if files:
                self.add_files_to_queue(files)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def _create_transcription_tab(self):
        """Создание вкладки транскрибации"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Выбор файлов
        file_group = QGroupBox("📁 Выбор файлов")
        file_layout = QVBoxLayout()

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

        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self.add_files_to_queue)
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

        # Кнопки сохранения
        save_scroll_area = QScrollArea()
        save_scroll_area.setWidgetResizable(True)
        save_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        save_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        save_scroll_area.setMaximumHeight(55)
        save_scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
        """)

        save_widget = QWidget()
        save_widget.setStyleSheet("background: transparent;")
        save_layout = QHBoxLayout(save_widget)
        save_layout.setContentsMargins(0, 0, 0, 0)

        self.save_txt_btn = self._create_save_button("💾 TXT", "#48bb78")
        self.save_docx_btn = self._create_save_button("📄 DOCX", "#4299e1")
        self.save_json_btn = self._create_save_button("📊 JSON", "#ed8936")
        self.save_all_btn = self._create_save_button("📦 Сохранить все", "#9f7aea")

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

    def _create_settings_tab(self):
        """Создание вкладки настроек"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Основные настройки
        basic_group = QGroupBox("⚙️ ОСНОВНЫЕ НАСТРОЙКИ")
        basic_layout = QVBoxLayout()

        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("ASR движок:"))
        self.asr_backend_combo = QComboBox()
        self.asr_backend_combo.addItem("whisper - faster-whisper (рекомендуется)", "whisper")
        self.asr_backend_combo.addItem("nemo - NVIDIA NeMo Conformer", "nemo")
        if not FASTER_WHISPER_AVAILABLE:
            self.asr_backend_combo.setItemData(0, 0, Qt.ItemDataRole.UserRole - 1)
        if not NEMO_AVAILABLE:
            self.asr_backend_combo.setItemData(1, 0, Qt.ItemDataRole.UserRole - 1)
        if not FASTER_WHISPER_AVAILABLE and NEMO_AVAILABLE:
            self.asr_backend_combo.setCurrentIndex(1)
        backend_layout.addWidget(self.asr_backend_combo)
        backend_layout.addStretch()

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Язык:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ru - Русский", "en - English", "auto - Автоопределение"])
        self.language_combo.setCurrentIndex(0)
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()

        self.whisper_model_container = QWidget()
        model_layout = QHBoxLayout(self.whisper_model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(QLabel("Модель Whisper:"))
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

        self.nemo_model_container = QWidget()
        nemo_layout = QHBoxLayout(self.nemo_model_container)
        nemo_layout.setContentsMargins(0, 0, 0, 0)
        nemo_layout.addWidget(QLabel("NeMo модель:"))
        self.nemo_model_combo = QComboBox()
        for label, model_name in NEMO_CONFORMER_MODELS:
            self.nemo_model_combo.addItem(f"{label} ({model_name})", model_name)
        self.nemo_model_combo.addItem("Custom...", "custom")
        self.nemo_model_combo.setCurrentIndex(0)
        nemo_layout.addWidget(self.nemo_model_combo)
        self.nemo_model_edit = QLineEdit()
        self.nemo_model_edit.setPlaceholderText("Например: stt_en_conformer_ctc_large")
        self.nemo_model_edit.setText(NEMO_DEFAULT_MODEL)
        self.nemo_model_edit.setVisible(False)
        nemo_layout.addWidget(self.nemo_model_edit)
        nemo_layout.addStretch()

        basic_layout.addLayout(backend_layout)
        basic_layout.addLayout(lang_layout)
        basic_layout.addWidget(self.whisper_model_container)
        basic_layout.addWidget(self.nemo_model_container)
        basic_group.setLayout(basic_layout)

        # Диаризация
        diarization_group = QGroupBox("👥 ДИАРИЗАЦИЯ СПИКЕРОВ")
        diarization_layout = QVBoxLayout()

        self.diarization_checkbox = QCheckBox("Включить диаризацию спикеров")
        self.diarization_checkbox.setChecked(True)

        # Выбор backend диаризации
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("Метод:"))
        self.diarization_backend_combo = QComboBox()
        self.diarization_backend_combo.addItems([
            "heuristic - Эвристический (без доп. зависимостей)",
            "pyannote - Нейросеть (лучшее качество, нужен HF токен)"
        ])
        # По умолчанию heuristic, если pyannote недоступен
        if not PYANNOTE_AVAILABLE:
            self.diarization_backend_combo.setCurrentIndex(0)
            self.diarization_backend_combo.setItemData(1, 0, Qt.ItemDataRole.UserRole - 1)  # Disable pyannote
        backend_layout.addWidget(self.diarization_backend_combo)
        backend_layout.addStretch()

        # HuggingFace токен для pyannote
        hf_layout = QHBoxLayout()
        hf_layout.addWidget(QLabel("HF Token:"))
        self.hf_token_edit = QTextEdit()
        self.hf_token_edit.setMaximumHeight(30)
        self.hf_token_edit.setPlaceholderText("Токен HuggingFace для pyannote (hf_xxx...)")
        if DEFAULT_HF_TOKEN:
            self.hf_token_edit.setPlainText(DEFAULT_HF_TOKEN)
        hf_layout.addWidget(self.hf_token_edit)

        pyannote_status = "✅ Установлен" if PYANNOTE_AVAILABLE else "❌ Не установлен (pip install pyannote-audio)"
        pyannote_label = QLabel(f"pyannote-audio: {pyannote_status}")
        pyannote_label.setStyleSheet(f"color: {'#56d364' if PYANNOTE_AVAILABLE else '#f87171'}; font-size: 11px;")

        self.show_timestamps_checkbox = QCheckBox("Показывать время каждой реплики")
        self.show_timestamps_checkbox.setChecked(False)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Мин. пауза (сек):"))
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
        self.min_silence_spin.setValue(800)
        params_layout.addWidget(self.min_silence_spin)
        params_layout.addStretch()

        diarization_layout.addWidget(self.diarization_checkbox)
        diarization_layout.addLayout(backend_layout)
        diarization_layout.addLayout(hf_layout)
        diarization_layout.addWidget(pyannote_label)
        diarization_layout.addWidget(self.show_timestamps_checkbox)
        diarization_layout.addLayout(params_layout)
        diarization_group.setLayout(diarization_layout)

        self.diarization_checkbox.stateChanged.connect(self._on_diarization_changed)

        # Информация о системе
        info_group = QGroupBox("💻 Информация о системе")
        info_layout = QVBoxLayout()

        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMaximumHeight(150)
        self._update_system_info()

        info_layout.addWidget(self.system_info)
        info_group.setLayout(info_layout)

        layout.addWidget(basic_group)
        layout.addWidget(diarization_group)
        layout.addWidget(info_group)
        layout.addStretch()

        self.asr_backend_combo.currentIndexChanged.connect(self._on_asr_backend_changed)
        self.nemo_model_combo.currentIndexChanged.connect(self._on_nemo_model_changed)
        self._on_asr_backend_changed()
        self._on_nemo_model_changed()

        return widget

    def _current_asr_backend(self) -> str:
        backend = self.asr_backend_combo.currentData()
        return backend or "whisper"

    def _current_model_label(self) -> str:
        backend = self._current_asr_backend()
        if backend == "nemo":
            model_name = self._resolve_nemo_model_name()
            return f"NeMo: {model_name}"
        return self.model_combo.currentText()

    def _resolve_nemo_model_name(self) -> str:
        model_name = self.nemo_model_combo.currentData()
        if model_name == "custom":
            model_name = self.nemo_model_edit.text().strip()
        return model_name or NEMO_DEFAULT_MODEL

    def _on_asr_backend_changed(self):
        backend = self._current_asr_backend()
        use_whisper = backend == "whisper"
        self.whisper_model_container.setVisible(use_whisper)
        self.nemo_model_container.setVisible(not use_whisper)
        self.download_model_btn.setEnabled(use_whisper)

    def _on_nemo_model_changed(self):
        is_custom = self.nemo_model_combo.currentData() == "custom"
        self.nemo_model_edit.setVisible(is_custom)

    def _create_logs_tab(self):
        """Создание вкладки логов"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_widget = LogWidget()

        controls_layout = QHBoxLayout()
        clear_btn = QPushButton("🗑️ Очистить")
        clear_btn.clicked.connect(self.log_widget.clear)
        export_btn = QPushButton("💾 Экспорт")
        export_btn.clicked.connect(self.export_logs)
        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)
        controls_layout.addWidget(export_btn)

        stats_group = QGroupBox("📊 Статистика последней транскрибации")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlaceholderText("Статистика появится после транскрибации...")
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)

        layout.addWidget(self.log_widget)
        layout.addLayout(controls_layout)
        layout.addWidget(stats_group)

        return widget

    def _create_status_bar(self):
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
            QStatusBar::item { border: none; }
        """)

        label_style = "QLabel { color: #94a3b8; padding: 0 15px; background: transparent; }"

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

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)

    def _create_save_button(self, text, color):
        """Создание кнопки сохранения"""
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
            QPushButton:hover {{ background: {color}dd; }}
            QPushButton:disabled {{ background: #1a1a2e; color: #4a5568; }}
        """)
        return btn

    def _on_diarization_changed(self, state):
        """Обработчик изменения чекбокса диаризации"""
        enabled = state == Qt.CheckState.Checked.value
        self.min_pause_spin.setEnabled(enabled)
        self.min_silence_spin.setEnabled(enabled)
        self.show_timestamps_checkbox.setEnabled(enabled)
        self.log_widget.log(f"Диаризация {'включена' if enabled else 'отключена'}", "INFO")

    def _update_system_info(self):
        """Обновление информации о системе"""
        info = []
        info.append(f"Python: {sys.version.split()[0]}")

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
                info.append("Режим: CPU (int8)")
        else:
            info.append("PyTorch не установлен - CPU режим")

        ffmpeg_found = check_ffmpeg()
        info.append(f"FFmpeg: {'✅ Найден' if ffmpeg_found else '❌ Не найден'}")

        # Путь к моделям
        cache_dir = get_models_cache_dir()
        info.append(f"Путь к моделям: {cache_dir}")
        if cache_dir.exists():
            # Считаем только директории моделей
            model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
            info.append(f"Скачано моделей: {len(model_dirs)}")

        # pyannote статус
        info.append(f"pyannote-audio: {'✅' if PYANNOTE_AVAILABLE else '❌ Не установлен'}")

        info.append("Защита от крашей: активна")
        info.append("Множественные файлы: поддерживается")
        info.append("Drag & Drop: активен")
        info.append(f"Модульная архитектура: v{APP_VERSION}")

        self.system_info.setPlainText("\n".join(info))

    def _update_status(self):
        """Обновление статус бара"""
        try:
            import psutil
            memory = psutil.Process().memory_info().rss / (1024 ** 3)
            self.memory_label.setText(f"RAM: {memory:.1f} GB")
        except:
            pass

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

        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))
        self.queue_label.setText(f"Очередь: {len(self.file_queue)}")

    def select_files(self):
        """Выбор файлов"""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Выберите видео или аудио файлы", "",
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
            if any(item.path == file_path for item in self.file_queue):
                self.log_widget.log(f"Файл уже в очереди: {os.path.basename(file_path)}", "WARNING")
                continue

            file_item = FileQueueItem(file_path)
            self.file_queue.append(file_item)

            list_item = QListWidgetItem(f"📄 {file_item.name} ({file_item.size:.1f} MB)")
            self.file_list_widget.addItem(list_item)

            self.log_widget.log(f"Добавлен в очередь: {file_item.name}", "SUCCESS")
            added_count += 1

        if added_count > 0:
            self.log_widget.log(f"Добавлено файлов: {added_count}", "INFO")
            self.transcribe_btn.setEnabled(True)
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
            if row < len(self.file_queue):
                removed_file = self.file_queue.pop(row)
                self.log_widget.log(f"Удален из очереди: {removed_file.name}", "INFO")

        if not self.file_queue:
            self.transcribe_btn.setEnabled(False)

    def clear_file_queue(self):
        """Очистка всей очереди файлов"""
        if not self.file_queue:
            return

        reply = QMessageBox.question(
            self, "Подтверждение",
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
        if self._current_asr_backend() != "whisper":
            QMessageBox.information(
                self,
                "Информация",
                "Скачивание доступно только для Whisper моделей.\n"
                "NeMo модели загружаются автоматически при первом запуске."
            )
            return

        model_name = self.model_combo.currentText().split(' - ')[0]

        reply = QMessageBox.question(
            self, "Скачивание модели",
            f"Скачать модель {model_name}?\n\nЭто может занять время в зависимости от размера модели.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.download_model_btn.setEnabled(False)
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            self.status_label.show()

            self.model_downloader = ModelDownloader(model_name)
            self.model_downloader.progress_signal.connect(self._on_download_progress)
            self.model_downloader.finished_signal.connect(self._on_download_finished)
            self.model_downloader.log_signal.connect(self.log_widget.log)
            self.model_downloader.start()

    @Slot(int, str)
    def _on_download_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    @Slot(bool, str)
    def _on_download_finished(self, success, message):
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
        asr_backend = self._current_asr_backend()
        nemo_model = self._resolve_nemo_model_name()

        if asr_backend == "nemo" and not nemo_model:
            QMessageBox.warning(self, "Предупреждение", "Укажите имя NeMo модели")
            self.transcribe_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.select_file_btn.setEnabled(True)
            self.clear_queue_btn.setEnabled(True)
            self.remove_selected_btn.setEnabled(True)
            self._on_asr_backend_changed()
            return

        # Определяем backend диаризации
        diar_backend_text = self.diarization_backend_combo.currentText()
        diar_backend = "heuristic" if "heuristic" in diar_backend_text else "pyannote"

        # Получаем HF токен
        hf_token = self.hf_token_edit.toPlainText().strip() or None

        settings = {
            'language': lang,
            'model_size': model,
            'asr_backend': asr_backend,
            'nemo_model': nemo_model,
            'use_diarization': self.diarization_checkbox.isChecked(),
            'diarization_backend': diar_backend,
            'hf_token': hf_token,
            'min_pause': float(self.min_pause_spin.value()),
            'min_silence': int(self.min_silence_spin.value()),
            'max_speakers': 10,
            'show_timestamps': self.show_timestamps_checkbox.isChecked()
        }

        file_paths = [item.path for item in self.file_queue]

        self.log_widget.log("=" * 50, "INFO")
        self.log_widget.log(f"Начало пакетной транскрибации: {len(file_paths)} файлов", "INFO")
        self.log_widget.log(f"Настройки: {json.dumps(settings, ensure_ascii=False)}", "DEBUG")

        try:
            gc.collect()
        except:
            pass

        try:
            self.transcription_thread = TranscriptionWorker(file_paths, settings)
            self.transcription_thread.progress_signal.connect(self._on_progress)
            self.transcription_thread.log_signal.connect(self.log_widget.log)
            self.transcription_thread.finished_signal.connect(self._on_finished)
            self.transcription_thread.segment_signal.connect(self._on_segment)
            self.transcription_thread.stats_signal.connect(self._on_stats)
            self.transcription_thread.file_completed_signal.connect(self._on_file_completed)
            self.transcription_thread.start()

            self.log_widget.log("Рабочий поток запущен", "SUCCESS")
        except Exception as thread_error:
            self.log_widget.log(f"Ошибка запуска потока: {thread_error}", "ERROR")
            self._on_finished(f"Ошибка запуска: {thread_error}")

    def stop_transcription(self):
        """Остановка транскрибации"""
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.log_widget.log("Остановка транскрибации...", "WARNING")

            try:
                self.transcription_thread.stop()
                if not self.transcription_thread.wait(5000):
                    self.log_widget.log("Принудительное завершение потока...", "WARNING")
                    self.transcription_thread.terminate()
                    self.transcription_thread.wait(2000)
                self.log_widget.log("Поток остановлен", "INFO")
            except Exception as e:
                self.log_widget.log(f"Ошибка остановки потока: {e}", "ERROR")
            finally:
                self._on_finished("Транскрибация остановлена пользователем")

    @Slot(int, str)
    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    @Slot(str)
    def _on_segment(self, segment):
        self.live_text.append(segment)
        cursor = self.live_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.live_text.setTextCursor(cursor)

    @Slot(dict)
    def _on_stats(self, stats):
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
    def _on_file_completed(self, file_path, result):
        file_name = os.path.basename(file_path)
        self.log_widget.log(f"Файл обработан: {file_name}", "SUCCESS")

        for i, item in enumerate(self.file_queue):
            if item.path == file_path:
                item.status = "Готово"
                item.result = result
                list_item = self.file_list_widget.item(i)
                if list_item:
                    list_item.setText(f"✅ {item.name} ({item.size:.1f} MB) - Готово")
                break

    @Slot(str)
    def _on_finished(self, result):
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
                if '<div' in result or '<span' in result:
                    self.result_text.setHtml(result)
                    self.transcribed_text = self.result_text.toPlainText()
                else:
                    self.transcribed_text = result
                    self.result_text.setPlainText(result)

                for btn in [self.save_txt_btn, self.save_docx_btn, self.save_json_btn, self.save_all_btn]:
                    btn.setEnabled(True)

                cursor = self.result_text.textCursor()
                cursor.setPosition(0)
                self.result_text.setTextCursor(cursor)

                self.log_widget.log(f"Транскрибация завершена успешно: {len(self.file_queue)} файлов", "SUCCESS")
                self.log_widget.log("=" * 50, "INFO")

            if self.transcription_thread:
                try:
                    if self.transcription_thread.isRunning():
                        self.transcription_thread.wait(1000)
                except:
                    pass
                finally:
                    self.transcription_thread = None

            try:
                gc.collect()
                time.sleep(0.1)
            except:
                pass

            self.log_widget.log("Система готова к новой транскрибации", "INFO")

        except Exception as e:
            self.log_widget.log(f"Ошибка завершения транскрибации: {e}", "ERROR")
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
                self, "Сохранить как текст",
                f"{base_name}_{timestamp}.txt", "Текстовые файлы (*.txt)"
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
                self, "Сохранить как Word",
                f"{base_name}_{timestamp}.docx", "Документы Word (*.docx)"
            )
            if file_path:
                try:
                    from docx import Document
                    doc = Document()
                    doc.add_heading('ПРОФЕССИОНАЛЬНАЯ ТРАНСКРИБАЦИЯ', 0)
                    doc.add_paragraph(f'Файлов обработано: {len(self.file_queue)}')
                    for item in self.file_queue:
                        doc.add_paragraph(f'• {item.name} ({item.size:.1f} MB)')
                    doc.add_paragraph(f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    doc.add_paragraph(f'Программа: {APP_NAME} v{APP_VERSION}')
                    doc.add_paragraph(f'Автор: {AUTHOR}')
                    doc.add_heading('Результат', level=1)

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
                self, "Сохранить как JSON",
                f"{base_name}_{timestamp}.json", "JSON файлы (*.json)"
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
                            'model': self._current_model_label(),
                            'language': self.language_combo.currentText(),
                            'professional_edition': True
                        },
                        'text': self.transcribed_text,
                        'individual_results': {},
                        'statistics': self.stats_text.toPlainText() if self.stats_text.toPlainText() else None
                    }

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
            self, "Выберите папку для сохранения результатов"
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
                txt_path = os.path.join(folder_path, f"{base_name}_{timestamp}.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"ТРАНСКРИБАЦИЯ: {item.name}\n")
                    f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Размер файла: {item.size:.1f} MB\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(item.result)

                saved_count += 1
                self.log_widget.log(f"Сохранен: {txt_path}", "SUCCESS")

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
                self, "Успех",
                f"Результаты сохранены!\n\nСохранено файлов: {saved_count}\nПапка: {folder_path}"
            )

        except Exception as e:
            self.log_widget.log(f"Ошибка сохранения результатов: {e}", "ERROR")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты: {e}")

    def export_logs(self):
        """Экспорт логов"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт логов",
            f"professional_logs_{timestamp}.txt", "Текстовые файлы (*.txt)"
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

    def show_about(self):
        """Показать диалог О программе"""
        AboutDialog(self).exec()

    def closeEvent(self, event):
        """Закрытие приложения"""
        active_threads = []

        if self.transcription_thread and self.transcription_thread.isRunning():
            active_threads.append("транскрибация")

        if self.model_downloader and self.model_downloader.isRunning():
            active_threads.append("скачивание модели")

        if active_threads:
            reply = QMessageBox.question(
                self, 'Подтверждение',
                f'Выполняется: {", ".join(active_threads)}.\nВы уверены, что хотите выйти?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.log_widget.log("Принудительное закрытие программы...", "WARNING")
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
            event.accept()
