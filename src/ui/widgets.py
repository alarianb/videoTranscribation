"""
UI виджеты
"""

import os
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QTextBrowser, QListWidget, QListWidgetItem,
    QVBoxLayout, QLabel, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor, QDragEnterEvent, QDropEvent

from ..utils import SUPPORTED_FORMATS


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
        self.file_list.setAcceptDrops(False)
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
                    if ext in SUPPORTED_FORMATS:
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
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in SUPPORTED_FORMATS:
                        files.append(file_path)

            if files:
                self.files_dropped.emit(files)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
