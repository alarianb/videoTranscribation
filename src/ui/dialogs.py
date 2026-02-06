"""
Диалоговые окна
"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt

from ..utils import APP_NAME, APP_VERSION, AUTHOR


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
            "спикеров на базе Whisper или NeMo\n\n"
            "✨ Поддержка множественных файлов\n"
            "✨ Drag & Drop интерфейс\n"
            "✨ Пакетная обработка\n"
            "✨ Модульная архитектура"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 11px; color: #8b949e; margin: 15px;")

        # Технологии
        tech = QLabel("Использует: OpenAI Whisper, NVIDIA NeMo, PyAnnote, PySide6, PyTorch")
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
