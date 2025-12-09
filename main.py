#!/usr/bin/env python3
"""
Audio/Video Transcription Pro - Главная точка входа

Запуск:
    python main.py

Сборка exe:
    python build.py

Сборка Docker:
    docker build -t transcription-pro .
"""

import sys

from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory

from src.utils import APP_NAME, APP_VERSION, AUTHOR, FASTER_WHISPER_AVAILABLE
from src.ui import MainWindow
from src.ui.dialogs import AboutDialog


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

        # Создаем главное окно
        window = MainWindow()

        # Добавляем меню
        menubar = window.menuBar()
        help_menu = menubar.addMenu("Помощь")
        about_action = help_menu.addAction("О программе")
        about_action.triggered.connect(window.show_about)

        window.show()

        # Стартовые логи
        window.log_widget.log("=" * 50, "INFO")
        window.log_widget.log(f"{APP_NAME} v{APP_VERSION} запущен", "SUCCESS")
        window.log_widget.log(f"Автор: {AUTHOR}", "INFO")
        window.log_widget.log("✨ Модульная архитектура v7.0", "INFO")
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
