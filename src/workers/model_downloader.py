"""
Worker для скачивания моделей
"""

from PySide6.QtCore import QThread, Signal

from ..core import download_model, get_models_cache_dir


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
            success = download_model(
                self.model_name,
                progress_callback=self._on_progress,
                log_func=self._on_log
            )

            if success:
                self.finished_signal.emit(True, "Модель успешно загружена")
            else:
                self.finished_signal.emit(False, "Не удалось загрузить модель")

        except Exception as e:
            self.log_signal.emit(f"Ошибка загрузки модели: {str(e)}", "ERROR")
            self.finished_signal.emit(False, str(e))

    def _on_progress(self, value, message):
        if not self.should_stop:
            self.progress_signal.emit(value, message)

    def _on_log(self, message, level):
        self.log_signal.emit(message, level)

    def stop(self):
        self.should_stop = True
        self.quit()
        self.wait()

    @staticmethod
    def get_models_cache_dir():
        """Для обратной совместимости"""
        return get_models_cache_dir()
