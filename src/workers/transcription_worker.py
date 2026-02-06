"""
Worker для транскрибации
"""

import os
import gc
import time
import tempfile
from typing import List

from PySide6.QtCore import QThread, Signal

from ..utils import CrashSafeMemoryManager, format_simple_text
from ..core import (
    load_model, extract_audio, transcribe_audio,
    apply_diarization, validate_segment, finalize_transcription,
    load_nemo_model, transcribe_nemo_audio
)


class TranscriptionWorker(QThread):
    """Рабочий поток транскрибации с защитой от крашей"""

    progress_signal = Signal(int, str)
    log_signal = Signal(str, str)
    finished_signal = Signal(str)
    segment_signal = Signal(str)
    stats_signal = Signal(dict)
    file_completed_signal = Signal(str, str)  # file_path, result

    def __init__(self, file_paths: List[str], settings: dict):
        super().__init__()
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.settings = settings
        self.temp_dir = None
        self._is_running = True
        self.start_time = None
        self.current_file_index = 0
        self._plain_text_result = ""

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
                result = self._process_single_file(file_path)

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
            self._cleanup()

    def _process_single_file(self, file_path: str):
        """Обработка одного файла"""
        model = None
        try:
            asr_backend = self.settings.get("asr_backend", "whisper")
            # Создаем временную директорию
            self.temp_dir = tempfile.TemporaryDirectory()
            output_audio = os.path.join(self.temp_dir.name, "audio.wav")

            # Этап 1: Извлечение аудио
            base_progress = (self.current_file_index * 100) // len(self.file_paths)
            step_progress = 100 // len(self.file_paths)

            self.progress_signal.emit(
                base_progress + step_progress * 10 // 100,
                f"[{self.current_file_index+1}/{len(self.file_paths)}] Извлечение аудио..."
            )
            # Используем профиль фильтров из настроек (по умолчанию "soft")
            filter_profile = self.settings.get('audio_filter', 'soft')
            extract_audio(file_path, output_audio, self._log, filter_profile=filter_profile)

            if not self._is_running:
                return None

            # Предварительная очистка памяти перед загрузкой модели
            CrashSafeMemoryManager.safe_gpu_cleanup("before model loading")

            # Этап 2: Загрузка модели
            self.progress_signal.emit(
                base_progress + step_progress * 20 // 100,
                f"[{self.current_file_index+1}/{len(self.file_paths)}] Загрузка модели..."
            )
            if asr_backend == "nemo":
                model = load_nemo_model(self.settings['nemo_model'], self._log)
            else:
                model = load_model(self.settings['model_size'], self._log)

            if not self._is_running:
                if model:
                    del model
                    model = None
                    CrashSafeMemoryManager.safe_gpu_cleanup("after early stop")
                return None

            # Этап 3: Транскрибация
            self.progress_signal.emit(
                base_progress + step_progress * 30 // 100,
                f"[{self.current_file_index+1}/{len(self.file_paths)}] Распознавание речи..."
            )
            if asr_backend == "nemo":
                segments = transcribe_nemo_audio(
                    output_audio, model, self.settings, log_func=self._log
                )
            else:
                segments = transcribe_audio(
                    output_audio, model, self.settings,
                    log_func=self._log,
                    segment_callback=self._on_segment,
                    running_check=self._is_running_check
                )

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

            # Этап 3.5: Финальная постобработка текста
            self.progress_signal.emit(
                base_progress + step_progress * 60 // 100,
                f"[{self.current_file_index+1}/{len(self.file_paths)}] Постобработка текста..."
            )

            # Определяем язык для постобработки
            detected_language = self.settings.get('language', 'ru')
            if detected_language == 'auto':
                detected_language = 'ru'  # Fallback

            # Опции постобработки из настроек
            postprocess_options = {
                'fix_case': self.settings.get('fix_case', True),
                'fix_punctuation': self.settings.get('fix_punctuation', True),
                'convert_numbers': self.settings.get('convert_numbers', True),
                'fix_asr': self.settings.get('fix_asr', True),
                'clean_repetitions': self.settings.get('clean_repetitions', True),
            }

            # Применяем финальную постобработку
            segments = finalize_transcription(
                segments,
                language=detected_language,
                options=postprocess_options
            )

            self.log_signal.emit(f"Постобработка завершена: {len(segments)} сегментов", "INFO")

            # Этап 4: Диаризация или простое форматирование
            formatted_text = ""
            use_diarization = self.settings.get('use_diarization')
            if asr_backend == "nemo" and use_diarization:
                self.log_signal.emit(
                    "NeMo не предоставляет таймкоды сегментов - диаризация отключена",
                    "WARNING"
                )
                use_diarization = False

            if use_diarization:
                self.progress_signal.emit(
                    base_progress + step_progress * 70 // 100,
                    f"[{self.current_file_index+1}/{len(self.file_paths)}] Диаризация..."
                )
                # Передаём audio_path для pyannote диаризации
                formatted_text = apply_diarization(
                    segments, self.settings,
                    log_func=self._log,
                    running_check=self._is_running_check,
                    audio_path=output_audio  # Путь к извлечённому аудио для pyannote
                )
            else:
                formatted_text = format_simple_text(segments, validate_segment)

            # Очистка памяти после диаризации
            CrashSafeMemoryManager.safe_gpu_cleanup("after diarization")

            if not formatted_text or formatted_text.strip() == "":
                formatted_text = "Транскрибация завершена, но результат пуст. Проверьте аудио файл."

            # Статистика
            try:
                self._send_statistics(segments, formatted_text)
            except Exception as stats_error:
                self.log_signal.emit(f"Ошибка статистики: {stats_error}", "WARNING")

            self.progress_signal.emit(
                base_progress + step_progress,
                f"[{self.current_file_index+1}/{len(self.file_paths)}] Готово"
            )

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

    def _log(self, message: str, level: str = "INFO"):
        """Отправка лога"""
        self.log_signal.emit(message, level)

    def _on_segment(self, segment: str):
        """Callback для сегментов"""
        self.segment_signal.emit(segment)

    def _is_running_check(self):
        """Проверка, продолжать ли работу"""
        return self._is_running

    def _send_statistics(self, segments, text):
        """Отправка статистики"""
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

    def stop(self):
        """Остановка процесса"""
        self._is_running = False
        self.log_signal.emit("Получен сигнал остановки...", "WARNING")

        # Даем время на корректное завершение
        try:
            if self.isRunning():
                self.quit()
                if not self.wait(3000):
                    self.log_signal.emit("Принудительное завершение...", "WARNING")
                    self.terminate()
                    self.wait(1000)
        except Exception as e:
            self.log_signal.emit(f"Ошибка остановки: {e}", "ERROR")

    def _cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.temp_dir:
                self.temp_dir.cleanup()
                self.log_signal.emit("Временные файлы удалены", "DEBUG")
        except Exception as e:
            self.log_signal.emit(f"Предупреждение при удалении временных файлов: {e}", "WARNING")

        # Мягкая финальная очистка памяти
        try:
            gc.collect()
        except:
            pass
