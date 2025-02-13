import sys
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QLineEdit, QTextEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QMessageBox, QStyleFactory, QProgressBar, QComboBox, QStackedWidget, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from pydub import AudioSegment
import speech_recognition as sr
import requests
from docx import Document
import whisper  # Для транскрибации через OpenAI Whisper

############################################################
# Функция для скачивания аудио с YouTube с использованием yt-dlp
############################################################

def download_youtube_audio(url, cookie_file, output_dir="downloads"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mp3_file = os.path.join(output_dir, "audio.mp3")
    command = [
         "yt-dlp",
         "--cookies", cookie_file,
         "-f", "bestaudio",
         "--extract-audio",
         "--audio-format", "mp3",
         "-o", mp3_file,
         url
    ]
    print("Скачивание аудио с YouTube...")
    subprocess.run(command, check=True)
    print(f"Аудио сохранено в {mp3_file}")
    return mp3_file

############################################################
# Функция транскрибации с использованием OpenAI Whisper
############################################################

def transcribe_with_whisper(file_path, language="ru"):
    model = whisper.load_model("turbo")
    # Whisper ожидает код языка в виде, например, "ru" или "en" – поэтому берем первые 2 символа из формата ru-RU/en-US
    lang = language.split('-')[0]
    result = model.transcribe(file_path, language=lang)
    return result["text"]

############################################################
# Фоновый поток для асинхронной транскрибации
############################################################

class TranscriptionWorker(QThread):
    progress_signal = Signal(int)      # Прогресс (в %)
    status_signal = Signal(str, str)     # (уровень, сообщение)
    finished_signal = Signal(str)        # Итоговый текст

    def __init__(self, video_file, language="ru-RU", engine="google", is_temp=False, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.language = language
        self.engine = engine            # "google" или "whisper"
        self.is_temp = is_temp          # True, если файл был скачан (временный)
        self._is_running = True
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_audio_file = os.path.join(self.temp_dir.name, "extracted_audio.wav")

    def run(self):
        try:
            self.log("INFO", f"\nОбработка файла: {self.video_file}")
            file_ext = os.path.splitext(self.video_file)[1].lower()
            if file_ext in [".mp4", ".mov", ".avi", ".mkv"]:
                self.log("INFO", "Извлечение аудио из видео...")
                extract_audio(self.video_file, self.output_audio_file)
                self.log("INFO", "Аудио успешно извлечено.")
            elif file_ext == ".mp3":
                if self.engine == "google":
                    self.log("INFO", "Конвертация MP3 в WAV для Google Speech API...")
                    audio = AudioSegment.from_mp3(self.video_file)
                    audio.export(self.output_audio_file, format="wav")
                else:
                    self.log("INFO", "Использование MP3 файла для Whisper...")
                    self.output_audio_file = self.video_file
            else:
                self.log("WARNING", "Неизвестный формат файла, попробуем использовать его напрямую.")
                self.output_audio_file = self.video_file

            # Транскрибация
            if self.engine == "whisper":
                self.log("INFO", "Распознавание речи с использованием OpenAI Whisper...")
                result_text = transcribe_with_whisper(self.output_audio_file, self.language)
                self.progress_signal.emit(100)
            else:
                self.log("INFO", "Распознавание речи с использованием Google Speech API...")
                result_text = self.parallel_transcribe_chunks(self.output_audio_file)

            self.log("INFO", "Финальный текст собран успешно.")
            self.finished_signal.emit(result_text)

        except Exception as e:
            import traceback
            error_msg = f"Ошибка:\n{traceback.format_exc()}"
            self.finished_signal.emit(error_msg)
        finally:

            # Если файл был скачан (временный), удаляем его
            if self.is_temp:
                try:
                    os.remove(self.video_file)
                    self.log("INFO", "Временный MP3 файл удален.")
                except Exception as e:
                    self.log("WARNING", f"Не удалось удалить временный MP3 файл: {str(e)}")

    def parallel_transcribe_chunks(self, file_path):
        try:
            audio = AudioSegment.from_wav(file_path)
            chunk_length = 30 * 1000  # 30 секунд
            chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
            results = {}
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {}
                for i, chunk in enumerate(chunks):
                    if len(chunk) < 1000:
                        self.log("WARNING", f"Чанк {i} слишком короткий и пропущен.")
                        results[i] = "[Чанк слишком короткий]"
                        continue
                    temp_file = os.path.join(self.temp_dir.name, f"chunk_{i}.flac")
                    try:
                        chunk.export(
                            temp_file,
                            format="flac",
                            parameters=["-ac", "1", "-compression_level", "5"]
                        )
                        if not os.path.exists(temp_file):
                            raise RuntimeError(f"Не удалось создать чанк {i}")
                        futures[executor.submit(transcribe_chunk, temp_file, self.language)] = i
                        self.log("DEBUG", f"Чанк {i} экспортирован: {temp_file}")
                    except Exception as e:
                        self.log("ERROR", f"Чанк {i} ошибка экспорта: {str(e)}")
                        results[i] = f"[Ошибка экспорта: {str(e)}]"
                        continue
                completed = 0
                for future in as_completed(futures):
                    if not self._is_running:
                        break
                    chunk_index = futures[future]
                    try:
                        text = future.result()
                        if text is None:
                            text = "[Пустой результат]"
                        results[chunk_index] = text
                        self.log("INFO", f"Чанк {chunk_index} распознан: {text}")
                    except Exception as e:
                        error_msg = f"[Ошибка: {str(e)}]"
                        results[chunk_index] = error_msg
                        self.log("ERROR", f"Чанк {chunk_index} ошибка: {error_msg}")
                    completed += 1
                    self.progress_signal.emit(int((completed / len(chunks)) * 100))
            final_text = "\n".join(results.get(i, "[Не распознан]") for i in range(len(chunks)))
            return final_text
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    def log(self, level, message):
        self.status_signal.emit(level, message)

############################################################
# Утилиты для транскрибации и конвертации
############################################################

def extract_audio(video_file, output_audio_file):
    output_dir = os.path.dirname(output_audio_file)
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_file,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_file
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr.decode('utf-8')}"
        raise RuntimeError(error_msg)
    if os.path.getsize(output_audio_file) < 1024:
         raise RuntimeError("Слишком маленький аудиофайл")

def transcribe_chunk(chunk_file, language="ru-RU"):
    recognizer = sr.Recognizer()
    try:
        if not os.path.exists(chunk_file):
            return f"[Файл {os.path.basename(chunk_file)} не найден]"
        with sr.AudioFile(chunk_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return "[Не удалось распознать речь]"
    except Exception as e:
        import traceback
        print(f"Ошибка в чанке {chunk_file}:\n{traceback.format_exc()}")
        return f"[Ошибка: {str(e)}]"

def send_to_api(transcribed_text, api_key, user_prompt):
    url_endpoint = "https://api.gen-api.ru/api/v1/networks/gpt-4o"
    input_payload = {
        "is_sync": True,
        "model": "gpt-4o-2024-08-06",
        "stream": False,
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 4096,
        "response_format": {"type": "text"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Вот текст, который нужно обработать:\n\n{transcribed_text}\n\n{user_prompt}"
                    }
                ]
            }
        ]
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Bearer {api_key.strip()}"
    }
    response = requests.post(url_endpoint, json=input_payload, headers=headers)
    try:
        result = response.json()
    except:
        return "Ошибка: Невозможно декодировать ответ."
    try:
        return result["response"][0]["message"]["content"]
    except KeyError:
        return f"Ошибка: {result}"

############################################################
# Главное окно приложения
############################################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.transcribed_text = ""
        self.setWindowTitle("Видео-транскрибация и генерация ответов")
        self.resize(900, 650)
        self.log_message_count = 0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.video_file_path = None
        self.transcription_thread = None

        # Группа выбора источника видео (комбобокс + стек виджетов)
        self.source_group = QGroupBox("[Источник видео]")
        layout_source = QVBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItem("Локальный файл", "local")
        self.source_combo.addItem("YouTube", "youtube")
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        layout_source.addWidget(self.source_combo)

        self.source_stack = QStackedWidget()
        # Страница для локального файла
        self.page_local = QWidget()
        layout_local = QHBoxLayout()
        self.select_video_btn = QPushButton("Выбрать видео")
        self.select_video_btn.clicked.connect(self.select_video_file)
        self.local_video_label = QLabel("Видео не выбрано")
        layout_local.addWidget(self.select_video_btn)
        layout_local.addWidget(self.local_video_label)
        self.page_local.setLayout(layout_local)
        # Страница для YouTube
        self.page_youtube = QWidget()
        layout_youtube = QFormLayout()
        self.youtube_url_edit = QLineEdit()
        self.youtube_url_edit.setPlaceholderText("Введите ссылку на YouTube")
        self.cookie_file_edit = QLineEdit()
        self.cookie_file_edit.setPlaceholderText("Путь к файлу cookies")
        self.select_cookie_btn = QPushButton("Выбрать cookie файл")
        self.select_cookie_btn.clicked.connect(self.select_cookie_file)
        cookie_layout = QHBoxLayout()
        cookie_layout.addWidget(self.cookie_file_edit)
        cookie_layout.addWidget(self.select_cookie_btn)
        layout_youtube.addRow("YouTube URL:", self.youtube_url_edit)
        layout_youtube.addRow("Cookie файл:", cookie_layout)
        self.page_youtube.setLayout(layout_youtube)

        self.source_stack.addWidget(self.page_local)
        self.source_stack.addWidget(self.page_youtube)
        layout_source.addWidget(self.source_stack)
        self.source_group.setLayout(layout_source)

        # Группа настроек транскрибации (язык и сервис)
        self.settings_group = QGroupBox("[Настройки транскрибации]")
        settings_layout = QHBoxLayout()
        lang_label = QLabel("Язык распознавания:")
        self.language_combo = QComboBox()
        self.language_combo.addItem("Русский (ru-RU)", "ru-RU")
        self.language_combo.addItem("English (en-US)", "en-US")
        engine_label = QLabel("Сервис:")
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Google Speech API", "google")
        self.engine_combo.addItem("OpenAI Whisper", "whisper")
        settings_layout.addWidget(lang_label)
        settings_layout.addWidget(self.language_combo)
        settings_layout.addSpacing(20)
        settings_layout.addWidget(engine_label)
        settings_layout.addWidget(self.engine_combo)
        settings_layout.addStretch()
        self.settings_group.setLayout(settings_layout)

        # Группа транскрибации
        self.video_group = QGroupBox("[1] Транскрибация")
        self.transcribe_btn = QPushButton("Запуск")
        self.transcribe_btn.clicked.connect(self.transcribe_video)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()

        g_video_layout = QVBoxLayout()
        g_video_layout.addWidget(self.source_group)
        g_video_layout.addWidget(self.settings_group)
        g_video_layout.addWidget(self.transcribe_btn)
        g_video_layout.addWidget(self.progress_bar)
        self.video_group.setLayout(g_video_layout)

        # Группа взаимодействия с ИИ
        self.prompt_group = QGroupBox("[2] Взаимодействие с ИИ")
        self.api_key_label = QLabel("API ключ:")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Введите свой API-ключ...")
        self.prompt_label = QLabel("Промпт:")
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Например, 'Составь краткое резюме...'")
        self.summarize_btn = QPushButton("Сформировать ответ")
        self.summarize_btn.clicked.connect(self.summarize_text)
        self.save_btn = QPushButton("Сохранить результат")
        self.save_btn.clicked.connect(self.save_results)

        g2_layout = QVBoxLayout()
        row_apikey = QHBoxLayout()
        row_apikey.addWidget(self.api_key_label)
        row_apikey.addWidget(self.api_key_edit)
        row_prompt_label = QHBoxLayout()
        row_prompt_label.addWidget(self.prompt_label)
        row_prompt_label.addStretch()
        g2_layout.addLayout(row_apikey)
        g2_layout.addLayout(row_prompt_label)
        g2_layout.addWidget(self.prompt_edit)
        row_summarize_btn = QHBoxLayout()
        row_summarize_btn.addStretch()
        row_summarize_btn.addWidget(self.summarize_btn)
        row_summarize_btn.addWidget(self.save_btn)
        g2_layout.addLayout(row_summarize_btn)
        self.prompt_group.setLayout(g2_layout)

        # Поле вывода логов
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setObjectName("LogOutput")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_group)
        main_layout.addWidget(self.prompt_group)
        main_layout.addWidget(self.output_text, stretch=1)
        central_widget.setLayout(main_layout)

        QApplication.setStyle(QStyleFactory.create("Fusion"))
        self.setStyleSheet(self._build_stylesheet())

    def closeEvent(self, event):
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread._is_running = False
            self.transcription_thread.quit()
            self.transcription_thread.wait(5000)
        event.accept()

    @Slot()
    def on_source_changed(self):
        source = self.source_combo.currentData()
        if source == "local":
            self.source_stack.setCurrentWidget(self.page_local)
        else:
            self.source_stack.setCurrentWidget(self.page_youtube)

    @Slot()
    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Видео файлы (*.mp4 *.mov *.avi *.mkv);;Аудио файлы (*.mp3);;Все файлы (*.*)"
        )
        if file_path:
            self.video_file_path = file_path
            self.local_video_label.setText(os.path.basename(file_path))
        else:
            self.local_video_label.setText("Видео не выбрано")
            self.video_file_path = None

    @Slot()
    def select_cookie_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите cookie файл",
            "",
            "Файлы cookies (*.*)"
        )
        if file_path:
            self.cookie_file_edit.setText(file_path)

    @Slot()
    def transcribe_video(self):
        self.transcribed_text = ""
        source = self.source_combo.currentData()
        is_temp = False

        if source == "local":
            if not self.video_file_path:
                self.show_message("Пожалуйста, выберите видео.")
                return
        else:  # YouTube
            youtube_url = self.youtube_url_edit.text().strip()
            cookie_file = self.cookie_file_edit.text().strip()
            if not youtube_url:
                self.show_message("Пожалуйста, введите ссылку на YouTube.")
                return
            if not cookie_file:
                self.show_message("Пожалуйста, выберите cookie файл.")
                return
            self.output_text.append("<b>Скачивание аудио с YouTube...</b>")
            try:
                self.video_file_path = download_youtube_audio(youtube_url, cookie_file)
                is_temp = True
                self.local_video_label.setText(os.path.basename(self.video_file_path))
                self.output_text.append("<b>Аудио успешно скачано.</b>")
            except Exception as e:
                self.show_message(f"Ошибка при скачивании аудио: {str(e)}")
                return

        self.output_text.append("<b>Запуск процесса транскрибации...</b>")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        selected_language = self.language_combo.currentData()
        selected_engine = self.engine_combo.currentData()
        self.transcription_thread = TranscriptionWorker(
            self.video_file_path,
            language=selected_language,
            engine=selected_engine,
            is_temp=is_temp
        )
        self.transcription_thread.progress_signal.connect(self.on_progress)
        self.transcription_thread.status_signal.connect(self.on_status)
        self.transcription_thread.finished_signal.connect(self.on_transcription_finished)
        self.transcription_thread.start()

    @Slot(int)
    def on_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str, str)
    def on_status(self, level, message):
        if level == "ERROR":
            color = "#FFCCCC"
        elif level == "WARNING":
            color = "#FFF3CD"
        elif level == "DEBUG":
            color = "#E2E3E5"
        else:
            color = "#D0F0C0"
        formatted_message = (
            f'<div style="background-color: {color}; padding: 6px; border-radius: 6px;">'
            f'<b>[{level}]</b> {message}</div>'
        )
        self.output_text.append(formatted_message)
        self.log_message_count += 1

    @Slot(str)
    def on_transcription_finished(self, result_text):
        self.progress_bar.hide()
        self.transcribed_text = result_text
        if result_text.startswith("Ошибка:"):
            self.output_text.append(
                f'<div style="color: red;"><b>{result_text}</b></div>'
            )
        else:
            self.output_text.append(
                f'<div style="background-color: #E9FAEA; padding: 6px; border-radius: 6px;">'
                f'<b>Транскрибированный текст:</b><br>{result_text}</div>'
            )
        self.transcription_thread = None

    @Slot()
    def summarize_text(self):
        api_key = self.api_key_edit.text().strip()
        user_prompt = self.prompt_edit.toPlainText().strip()
        if not api_key:
            self.show_message("Пожалуйста, введите API ключ.")
            return
        if not user_prompt:
            self.show_message("Пожалуйста, введите промпт для генерации ответа.")
            return
        if not self.transcribed_text.strip():
            self.show_message("Сначала выполните транскрибацию, чтобы получить текст.")
            return
        self.output_text.append("<b>Отправка текста в API...</b>")
        final_response = send_to_api(self.transcribed_text, api_key, user_prompt)
        self.output_text.append(
            f'<div style="background-color: #E9FAEA; padding: 6px; border-radius: 6px;">'
            f'<b>Ответ от модели:</b><br>{final_response}</div>'
        )

    def save_results(self):
        if not self.transcribed_text:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            "",
            "Text Files (*.txt);;Word Documents (*.docx)"
        )
        if file_path:
            try:
                if file_path.endswith('.docx'):
                    doc = Document()
                    doc.add_paragraph(self.transcribed_text)
                    doc.save(file_path)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.transcribed_text)
                QMessageBox.information(self, "Успех", "Файл сохранен")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {str(e)}")

    def show_message(self, text):
        QMessageBox.information(self, "Информация", text)

    def _build_stylesheet(self) -> str:
        return """
            QMainWindow {
                background-color: #f5f5f7;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            QGroupBox {
                font-size: 15px;
                color: #333;
                background: #fff;
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 6px 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QLabel {
                font-size: 14px;
                color: #444;
            }
            #VideoLabelPlaceholder {
                color: #999;
                font-style: italic;
            }
            QLineEdit, QTextEdit {
                font-size: 14px;
                background: #ffffff;
                color: #333;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                font-size: 14px;
                background-color: #4285f4;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #3a75d1;
            }
            QPushButton:pressed {
                background-color: #3367b0;
            }
            QProgressBar {
                border: 1px solid #bbb;
                background: #e6e6e6;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background-color: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 #5eb8ff,
                    stop: 1 #4285f4
                );
            }
            #LogOutput {
                background-color: #ffffff;
                color: #333;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 6px;
            }
        """

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
