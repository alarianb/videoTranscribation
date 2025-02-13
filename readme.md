**Видео-транскрибация и генерация ответов**

Это приложение на Python предоставляет графический интерфейс для транскрибации видео или аудио с использованием двух разных сервисов распознавания речи (Google Speech API и OpenAI Whisper). Кроме того, оно позволяет отправлять полученный текст в нейросети через API gen-api.ru.

---

**Функциональные возможности**

* **Выбор источника файла:**
  * **Локальный файл:** Загрузка видео (форматы .mp4, .mov, .avi, .mkv) или аудио (формат .mp3) с диска.
  * **YouTube:**  Загрузка аудио с YouTube с использованием утилиты [yt-dlp](https://github.com/yt-dlp/yt-dlp). Для скачивания используется cookie-файл для авторизации.
* **Настройки транскрибации:**
  * **Язык распознавания:** Выбор языка (например, Русский ru-RU или Английский en-US).
  * **Сервис распознавания:**
    * **Google Speech API:** Использует библиотеку speech\_recognition с параллельной        обработкой аудио чанками.
    * **OpenAI Whisper:** Использует библиотеку [Whisper](https://github.com/openai/whisper) для        распознавания речи.
* **Интерактивное взаимодействие с ИИ:**

Отправка транскрибированного текста в нейросети через API gen-api.ru.

* **Сохранение результата:**
  * Возможность сохранения финального транскрибированного текста в виде текстового файла или документа Word (DOCX).

---

**Установка и зависимости**

**Необходимые библиотеки Python**

Убедитесь, что у вас установлен Python 3.7 или выше. Для работы приложения потребуются следующие пакеты:

* [PySide6](https://pypi.org/project/PySide6/) – для      создания графического интерфейса.
* [pydub](https://github.com/jiaaro/pydub) – для      работы с аудиофайлами.
* [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) – для распознавания речи (Google Speech API).
* [whisper](https://github.com/openai/whisper) – для      транскрибации через OpenAI Whisper.
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) – для      скачивания аудио с YouTube.
* [python-docx](https://python-docx.readthedocs.io/) –      для создания и сохранения DOCX файлов.
* [requests](https://pypi.org/project/requests/) – для      отправки HTTP-запросов к API.

**Установка зависимостей**

Вы можете установить все необходимые зависимости с помощью pip:

```
pip install PySide6 pydub SpeechRecognition python-docx requests
pip install yt-dlppip install git+https://github.com/openai/whisper.git
```

**Примечание:**

* Для работы с pydub требуется наличие FFmpeg, который должен быть установлен и добавлен в PATH вашей системы.
* Убедитесь, что утилита yt-dlp установлена и доступна из командной строки.

---

**Запуск приложения**

1. Сохраните код в файл, например, main.py.
2. Запустите приложение командой:

python main.py

3. В  графическом интерфейсе выберите источник файла:
   * **Локальный файл:** нажмите кнопку для выбора файла.
   * **YouTube:** введите URL YouTube-видео и укажите путь к cookie-файлу (если требуется).
4. Выберите настройки транскрибации (язык и сервис распознавания).
5. Нажмите кнопку **"Запуск"** для начала транскрибации.
6. После завершения транскрибации вы сможете отправить полученный текст в API для генерации ответа или сохранить результат на диск.

---

**Дополнительные рекомендации**

* **Cookie файл для YouTube:**
  Если вы используете источник YouTube, вам потребуется cookie-файл для авторизации. Обычно его можно получить, экспортировав cookies из браузера.
* **Модель Whisper:**
  В коде используется модель turbo (можно заменить на другие модели, такие как base, small, medium или large в зависимости от ваших потребностей и вычислительных возможностей).
* **Удаление временных файлов:**
  Если аудио скачано с YouTube, после транскрибации временный MP3-файл автоматически удаляется.
