# Audio/Video Transcription Pro - Docker Image
#
# Сборка:
#   docker build -t transcription-pro .
#
# Запуск (с GUI через X11):
#   Linux:   docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/data:/app/data transcription-pro
#   Windows: docker run -e DISPLAY=host.docker.internal:0 -v ${PWD}/data:/app/data transcription-pro
#
# Запуск без GUI (только CLI, если будет добавлен):
#   docker run -v $(pwd)/data:/app/data transcription-pro python -c "from src.core import transcribe_audio; ..."

# Базовый образ с Python
FROM python:3.11-slim

# Метаданные
LABEL maintainer="Lebedev Nikolay"
LABEL description="Professional Audio/Video Transcription with Speaker Diarization"
LABEL version="7.0"

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_QPA_PLATFORM=xcb

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libegl1 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/
COPY main.py .

# Создаем директорию для данных
RUN mkdir -p /app/data /app/models

# Volumes для данных и моделей
VOLUME ["/app/data", "/app/models"]

# Точка входа
CMD ["python", "main.py"]
