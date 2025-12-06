# Отчёт по анализу качества ASR в videoTranscribation

**Дата анализа:** 2025-12-06
**Версия приложения:** 5.4-PROFESSIONAL
**Анализируемые файлы:** `videoTranscribition.py`, `readme.md`

---

## 1. Обзор текущего пайплайна распознавания

Текущий pipeline реализован в `videoTranscribition.py` и работает следующим образом:

1. **Извлечение аудио** (строки 625-645): FFmpeg извлекает аудио из видео/аудио файла с параметрами `-ar 16000 -ac 1 -acodec pcm_s16le` (16кГц, моно, PCM 16-bit). Применяются **агрессивные фильтры**: `highpass=f=200,lowpass=f=3000`.

2. **Предобработка**: Нормализация громкости **отсутствует**. Обрезка тишины **отсутствует** на уровне FFmpeg. Используется только полосовая фильтрация 200-3000 Гц.

3. **Загрузка модели** (строки 667-713): `WhisperModel` из faster-whisper, автоматический выбор device (cuda/cpu), compute_type (float16/int8).

4. **Транскрибация** (строки 715-781): `model.transcribe()` с параметрами:
   - `beam_size=5`, `best_of=5`, `temperature=0.0`, `patience=1`
   - `vad_filter=True` с кастомными `vad_parameters`
   - `condition_on_previous_text=False`
   - `word_timestamps=True`

5. **Обработка длинных файлов**: **Специальной логики нет**. Файл целиком передаётся в faster-whisper, который внутренне сегментирует аудио.

6. **Постобработка текста** (строки 783-799): Минимальная — удаление повторяющихся символов/слов и мусорных символов. Восстановление пунктуации/регистра **отсутствует**.

7. **Диаризация** (строки 801-875): Примитивная эвристика — смена спикера при паузе > `min_pause` секунд (по умолчанию 2.0).

---

## 2. Анализ влияния каждого этапа на качество

### 2.1 Извлечение и предобработка аудио

**Файл:** `videoTranscribition.py:631-637`

```python
cmd = [
    ffmpeg_exe, "-y",
    "-i", input_path,
    "-vn", "-acodec", "pcm_s16le",
    "-ar", "16000", "-ac", "1",
    "-af", "highpass=f=200,lowpass=f=3000",
    output_path
]
```

| Аспект | Оценка | Влияние на качество |
|--------|--------|---------------------|
| Частота дискретизации 16kHz | Хорошо | Оптимально для Whisper |
| Моно | Хорошо | Снижает сложность без потери качества |
| PCM 16-bit | Хорошо | Без потерь качества |
| **highpass=200Hz** | **Проблема** | Отрезает низкие частоты речи (мужские голоса — от 85 Гц) |
| **lowpass=3000Hz** | **Критическая проблема** | Отрезает ~50% речевого спектра (согласные звуки до 8-12 кГц) |
| Нормализация громкости | **Отсутствует** | Тихие записи распознаются хуже |

**Проявление:** Ухудшение распознавания шипящих/свистящих звуков (с, ш, щ, ч, ф), искажение мужских голосов, пропуск слов на тихих участках.

### 2.2 VAD и сегментация

**Файл:** `videoTranscribition.py:728-735`

```python
vad_filter=True,
vad_parameters=dict(
    threshold=0.5,
    min_speech_duration_ms=250,
    max_speech_duration_s=float('inf'),
    min_silence_duration_ms=self.settings.get('min_silence', 1000),
    speech_pad_ms=400
)
```

| Параметр | Значение | Оценка |
|----------|----------|--------|
| threshold=0.5 | Средний | Может пропускать тихую речь |
| min_speech_duration_ms=250 | Короткий | Может пропускать короткие слова |
| max_speech_duration_s=inf | Без ограничения | Риск галлюцинаций на очень длинных сегментах |
| min_silence_duration_ms=1000 | Настраиваемый | По умолчанию слишком длинный |
| speech_pad_ms=400 | Хорошо | Достаточный отступ |

**Проявление:** Пропуск тихих междометий, коротких слов, возможные галлюцинации на сегментах > 30 секунд.

### 2.3 Настройки Whisper/faster-whisper

**Файл:** `videoTranscribition.py:718-740`

```python
segments, info = model.transcribe(
    audio_path,
    language=...,
    task="transcribe",
    beam_size=5,
    best_of=5,
    patience=1,
    temperature=0.0,
    initial_prompt="Это транскрипция на русском языке." if lang == 'ru' else None,
    word_timestamps=True,
    vad_filter=True,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6
)
```

| Параметр | Оценка | Комментарий |
|----------|--------|-------------|
| beam_size=5 | Хорошо | Разумный баланс качества/скорости |
| best_of=5 | Хорошо | Увеличивает шанс лучшего варианта |
| temperature=0.0 | Осторожно | Детерминистично, но может зацикливаться |
| patience=1 | Норма | Стандартное значение |
| condition_on_previous_text=False | Хорошо | Предотвращает накопление ошибок |
| initial_prompt | **Проблема** | Слишком короткий и простой prompt |
| log_prob_threshold=-1.0 | **Проблема** | Слишком мягкий, пропускает галлюцинации |
| no_speech_threshold=0.6 | Норма | Стандартное значение |
| compression_ratio_threshold=2.4 | Хорошо | Фильтрует повторяющийся текст |

**Проявление:** Модель может «зацикливаться» при temperature=0 без fallback, слабый prompt не даёт контекста для специфичной лексики.

### 2.4 Длинные файлы и контекст

**Анализ:** Специальной логики для длинных файлов **нет**. Аудио полностью передаётся в faster-whisper, который внутренне делит на 30-секундные окна.

| Аспект | Статус | Риск |
|--------|--------|------|
| Ручная сегментация с overlap | Отсутствует | Потеря слов на границах |
| Сохранение контекста между чанками | Отсутствует | Разрыв смысла |
| Обработка очень длинных файлов (> 1 часа) | Нет оптимизации | OOM на GPU, деградация качества |

**Проявление:** На границах внутренних сегментов могут теряться слова, нет межсегментного контекста.

### 2.5 Постобработка текста

**Файл:** `videoTranscribition.py:783-799`

```python
def clean_text_safely(self, text):
    # Удаляем повторения
    text = re.sub(r'([а-яА-Яa-zA-Z])\1{3,}', r'\1\1', text)
    text = re.sub(r'(\b\w{1,3}\b)[\s-]*(?:\1[\s-]*){3,}', r'\1', text)
    # Удаляем мусорные символы
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\"\']+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

| Функция | Статус | Комментарий |
|---------|--------|-------------|
| Удаление повторов символов | Есть | Помогает с артефактами |
| Удаление мусорных символов | Есть | Минимальная чистка |
| Восстановление заглавных букв | **Отсутствует** | Текст выглядит «сырым» |
| Восстановление пунктуации | **Отсутствует** | Плохо читается |
| Нормализация чисел/сокращений | **Отсутствует** | «двадцать два» вместо «22» |
| Исправление типичных ошибок ASR | **Отсутствует** | Нет словарных эвристик |

**Проявление:** Текст без заглавных букв в начале предложений, «рваная» пунктуация, числа записаны словами.

### 2.6 Диаризация и сегменты по спикерам

**Файл:** `videoTranscribition.py:801-875`

```python
min_pause = float(self.settings.get('min_pause', 2.0))
# ...
if last_end > 0 and start_time - last_end > min_pause:
    current_speaker = 2 if current_speaker == 1 else 1
```

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| Метод | Примитивный | Только по паузам, не по голосу |
| Точность | Низкая | Работает только на явных диалогах с паузами |
| Максимум спикеров | 2 | Не подходит для групповых разговоров |
| Влияние на семантику | Нейтральное | Не ломает текст, но и не помогает |

**Проявление:** Смешивание спикеров при быстрых перебиваниях, неправильная атрибуция реплик, невозможность разделить > 2 говорящих.

---

## 3. Слабые места (только качество распознавания)

### 3.1 Критические проблемы

| # | Где | Что не так | Проявление у пользователя |
|---|-----|------------|--------------------------|
| 1 | `videoTranscribition.py:636` | **lowpass=3000Hz отрезает верхние частоты речи** | Плохое распознавание шипящих (ш, щ, ч), свистящих (с, з), глухих (ф, х). «Саша» -> «Ааа», «щётка» -> «отка» |
| 2 | `videoTranscribition.py:636` | **highpass=200Hz отрезает низкие частоты** | Искажение мужских голосов, особенно басов. Потеря интонационной информации |
| 3 | `videoTranscribition.py:636` | **Отсутствует нормализация громкости** | Тихие записи дают много пропусков и галлюцинаций. Записи с перепадами громкости — нестабильное качество |

### 3.2 Серьёзные проблемы

| # | Где | Что не так | Проявление у пользователя |
|---|-----|------------|--------------------------|
| 4 | `videoTranscribition.py:725-726` | **temperature=0.0 без fallback на более высокую** | При сложных аудио модель может зациклиться или дать некорректный вывод |
| 5 | `videoTranscribition.py:726` | **Примитивный initial_prompt** | Нет контекста для специфичной лексики (термины, имена). Слова «PyTorch», «Kubernetes» часто распознаются неверно |
| 6 | `videoTranscribition.py:732` | **max_speech_duration_s=inf** | На очень длинных непрерывных сегментах возможны галлюцинации и повторы |
| 7 | `videoTranscribition.py:738` | **log_prob_threshold=-1.0 слишком мягкий** | Не отфильтровываются галлюцинации с низкой уверенностью |

### 3.3 Средние проблемы

| # | Где | Что не так | Проявление у пользователя |
|---|-----|------------|--------------------------|
| 8 | `videoTranscribition.py:783-799` | **Нет восстановления пунктуации и регистра** | Текст плохо читается, выглядит «сырым», нужна ручная правка |
| 9 | `videoTranscribition.py:839-840` | **Примитивная диаризация по паузам** | Смешивание реплик при быстрых перебиваниях, только 2 спикера |
| 10 | `videoTranscribition.py:730` | **threshold=0.5 в VAD** | Может пропускать тихую речь, шёпот, фоновые комментарии |

### 3.4 Минорные проблемы

| # | Где | Что не так | Проявление у пользователя |
|---|-----|------------|--------------------------|
| 11 | `videoTranscribition.py:783-799` | **Нет нормализации числительных** | «двадцать два» вместо «22», «две тысячи двадцать пятый» |
| 12 | Нет файла | **Нет словаря для исправления типичных ошибок** | Нет автозамены «ща» -> «сейчас», «чё» -> «что» |

---

## 4. Детальные предложения по улучшению

### 4.1 Исправление фильтров FFmpeg (Критический приоритет)

**Проблема:** `highpass=f=200,lowpass=f=3000` — катастрофически узкий диапазон.

**Решение:**

```python
# videoTranscribition.py:631-638
# БЫЛО:
cmd = [
    ffmpeg_exe, "-y",
    "-i", input_path,
    "-vn", "-acodec", "pcm_s16le",
    "-ar", "16000", "-ac", "1",
    "-af", "highpass=f=200,lowpass=f=3000",
    output_path
]

# СТАЛО (рекомендуется):
cmd = [
    ffmpeg_exe, "-y",
    "-i", input_path,
    "-vn", "-acodec", "pcm_s16le",
    "-ar", "16000", "-ac", "1",
    "-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:LRA=11:TP=-1.5",
    output_path
]
```

**Что изменилось:**
- `highpass=f=80` — сохраняет низкие частоты мужских голосов (фундаментальная частота от 85 Гц)
- `lowpass=f=8000` — сохраняет высокие частоты (Nyquist для 16kHz = 8kHz), полный спектр согласных звуков
- `loudnorm` — нормализация громкости по EBU R128, делает тихие записи громче

**Выигрыш для пользователя:**
- Значительно лучшее распознавание шипящих/свистящих звуков
- Корректное распознавание мужских голосов
- Стабильное качество на записях с разной громкостью

### 4.2 Улучшение настроек temperature с fallback

**Проблема:** `temperature=0.0` без fallback может приводить к зацикливанию.

**Решение:**

```python
# videoTranscribition.py:718-740
# Вместо фиксированного temperature=0.0, использовать кортеж с fallback:

segments, info = model.transcribe(
    audio_path,
    language=self.settings['language'] if self.settings['language'] != 'auto' else None,
    task="transcribe",
    beam_size=5,
    best_of=5,
    patience=1.0,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Fallback температуры
    initial_prompt=self._get_enhanced_prompt(),   # Улучшенный prompt
    word_timestamps=True,
    vad_filter=True,
    vad_parameters=dict(
        threshold=0.35,                           # Ниже для тихой речи
        min_speech_duration_ms=200,               # Короче для коротких слов
        max_speech_duration_s=30.0,               # Ограничение против галлюцинаций
        min_silence_duration_ms=self.settings.get('min_silence', 500),
        speech_pad_ms=400
    ),
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-0.5,                      # Строже для фильтрации галлюцинаций
    no_speech_threshold=0.6
)

def _get_enhanced_prompt(self):
    """Генерация расширенного initial_prompt"""
    lang = self.settings['language']

    if lang == 'ru':
        return (
            "Это профессиональная транскрипция на русском языке. "
            "Текст содержит чёткую речь. Используется правильная пунктуация "
            "и орфография. Числа записываются цифрами."
        )
    elif lang == 'en':
        return (
            "This is a professional transcription in English. "
            "The text contains clear speech with proper punctuation "
            "and spelling. Numbers are written as digits."
        )
    return None
```

**Выигрыш для пользователя:**
- Модель не зацикливается на сложных аудио
- Лучший контекст для распознавания
- Меньше галлюцинаций благодаря строгому log_prob_threshold
- Лучше распознаётся тихая речь (threshold=0.35)

### 4.3 Добавление постобработки текста

**Проблема:** Минимальная постобработка, текст плохо читается.

**Решение:** Добавить улучшенную функцию постобработки:

```python
# Добавить после clean_text_safely (строка ~800)

def postprocess_text(self, text: str, language: str = 'ru') -> str:
    """Улучшенная постобработка текста для читаемости"""
    if not text:
        return text

    # 1. Базовая очистка (существующий код)
    text = self.clean_text_safely(text)

    # 2. Восстановление заглавных букв в начале предложений
    sentences = re.split(r'([.!?]\s+)', text)
    result = []
    for i, part in enumerate(sentences):
        if i % 2 == 0 and part:  # Это предложение, не разделитель
            part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
        result.append(part)
    text = ''.join(result)

    # 3. Заглавная буква в начале текста
    if text:
        text = text[0].upper() + text[1:]

    # 4. Нормализация пробелов перед знаками препинания
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])([а-яА-Яa-zA-Z])', r'\1 \2', text)

    # 5. Типичные исправления для русского языка
    if language == 'ru':
        replacements = {
            r'\bща\b': 'сейчас',
            r'\bчё\b': 'что',
            r'\bнету\b': 'нет',
            r'\bваще\b': 'вообще',
            r'\bкороч\b': 'короче',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 6. Удаление двойных пробелов
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()
```

**Выигрыш для пользователя:**
- Текст выглядит профессионально с правильными заглавными буквами
- Корректная пунктуация
- Исправлены типичные разговорные сокращения

### 4.4 Ограничение длины сегментов для борьбы с галлюцинациями

**Проблема:** `max_speech_duration_s=inf` позволяет очень длинные сегменты.

**Решение:** Уже включено в пункт 4.2:
```python
max_speech_duration_s=30.0,  # Ограничение длины сегмента
```

**Выигрыш для пользователя:**
- Меньше галлюцинаций и повторов на длинных непрерывных участках речи

### 4.5 Пример оптимального пайплайна

```python
# Полный оптимизированный пайплайн

def extract_audio_optimized(self, input_path, output_path):
    """Оптимизированное извлечение аудио"""
    ffmpeg_exe = self.find_ffmpeg()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg не найден!")

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        # Щадящие фильтры + нормализация громкости
        "-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:LRA=11:TP=-1.5",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка FFmpeg: {result.stderr}")


def transcribe_audio_optimized(self, audio_path, model):
    """Оптимизированная транскрибация"""

    segments, info = model.transcribe(
        audio_path,
        language=self.settings['language'] if self.settings['language'] != 'auto' else None,
        task="transcribe",

        # Декодирование
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Fallback

        # Контекст
        initial_prompt=self._get_enhanced_prompt(),
        condition_on_previous_text=False,  # Безопасно для длинных файлов

        # VAD
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.35,                  # Чувствительнее к тихой речи
            min_speech_duration_ms=200,      # Короткие слова
            max_speech_duration_s=30.0,      # Против галлюцинаций
            min_silence_duration_ms=500,     # Разумный минимум паузы
            speech_pad_ms=400                # Отступы
        ),

        # Фильтрация галлюцинаций
        compression_ratio_threshold=2.4,
        log_prob_threshold=-0.5,             # Строже!
        no_speech_threshold=0.6,

        # Таймстампы
        word_timestamps=True,
    )

    segments_list = []
    for segment in segments:
        text = self.postprocess_text(segment.text.strip(), self.settings['language'])
        if text and len(text) > 2:
            segments_list.append({
                'start': float(segment.start),
                'end': float(segment.end),
                'text': text
            })

    return segments_list
```

---

## 5. Приоритизированный список улучшений

### Высокий приоритет (критическое влияние на качество)

| # | Улучшение | Файл:строка | Ожидаемый эффект |
|---|-----------|-------------|------------------|
| 1 | **Изменить lowpass с 3000 на 8000 Гц** | `videoTranscribition.py:636` | +20-30% точности на шипящих/свистящих |
| 2 | **Изменить highpass с 200 на 80 Гц** | `videoTranscribition.py:636` | Улучшение распознавания мужских голосов |
| 3 | **Добавить loudnorm** | `videoTranscribition.py:636` | Стабильное качество на тихих записях |
| 4 | **Добавить temperature fallback** | `videoTranscribition.py:725` | Устранение зацикливания модели |
| 5 | **Установить max_speech_duration_s=30** | `videoTranscribition.py:732` | Меньше галлюцинаций на длинных сегментах |
| 6 | **Ужесточить log_prob_threshold до -0.5** | `videoTranscribition.py:738` | Фильтрация галлюцинаций низкой уверенности |

### Средний приоритет (улучшение читаемости и точности)

| # | Улучшение | Файл:строка | Ожидаемый эффект |
|---|-----------|-------------|------------------|
| 7 | **Снизить VAD threshold до 0.35** | `videoTranscribition.py:730` | Лучше распознаётся тихая речь |
| 8 | **Улучшить initial_prompt** | `videoTranscribition.py:726` | Контекст для специфичной лексики |
| 9 | **Добавить восстановление заглавных букв** | `videoTranscribition.py:783` | Профессиональный вид текста |
| 10 | **Добавить нормализацию пунктуации** | `videoTranscribition.py:783` | Лучшая читаемость |

### Низкий приоритет (nice-to-have)

| # | Улучшение | Файл:строка | Ожидаемый эффект |
|---|-----------|-------------|------------------|
| 11 | **Добавить словарь исправлений разговорных форм** | новая функция | «ща» -> «сейчас» |
| 12 | **Улучшить диаризацию** (pyannote или аналог) | `videoTranscribition.py:801` | Реальное разделение спикеров по голосу |
| 13 | **Добавить нормализацию числительных** | новая функция | «двадцать два» -> «22» (опционально) |
| 14 | **Добавить UI для настройки параметров Whisper** | UI код | Гибкость для продвинутых пользователей |

---

## Итоговые рекомендации

**Минимальный набор исправлений для значительного улучшения качества:**

1. Заменить строку 636:
```python
# БЫЛО:
"-af", "highpass=f=200,lowpass=f=3000",

# СТАЛО:
"-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:LRA=11:TP=-1.5",
```

2. Изменить параметры в строках 722-740:
```python
temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # вместо 0.0
max_speech_duration_s=30.0,                   # вместо inf
threshold=0.35,                               # вместо 0.5
log_prob_threshold=-0.5,                      # вместо -1.0
```

Эти 4 изменения дадут **наибольший прирост качества при минимальных усилиях**.

---

## Ссылки и ресурсы

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper Best Practices](https://cookbook.openai.com/examples/whisper_prompting_guide)
- [Silero VAD](https://github.com/snakers4/silero-vad) - используется в faster-whisper
- [FFmpeg loudnorm filter](https://ffmpeg.org/ffmpeg-filters.html#loudnorm-1)
