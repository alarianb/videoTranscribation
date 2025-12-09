#!/usr/bin/env python3
"""
Скрипт сборки exe для Audio/Video Transcription Pro

Использование:
    python build.py          # Сборка в один файл
    python build.py --dir    # Сборка в папку (быстрее запуск)
    python build.py --clean  # Очистка артефактов сборки

Требования:
    pip install pyinstaller

После сборки:
    - Exe будет в папке dist/
    - Не забудьте положить ffmpeg.exe рядом с exe (для Windows)
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def clean_build_artifacts():
    """Удаление артефактов предыдущей сборки"""
    artifacts = ['build', 'dist', '*.spec']

    for artifact in artifacts:
        if '*' in artifact:
            for f in Path('.').glob(artifact):
                print(f"Удаление: {f}")
                f.unlink()
        else:
            path = Path(artifact)
            if path.exists():
                print(f"Удаление: {path}")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    print("✅ Очистка завершена")


def build_exe(onefile=True):
    """Сборка exe с PyInstaller"""

    # Проверяем PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller версия: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller не установлен!")
        print("Установите: pip install pyinstaller")
        sys.exit(1)

    # Параметры сборки
    app_name = "TranscriptionPro"

    # Базовые аргументы
    args = [
        'main.py',
        f'--name={app_name}',
        '--windowed',  # Без консоли
        '--noconfirm',  # Перезаписывать без вопросов
        '--clean',  # Очищать временные файлы
    ]

    # Один файл или папка
    if onefile:
        args.append('--onefile')
    else:
        args.append('--onedir')

    # Добавляем данные
    args.extend([
        '--add-data=src:src',
    ])

    # Скрытые импорты для PySide6
    hidden_imports = [
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        'faster_whisper',
        'ctranslate2',
        'huggingface_hub',
    ]

    for hi in hidden_imports:
        args.append(f'--hidden-import={hi}')

    # Исключения (уменьшаем размер)
    excludes = [
        'tkinter',
        'matplotlib',
        'numpy.testing',
        'PIL',
    ]

    for exc in excludes:
        args.append(f'--exclude-module={exc}')

    # Иконка (если есть)
    icon_path = Path('icon.ico')
    if icon_path.exists():
        args.append(f'--icon={icon_path}')

    # Запускаем PyInstaller
    print("\n🔨 Запуск сборки...")
    print(f"Команда: pyinstaller {' '.join(args)}\n")

    try:
        subprocess.run(['pyinstaller'] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка сборки: {e}")
        sys.exit(1)

    # Результат
    if onefile:
        exe_path = Path('dist') / f'{app_name}.exe' if sys.platform == 'win32' else Path('dist') / app_name
    else:
        exe_path = Path('dist') / app_name

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024) if exe_path.is_file() else sum(
            f.stat().st_size for f in exe_path.rglob('*') if f.is_file()
        ) / (1024 * 1024)

        print(f"\n✅ Сборка завершена!")
        print(f"📁 Расположение: {exe_path.absolute()}")
        print(f"📦 Размер: {size_mb:.1f} MB")
        print("\n⚠️  Важно для Windows:")
        print("    - Положите ffmpeg.exe в ту же папку что и exe")
        print("    - Или установите ffmpeg в системный PATH")
    else:
        print("\n❌ Exe не найден после сборки")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Сборка Transcription Pro в exe')
    parser.add_argument('--dir', action='store_true', help='Сборка в папку (быстрее запуск)')
    parser.add_argument('--clean', action='store_true', help='Только очистка артефактов')

    args = parser.parse_args()

    print("=" * 50)
    print("🎙️ Audio/Video Transcription Pro - Сборка")
    print("=" * 50)

    if args.clean:
        clean_build_artifacts()
        return

    # Очищаем старые артефакты
    clean_build_artifacts()

    # Собираем
    build_exe(onefile=not args.dir)


if __name__ == '__main__':
    main()
