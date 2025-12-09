"""
Безопасный менеджер памяти для предотвращения крашей
"""

import gc
import time
import threading

from PySide6.QtCore import QMutex

from .config import DEVICE, TORCH_AVAILABLE

# Глобальный мьютекс для безопасности памяти
MEMORY_MUTEX = QMutex()


class CrashSafeMemoryManager:
    """Безопасный менеджер памяти для предотвращения крашей"""

    _cleanup_in_progress = False
    _cleanup_lock = threading.Lock()

    @staticmethod
    def safe_gpu_cleanup(stage="unknown"):
        """Ультра-безопасная очистка GPU памяти"""
        # Предотвращаем множественные одновременные вызовы
        with CrashSafeMemoryManager._cleanup_lock:
            if CrashSafeMemoryManager._cleanup_in_progress:
                return

            CrashSafeMemoryManager._cleanup_in_progress = True

        try:
            # Мягкая сборка мусора Python
            try:
                gc.collect()
            except:
                pass

            # Очень осторожная работа с CUDA
            if DEVICE == "cuda" and TORCH_AVAILABLE:
                try:
                    import torch

                    # Множественные проверки безопасности
                    if not torch.cuda.is_available():
                        return

                    if torch.cuda.device_count() == 0:
                        return

                    # Проверяем, что есть инициализированный контекст
                    try:
                        current_device = torch.cuda.current_device()
                    except:
                        return

                    # Только мягкая очистка кэша, без синхронизации
                    try:
                        allocated_before = torch.cuda.memory_allocated(current_device) / (1024**2)
                        torch.cuda.empty_cache()
                        allocated_after = torch.cuda.memory_allocated(current_device) / (1024**2)

                        freed = allocated_before - allocated_after
                        if freed > 1:
                            print(f"GPU cleanup {stage}: freed {freed:.1f}MB")

                    except Exception as cache_error:
                        print(f"Cache cleanup warning: {cache_error}")

                except Exception as cuda_error:
                    print(f"CUDA cleanup warning: {cuda_error}")

        except Exception as e:
            print(f"Memory cleanup error: {e}")
        finally:
            CrashSafeMemoryManager._cleanup_in_progress = False

    @staticmethod
    def reset_for_new_transcription():
        """Полный сброс состояния для новой транскрибации"""
        try:
            # Принудительная сборка мусора
            for _ in range(3):
                gc.collect()

            # Очистка GPU памяти
            CrashSafeMemoryManager.safe_gpu_cleanup("reset for new transcription")

            # Небольшая пауза для стабилизации
            time.sleep(0.1)

        except Exception as e:
            print(f"Reset error: {e}")
