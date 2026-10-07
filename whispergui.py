"""
WhisperGUI — Windows-приложение для локальной транскрибации аудио и видео.
Зависимости: PyQt6, faster-whisper, imageio-ffmpeg, soundfile, numpy, huggingface-hub.
Запуск:      python whispergui.py
"""
from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import monotonic

from PyQt6.QtCore import (
    Qt, QTimer, QPointF, QSize, QObject, pyqtSignal, QThread,
    QPropertyAnimation, QEasingCurve,
)
from PyQt6.QtGui import (
    QAction, QActionGroup, QColor, QPen, QPainter, QGuiApplication,
    QPixmap, QIcon,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QGroupBox, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QRadioButton, QButtonGroup,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QMenu, QToolBar, QLineEdit, QAbstractItemView, QFrame, QDialog,
    QDialogButtonBox, QGraphicsOpacityEffect, QScrollArea,
)

# =========================================================================
#     WINDOWED-СБОРКА: ЗАГЛУШКИ stdout/stderr (иначе tqdm/hf_hub падают)
# =========================================================================
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", buffering=1)
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", buffering=1)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# =========================================================================
#        SUBPROCESS БЕЗ ЧЁРНЫХ КОНСОЛЬНЫХ ОКОН (Windows)
# =========================================================================

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _silent_popen_kwargs() -> dict:
    kw: dict = {}
    if sys.platform == "win32":
        kw["creationflags"] = CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kw["startupinfo"] = si
    return kw


# =========================================================================
#                       КОРНЕВАЯ ПАПКА / ПУТИ
# =========================================================================


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bootstrap_path() -> Path:
    return app_dir() / "whispergui.location.json"


def load_bootstrap_root():
    try:
        with open(bootstrap_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        r = data.get("root")
        if not r:
            return None
        p = Path(r)
        if p.is_dir() and os.access(p, os.W_OK):
            return p
    except FileNotFoundError:
        return None
    except Exception:
        pass
    return None


def save_bootstrap_root(root: Path) -> None:
    try:
        with open(bootstrap_path(), "w", encoding="utf-8") as f:
            json.dump({"root": str(root)}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


ROOT: Path = Path()
MODELS_DIR: Path = Path()
CUDA_DIR: Path = Path()
LOGS_DIR: Path = Path()
CONFIG_DIR: Path = Path()
OUTPUT_DIR: Path = Path()
TMP_DIR: Path = Path()


def apply_root(root: Path, create: bool = True) -> None:
    global ROOT, MODELS_DIR, CUDA_DIR, LOGS_DIR, CONFIG_DIR, OUTPUT_DIR, TMP_DIR
    ROOT = Path(root)
    MODELS_DIR = ROOT / "models"
    CUDA_DIR = ROOT / "cuda"
    LOGS_DIR = ROOT / "logs"
    CONFIG_DIR = ROOT / "config"
    OUTPUT_DIR = ROOT / "output"
    TMP_DIR = OUTPUT_DIR / "tmp"
    if create:
        for d in (MODELS_DIR, CUDA_DIR, LOGS_DIR, CONFIG_DIR,
                  OUTPUT_DIR, TMP_DIR):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass


apply_root(app_dir(), create=False)


def app_icon_path() -> Path:
    return app_dir() / "whispergui.ico"


def load_app_icon() -> QIcon:
    p = app_icon_path()
    if p.exists():
        try:
            return QIcon(str(p))
        except Exception:
            pass
    return QIcon()


# =========================================================================
#                            ЛОГИ
# =========================================================================

LOG = logging.getLogger("whispergui")
LOG.setLevel(logging.INFO)


def setup_logging() -> logging.Logger:
    LOG.setLevel(logging.INFO)
    if LOG.handlers:
        return LOG
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            LOGS_DIR / "app.log", maxBytes=5 * 1024 * 1024,
            backupCount=5, encoding="utf-8",
        )
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    except Exception:
        pass
    return LOG


# =========================================================================
#                            НАСТРОЙКИ
# =========================================================================


class Settings:
    @property
    def PATH(self) -> Path:
        return CONFIG_DIR / "settings.json"

    def __init__(self):
        self.language = "ru"
        self.last_model = "small"
        self.device = "cpu"
        self.export_dir = ""
        self.cuda_may_not_work = False
        self.custom_models: list = []
        self.load()

    def load(self):
        try:
            with open(self.PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        except FileNotFoundError:
            pass
        except Exception:
            LOG.exception("Не удалось прочитать settings.json")
        if not self.export_dir:
            self.export_dir = str(OUTPUT_DIR)

    def save(self):
        try:
            data = {k: v for k, v in self.__dict__.items()
                    if not k.startswith("_")}
            self.PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            LOG.exception("Не удалось сохранить settings.json")


SETTINGS = Settings()


# =========================================================================
#                             ЛОКАЛИЗАЦИЯ
# =========================================================================

I18N = {
    "ru": {
        "app.title": "Whisper — Nova",
        "menu.language": "Язык",
        "menu.language.ru": "Русский",
        "menu.language.en": "English",
        "menu.change_root": "Сменить папку данных…",

        "root.dialog.title": "Папка данных WhisperGUI",
        "root.dialog.about": (
            "Выберите папку, где будут храниться модели, CUDA-компоненты, "
            "логи, настройки и результаты.\n\n"
            "EXE можно переносить куда угодно — путь к папке данных "
            "сохранится рядом с ним, и программа найдёт её автоматически."
        ),
        "root.dialog.browse": "Обзор…",
        "root.invalid": "Папка недоступна для записи.",
        "root.change_restart": (
            "Новая папка данных сохранена.\n"
            "Изменения вступят в силу после перезапуска."
        ),

        "files.group": "Очередь файлов",
        "files.add": "Добавить",
        "files.remove": "Удалить",
        "files.clear": "Очистить",
        "files.col.name": "Имя",
        "files.col.size": "Размер",
        "files.col.status": "Статус",
        "files.drag_hint": "Перетащите файлы или папки сюда",
        "files.status.pending": "В очереди",
        "files.status.transcoding": "Транскодирование",
        "files.status.transcribing": "Транскрибация",
        "files.status.done": "Готово",
        "files.status.error": "Ошибка",
        "files.status.cancelled": "Отменено",
        "files.too_big": "Файл «{name}» превышает 4 ГБ.",

        "completed.group": "Готовые файлы",
        "completed.col.name": "Имя",
        "completed.col.size": "Размер",
        "completed.hint": "Выберите файл, чтобы увидеть его текст",
        "completed.remove_confirm": "Удалить выбранные файлы из готовых?",
        "completed.clear_confirm": "Очистить весь список готовых файлов?",
        "completed.remove_nothing": "Ничего не выбрано.",

        "models.group": "Модель",
        "models.label": "Модель:",
        "models.download": "Скачать",
        "models.delete": "Удалить",
        "models.open_folder": "Папка моделей",
        "models.custom_placeholder": "HF-репозиторий…",
        "models.add_custom": "Добавить",
        "models.downloaded": "✓ Скачана",
        "models.not_downloaded": "○ Не скачана",
        "models.info.size": "Вес:",
        "models.info.quality": "Качество:",
        "models.info.ram": "RAM/VRAM:",
        "models.info.status": "Статус:",

        "models.qual.very_low":  "Очень низкое",
        "models.qual.low":       "Низкое",
        "models.qual.medium":    "Среднее",
        "models.qual.high":      "Высокое",
        "models.qual.very_high": "Очень высокое",
        "models.qual.max":       "Максимальное",

        "models.dl.title": "Скачивание модели",
        "models.dl.about": "Загрузка «{name}» из HuggingFace…",
        "models.dl.done": "Модель «{name}» скачана.",
        "models.dl.fail": "Не удалось скачать модель: {err}",
        "models.dl.no_space": "Мало места: нужно ~{need} МБ, свободно {free} МБ.",
        "models.rm.confirm": "Удалить модель «{name}» ({size} МБ)?",
        "models.rm.done": "Модель удалена.",
        "models.rm.fail": "Не удалось удалить модель: {err}",
        "models.custom.invalid": "Некорректный HF-репозиторий.",
        "models.custom.exists": "Такая модель уже есть в списке.",

        "device.group": "Устройство",
        "device.cpu": "CPU",
        "device.gpu": "GPU (CUDA)",
        "device.cuda_download": "CUDA",
        "device.cuda_check": "Проверить",
        "device.cuda_remove": "Удалить",
        "device.gpu_not_found": "NVIDIA GPU не обнаружена",
        "device.gpu_driver_old": "Драйвер NVIDIA слишком старый ({v}). Нужен 525 или новее для CUDA 12.",
        "device.cuda_not_installed": "CUDA-компоненты не установлены",
        "device.cuda_installed": "CUDA-компоненты установлены",
        "device.cuda_dialog_title": "CUDA-компоненты",
        "device.cuda_dialog_about": (
            "Для работы на NVIDIA GPU нужны библиотеки cuDNN 9 и cuBLAS 12 "
            "(~500 МБ). Они будут скачаны в папку cuda/"
        ),
        "device.cuda.dl.title": "Загрузка CUDA",
        "device.cuda.dl.about": "Скачивание CUDA-компонентов…",
        "device.cuda.dl.done": "CUDA-компоненты установлены.",
        "device.cuda.dl.fail": "Не удалось скачать CUDA: {err}",
        "device.cuda.rm.done": "CUDA-компоненты удалены.",
        "device.cuda.check.ok": "CUDA-библиотеки загружены успешно.",
        "device.cuda.check.fail": "Не удалось загрузить CUDA-библиотеки: {err}",
        "device.cuda.no_disk": "Недостаточно свободного места (нужно ~600 МБ).",
        "device.status.gpu_ok": "GPU: {name} · драйвер {drv}",
        "device.status.cuda_ok": "CUDA-компоненты готовы к работе.",

        "progress.group": "Процесс",
        "progress.start": "Старт",
        "progress.pause": "Пауза",
        "progress.resume": "Продолжить",
        "progress.stop": "Стоп",
        "progress.current_empty": "Файл не выбран",
        "progress.current_sub": "—",
        "progress.file": "Файл {n} из {m}",
        "progress.stage": "Этап: {stage}",
        "progress.stage.idle": "ожидание",
        "progress.stage.load_model": "загрузка модели",
        "progress.stage.transcoding": "транскодирование",
        "progress.stage.transcribing": "транскрибация",
        "progress.stage.done": "завершено",
        "progress.audio": "Обработано: {cur} / {total}",
        "progress.elapsed": "Прошло: {t}",
        "progress.eta": "Осталось: ~{t}",
        "progress.speed": "Скорость: {x}×",

        "result.group": "Результат",
        "result.show_timecodes": "Таймкоды",
        "result.copy_all": "Копировать",
        "result.placeholder": "Выберите готовый файл слева, чтобы увидеть текст…",
        "result.edited_warn": "Ваши правки в тексте будут потеряны. Продолжить?",

        "export.group": "Экспорт",
        "export.format": "Формат:",
        "export.save_as": "Сохранить как…",
        "export.save_all": "Все файлы в папку…",

        "dialog.info": "Информация",
        "dialog.warning": "Предупреждение",
        "dialog.error": "Ошибка",

        "queue.empty": "Очередь пуста.",
        "queue.no_installed_model": "Сначала скачайте выбранную модель.",
        "queue.cuda_missing": "CUDA-компоненты не установлены. Скачать их или запустить на CPU?",
        "queue.cuda_missing_title": "CUDA не установлена",
        "start.failed": "Не удалось запустить: {err}",

        "export.saved": "Сохранено: {path}",
        "export.no_text": "Нечего сохранять — результат пуст.",
        "export.choose_dir": "Выберите папку для экспорта",
        "export.done": "Экспортировано файлов: {n}",

        "dl.file": "Файл {i}/{n} — {rel}",
        "dl.unpack": "Распаковка {pkg}…",
        "dl.pkg": "{pkg} ({size} МБ)",
    },

    "en": {
        "app.title": "WhisperGUI — Nova",
        "menu.language": "Language",
        "menu.language.ru": "Русский",
        "menu.language.en": "English",
        "menu.change_root": "Change data folder…",

        "root.dialog.title": "WhisperGUI data folder",
        "root.dialog.about": (
            "Choose the folder where models, CUDA components, logs, "
            "settings and results will be stored.\n\n"
            "The EXE can be moved anywhere — the path to the data folder "
            "is saved next to it and found automatically."
        ),
        "root.dialog.browse": "Browse…",
        "root.invalid": "Folder is not writable.",
        "root.change_restart": (
            "New data folder saved.\n"
            "It will take effect after restart."
        ),

        "files.group": "File queue",
        "files.add": "Add",
        "files.remove": "Remove",
        "files.clear": "Clear",
        "files.col.name": "Name",
        "files.col.size": "Size",
        "files.col.status": "Status",
        "files.drag_hint": "Drag & drop files or folders here",
        "files.status.pending": "Pending",
        "files.status.transcoding": "Transcoding",
        "files.status.transcribing": "Transcribing",
        "files.status.done": "Done",
        "files.status.error": "Error",
        "files.status.cancelled": "Cancelled",
        "files.too_big": "File “{name}” exceeds 4 GB.",

        "completed.group": "Completed files",
        "completed.col.name": "Name",
        "completed.col.size": "Size",
        "completed.hint": "Pick a file to view its text",
        "completed.remove_confirm": "Remove selected files from completed?",
        "completed.clear_confirm": "Clear the entire completed list?",
        "completed.remove_nothing": "Nothing selected.",

        "models.group": "Model",
        "models.label": "Model:",
        "models.download": "Download",
        "models.delete": "Delete",
        "models.open_folder": "Models folder",
        "models.custom_placeholder": "HF repo…",
        "models.add_custom": "Add",
        "models.downloaded": "✓ Downloaded",
        "models.not_downloaded": "○ Not downloaded",
        "models.info.size": "Size:",
        "models.info.quality": "Quality:",
        "models.info.ram": "RAM/VRAM:",
        "models.info.status": "Status:",

        "models.qual.very_low":  "Very low",
        "models.qual.low":       "Low",
        "models.qual.medium":    "Medium",
        "models.qual.high":      "High",
        "models.qual.very_high": "Very high",
        "models.qual.max":       "Maximum",

        "models.dl.title": "Downloading model",
        "models.dl.about": "Fetching “{name}” from HuggingFace…",
        "models.dl.done": "Model “{name}” downloaded.",
        "models.dl.fail": "Failed to download model: {err}",
        "models.dl.no_space": "Not enough space: ~{need} MB needed, {free} MB free.",
        "models.rm.confirm": "Delete model “{name}” ({size} MB)?",
        "models.rm.done": "Model removed.",
        "models.rm.fail": "Failed to remove model: {err}",
        "models.custom.invalid": "Invalid HF repository.",
        "models.custom.exists": "This model is already in the list.",

        "device.group": "Device",
        "device.cpu": "CPU",
        "device.gpu": "GPU (CUDA)",
        "device.cuda_download": "CUDA",
        "device.cuda_check": "Verify",
        "device.cuda_remove": "Delete",
        "device.gpu_not_found": "NVIDIA GPU not detected",
        "device.gpu_driver_old": "NVIDIA driver is too old ({v}). Need 525+ for CUDA 12.",
        "device.cuda_not_installed": "CUDA components not installed",
        "device.cuda_installed": "CUDA components installed",
        "device.cuda_dialog_title": "CUDA components",
        "device.cuda_dialog_about": (
            "To run on NVIDIA GPU you need cuDNN 9 and cuBLAS 12 (~500 MB). "
            "They will be downloaded to the cuda/ folder near the app."
        ),
        "device.cuda.dl.title": "Downloading CUDA",
        "device.cuda.dl.about": "Fetching CUDA components…",
        "device.cuda.dl.done": "CUDA components installed.",
        "device.cuda.dl.fail": "Failed to download CUDA: {err}",
        "device.cuda.rm.done": "CUDA components removed.",
        "device.cuda.check.ok": "CUDA libraries loaded successfully.",
        "device.cuda.check.fail": "Failed to load CUDA libraries: {err}",
        "device.cuda.no_disk": "Not enough free disk space (~600 MB needed).",
        "device.status.gpu_ok": "GPU: {name} · driver {drv}",
        "device.status.cuda_ok": "CUDA components are ready.",

        "progress.group": "Progress",
        "progress.start": "Start",
        "progress.pause": "Pause",
        "progress.resume": "Resume",
        "progress.stop": "Stop",
        "progress.current_empty": "No file selected",
        "progress.current_sub": "—",
        "progress.file": "File {n} of {m}",
        "progress.stage": "Stage: {stage}",
        "progress.stage.idle": "idle",
        "progress.stage.load_model": "loading model",
        "progress.stage.transcoding": "transcoding",
        "progress.stage.transcribing": "transcribing",
        "progress.stage.done": "done",
        "progress.audio": "Processed: {cur} / {total}",
        "progress.elapsed": "Elapsed: {t}",
        "progress.eta": "Remaining: ~{t}",
        "progress.speed": "Speed: {x}×",

        "result.group": "Result",
        "result.show_timecodes": "Timecodes",
        "result.copy_all": "Copy",
        "result.placeholder": "Pick a completed file on the left to see its text…",
        "result.edited_warn": "Your edits will be lost. Continue?",

        "export.group": "Export",
        "export.format": "Format:",
        "export.save_as": "Save as…",
        "export.save_all": "All files to folder…",

        "dialog.info": "Information",
        "dialog.warning": "Warning",
        "dialog.error": "Error",

        "queue.empty": "The queue is empty.",
        "queue.no_installed_model": "Please download the selected model first.",
        "queue.cuda_missing": "CUDA components not installed. Download them or run on CPU?",
        "queue.cuda_missing_title": "CUDA not installed",
        "start.failed": "Failed to start: {err}",

        "export.saved": "Saved: {path}",
        "export.no_text": "Nothing to save — result is empty.",
        "export.choose_dir": "Choose export folder",
        "export.done": "Files exported: {n}",

        "dl.file": "File {i}/{n} — {rel}",
        "dl.unpack": "Unpacking {pkg}…",
        "dl.pkg": "{pkg} ({size} MB)",
    },
}


class Translator(QObject):
    language_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._lang = "ru"

    def set_language(self, lang: str):
        if lang == self._lang:
            return
        self._lang = lang
        SETTINGS.language = lang
        SETTINGS.save()
        self.language_changed.emit(lang)

    def language(self) -> str:
        return self._lang

    def tr(self, key: str, **kwargs) -> str:
        s = I18N.get(self._lang, {}).get(key) or I18N["ru"].get(key) or key
        if kwargs:
            try:
                s = s.format(**kwargs)
            except Exception:
                pass
        return s


TR = Translator()


def tr(key: str, **kwargs) -> str:
    return TR.tr(key, **kwargs)


def _fmt_bytes(b: int) -> str:
    b = float(b)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} PB"


def _fmt_size_bytes(b: float) -> str:
    return _fmt_bytes(int(b))


def _dot_icon(installed: bool, diameter: int = 12) -> QIcon:
    pix = QPixmap(diameter, diameter)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#4ade80") if installed else QColor("#6b7280"))
    p.drawEllipse(1, 1, diameter - 2, diameter - 2)
    p.end()
    return QIcon(pix)


QUALITY_KEYS = {
    1.0: "models.qual.very_low",
    2.0: "models.qual.low",
    3.0: "models.qual.medium",
    4.0: "models.qual.high",
    4.5: "models.qual.very_high",
    5.0: "models.qual.max",
}


def quality_label(value: float) -> str:
    key = QUALITY_KEYS.get(float(value))
    if key is None:
        nearest = min(QUALITY_KEYS.keys(), key=lambda k: abs(k - value))
        key = QUALITY_KEYS[nearest]
    return tr(key)


# =========================================================================
#                             ДАННЫЕ МОДЕЛЕЙ
# =========================================================================

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts"}
MEDIA_EXT = AUDIO_EXT | VIDEO_EXT
MAX_FILE_BYTES = 4 * 1024 * 1024 * 1024
CHUNK_SECONDS = 60.0
CHUNK_OVERLAP_SECONDS = 1.5

BUILTIN_MODELS = [
    ("Systran/faster-whisper-tiny",               "tiny",              75,  1.0, "~1 GB",   False),
    ("Systran/faster-whisper-base",               "base",             145,  2.0, "~1.5 GB", False),
    ("Systran/faster-whisper-small",              "small",            488,  3.0, "~2.5 GB", False),
    ("Systran/faster-whisper-medium",             "medium",          1530,  4.0, "~5 GB",   False),
    ("Systran/faster-whisper-large-v1",           "large-v1",        3090,  4.5, "~10 GB",  False),
    ("Systran/faster-whisper-large-v2",           "large-v2",        3090,  4.5, "~10 GB",  False),
    ("Systran/faster-whisper-large-v3",           "large-v3",        3090,  5.0, "~10 GB",  False),
    ("deepdml/faster-whisper-large-v3-turbo-ct2", "large-v3-turbo",  1620,  4.5, "~6 GB",   False),
    ("Systran/faster-distil-whisper-large-v3",    "distil-large-v3", 1510,  4.5, "~6 GB",   False),
]


def model_dir(name: str) -> Path:
    return MODELS_DIR / name


def model_installed(name: str) -> bool:
    d = model_dir(name)
    return d.is_dir() and (d / "model.bin").exists()


def list_models() -> list:
    out = []
    for repo, name, size, qual, ram, _ in BUILTIN_MODELS:
        out.append({
            "repo": repo, "name": name, "size": size,
            "qual": qual, "ram": ram, "custom": False,
            "installed": model_installed(name),
        })
    for cm in SETTINGS.custom_models:
        name = cm.get("name")
        if not name:
            continue
        out.append({
            "repo": cm.get("repo", ""), "name": name,
            "size": int(cm.get("size", 0)),
            "qual": float(cm.get("qual", 3.0)),
            "ram": cm.get("ram", "?"), "custom": True,
            "installed": model_installed(name),
        })
    return out


# =========================================================================
#               ЗАГРУЗКА / УДАЛЕНИЕ МОДЕЛЕЙ
# =========================================================================


def _http_download(url: str, dest: Path,
                   on_bytes=None, headers: dict | None = None) -> int:
    req = urllib.request.Request(url, headers=headers or {})
    got = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(1024 * 256)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if on_bytes:
                    on_bytes(got, total)
        tmp.replace(dest)
    return got


def list_repo_files_with_sizes(repo: str) -> list:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(repo_id=repo, files_metadata=True)
    out = []
    for s in info.siblings:
        rel = s.rfilename
        if rel.startswith("."):
            continue
        out.append((rel, int(getattr(s, "size", 0) or 0)))
    return out


def download_model(repo: str, name: str, on_file=None, on_progress=None) -> None:
    from huggingface_hub import hf_hub_url

    target = model_dir(name)
    target.mkdir(parents=True, exist_ok=True)

    files = list_repo_files_with_sizes(repo)
    if not files:
        raise RuntimeError(f"Empty repo: {repo}")
    total_bytes = sum(sz for _, sz in files) or 1
    n = len(files)
    done_bytes = 0
    last_emit = 0.0

    for i, (rel, sz) in enumerate(files, 1):
        if on_file:
            on_file(rel, i, n)
        url = hf_hub_url(repo_id=repo, filename=rel)
        dest = target / rel

        base = done_bytes

        def _cb(got: int, _total: int, base=base):
            nonlocal last_emit
            now = monotonic()
            if on_progress and (now - last_emit > 0.08):
                last_emit = now
                done = base + got
                on_progress(int(done * 100 / total_bytes), done, total_bytes)

        got = _http_download(url, dest, on_bytes=_cb)
        done_bytes += got if got else sz
        if on_progress:
            on_progress(int(done_bytes * 100 / total_bytes),
                        done_bytes, total_bytes)

    if on_progress:
        on_progress(100, total_bytes, total_bytes)


def remove_model(name: str) -> None:
    d = model_dir(name)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


# =========================================================================
#                              CUDA
# =========================================================================

CUDA_MIN_DRIVER = (525, 60)


def find_nvidia_smi():
    candidates = [
        shutil.which("nvidia-smi"),
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def query_gpu():
    smi = find_nvidia_smi()
    if not smi:
        return None, None, ""
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=8,
            **_silent_popen_kwargs(),
        ).decode("utf-8", errors="ignore").strip()
        if not out:
            return None, None, ""
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 2:
            return None, None, ""
        name, drv = parts[0], parts[1]
        m = re.match(r"(\d+)\.(\d+)", drv)
        drv_tuple = (int(m.group(1)), int(m.group(2))) if m else None
        return name, drv_tuple, drv
    except Exception:
        LOG.exception("nvidia-smi query failed")
        return None, None, ""


def cuda_installed() -> bool:
    cudnn_bin = CUDA_DIR / "nvidia" / "cudnn" / "bin"
    cublas_bin = CUDA_DIR / "nvidia" / "cublas" / "bin"
    return cudnn_bin.exists() and cublas_bin.exists()


def remove_cuda() -> None:
    if CUDA_DIR.exists():
        shutil.rmtree(CUDA_DIR, ignore_errors=True)
    CUDA_DIR.mkdir(parents=True, exist_ok=True)


def activate_cuda() -> bool:
    cudnn_bin = CUDA_DIR / "nvidia" / "cudnn" / "bin"
    cublas_bin = CUDA_DIR / "nvidia" / "cublas" / "bin"
    if not (cudnn_bin.exists() and cublas_bin.exists()):
        return False
    try:
        extra = os.pathsep.join([str(cudnn_bin), str(cublas_bin)])
        os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for d in (cudnn_bin, cublas_bin):
                try:
                    os.add_dll_directory(str(d))
                except Exception:
                    pass
        return True
    except Exception:
        LOG.exception("activate_cuda failed")
        return False


def _pypi_wheel_info(package: str, platform_tag: str = "win_amd64"):
    with urllib.request.urlopen(
            f"https://pypi.org/pypi/{package}/json", timeout=30) as r:
        meta = json.load(r)
    version = meta["info"]["version"]
    files = meta.get("releases", {}).get(version, [])
    candidates = [f for f in files
                  if f.get("packagetype") == "bdist_wheel"
                  and platform_tag in f["filename"]]
    if not candidates:
        raise RuntimeError(f"Не найден wheel для {package} ({platform_tag})")
    candidates.sort(key=lambda f: 0 if "py3-none" in f["filename"] else 1)
    f = candidates[0]
    return f["url"], f["filename"], int(f.get("size", 0) or 0)


def download_cuda(on_status=None, on_progress=None) -> None:
    CUDA_DIR.mkdir(parents=True, exist_ok=True)
    packages = ["nvidia-cudnn-cu12", "nvidia-cublas-cu12"]

    plan = []
    total_bytes = 0
    for pkg in packages:
        url, fname, size = _pypi_wheel_info(pkg)
        plan.append((pkg, url, size, fname))
        total_bytes += size
    total_bytes = total_bytes or 1

    done_bytes = 0
    for pkg, url, size, fname in plan:
        if on_status:
            on_status(tr("dl.pkg", pkg=pkg, size=size // (1024 * 1024)))
        with tempfile.TemporaryDirectory() as td:
            whl = Path(td) / fname
            base = done_bytes
            last_emit = [0.0]

            def _cb(got, _total, base=base):
                now = monotonic()
                if on_progress and (now - last_emit[0] > 0.08):
                    last_emit[0] = now
                    done = base + got
                    on_progress(int(done * 100 / total_bytes),
                                done, total_bytes)

            _http_download(url, whl, on_bytes=_cb)
            if on_status:
                on_status(tr("dl.unpack", pkg=pkg))
            with zipfile.ZipFile(whl, "r") as z:
                for member in z.namelist():
                    if member.startswith("nvidia/"):
                        z.extract(member, CUDA_DIR)
            done_bytes += size

    for p in CUDA_DIR.glob("nvidia/*.dist-info"):
        shutil.rmtree(p, ignore_errors=True)

    if on_progress:
        on_progress(100, total_bytes, total_bytes)


def cuda_check_load():
    if not activate_cuda():
        return False, "CUDA DLL not found"
    try:
        cudnn_bin = CUDA_DIR / "nvidia" / "cudnn" / "bin"
        cublas_bin = CUDA_DIR / "nvidia" / "cublas" / "bin"
        loaded = []
        for d in (cudnn_bin, cublas_bin):
            if not d.exists():
                continue
            for dll in d.glob("*.dll"):
                try:
                    ctypes.WinDLL(str(dll))
                    loaded.append(dll.name)
                except OSError:
                    pass
        if not loaded:
            return False, "no DLL loaded"
        return True, f"loaded: {len(loaded)} DLL"
    except Exception as e:
        return False, str(e)


# =========================================================================
#                    АУДИО → WAV (ffmpeg, без окна)
# =========================================================================


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def transcode_to_wav(src: str, dst_wav: str) -> None:
    exe = ffmpeg_exe()
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-y",
        "-i", src, "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", dst_wav,
    ]
    r = subprocess.run(
        cmd, check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **_silent_popen_kwargs(),
    )
    if r.returncode != 0:
        msg = (r.stderr or b"").decode("utf-8", "replace").strip()[-400:]
        raise RuntimeError(f"ffmpeg: {msg or 'failed'}")


def _fmt_hms(sec: float) -> str:
    if sec is None or sec < 0 or math.isinf(sec) or math.isnan(sec):
        return "--:--:--"
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _fmt_vtt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# =========================================================================
#                       QTHREAD WORKER — ТРАНСКРИБАЦИЯ
# =========================================================================


class _Cancelled(Exception):
    pass


class TranscribeWorker(QThread):
    sig_status = pyqtSignal(int, str)
    sig_stage = pyqtSignal(str)
    sig_file_progress = pyqtSignal(float)
    sig_queue_progress = pyqtSignal(float)
    sig_file_time = pyqtSignal(int, float, float)
    sig_speed = pyqtSignal(float, float)
    # idx, полный список сегментов на текущий момент (для live-текста)
    sig_partial_segments = pyqtSignal(int, list)
    sig_file_done = pyqtSignal(int, str, list)
    sig_file_error = pyqtSignal(int, str, str)
    sig_finished = pyqtSignal()
    sig_log = pyqtSignal(str)

    def __init__(self, files, model_path: str, device: str,
                 compute_type: str, language=None):
        super().__init__()
        self.files = list(files)
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._resume_evt = threading.Event()
        self._resume_evt.set()
        self._stop_flag = False

    def pause(self):
        self._resume_evt.clear()

    def resume(self):
        self._resume_evt.set()

    def stop(self):
        self._stop_flag = True
        self._resume_evt.set()

    def _wait_if_paused(self) -> bool:
        while not self._resume_evt.is_set():
            if self._stop_flag:
                return False
            time.sleep(0.05)
        return not self._stop_flag

    def run(self):
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            LOG.exception("faster-whisper import failed")
            self.sig_log.emit(f"faster-whisper import failed: {e}")
            self.sig_finished.emit()
            return

        self.sig_stage.emit("progress.stage.load_model")
        try:
            model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            LOG.exception("WhisperModel init failed")
            for i, f in enumerate(self.files):
                self.sig_file_error.emit(i, f, str(e))
            self.sig_finished.emit()
            return

        n_total = len(self.files)
        for idx, path in enumerate(self.files):
            if not self._wait_if_paused():
                break
            self.sig_queue_progress.emit(idx / max(1, n_total))
            try:
                segments = self._transcribe_file(model, idx, path)
                self.sig_file_done.emit(idx, path, segments)
            except _Cancelled:
                self.sig_status.emit(idx, "files.status.cancelled")
                break
            except Exception as e:
                LOG.exception("transcribe file failed: %s", path)
                self.sig_file_error.emit(idx, path, str(e))
        self.sig_queue_progress.emit(1.0)
        self.sig_stage.emit("progress.stage.done")
        self.sig_finished.emit()

    def _transcribe_file(self, model, idx: int, path: str) -> list:
        self.sig_status.emit(idx, "files.status.transcoding")
        self.sig_stage.emit("progress.stage.transcoding")
        self.sig_file_progress.emit(0.0)

        tmp_dir = TMP_DIR if os.access(TMP_DIR, os.W_OK) else Path(tempfile.gettempdir())
        wav_path = tmp_dir / f"wg_{os.getpid()}_{idx}_{int(time.time())}.wav"
        try:
            transcode_to_wav(path, str(wav_path))
        except Exception:
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception:
                pass
            raise

        self.sig_status.emit(idx, "files.status.transcribing")
        self.sig_stage.emit("progress.stage.transcribing")

        out: list = []
        t_start = monotonic()
        last_emit = 0.0
        speed_window: list = []
        chunk_index = 0
        last_partial_emit = 0.0

        try:
            import soundfile as sf
            with sf.SoundFile(str(wav_path)) as f:
                sr = int(f.samplerate) or 16000
                total_frames = int(f.frames)
                total_dur = total_frames / sr if sr > 0 else 0.0

                chunk_frames = int(CHUNK_SECONDS * sr)
                overlap_frames = int(CHUNK_OVERLAP_SECONDS * sr)
                step_frames = max(1, chunk_frames - overlap_frames)

                def _emit_progress(processed_sec: float):
                    nonlocal last_emit
                    if total_dur <= 0:
                        return
                    frac = min(1.0, processed_sec / total_dur)
                    self.sig_file_progress.emit(frac)
                    self.sig_file_time.emit(idx, processed_sec, total_dur)
                    now = monotonic()
                    speed_window.append((processed_sec, now))
                    while speed_window and now - speed_window[0][1] > 10.0:
                        speed_window.pop(0)
                    if len(speed_window) >= 2:
                        dp = speed_window[-1][0] - speed_window[0][0]
                        dt = speed_window[-1][1] - speed_window[0][1]
                        speed_x = (dp / dt) if dt > 0.01 else 0.0
                        eta = (total_dur - processed_sec) / speed_x \
                            if speed_x > 0.01 else -1
                        if now - last_emit > 0.4:
                            last_emit = now
                            self.sig_speed.emit(speed_x, eta)

                def _emit_partial(force: bool = False):
                    """Отправляем накопленные сегменты наверх для live-текста."""
                    nonlocal last_partial_emit
                    now = monotonic()
                    if force or (now - last_partial_emit) > 0.35:
                        last_partial_emit = now
                        self.sig_partial_segments.emit(idx, list(out))

                offset = 0
                while offset < total_frames:
                    if self._stop_flag:
                        raise _Cancelled()
                    if not self._wait_if_paused():
                        raise _Cancelled()

                    f.seek(offset)
                    block = f.read(chunk_frames, dtype="float32", always_2d=False)
                    if block is None or len(block) == 0:
                        break

                    chunk_start_sec = offset / sr
                    chunk_len_sec = len(block) / sr

                    segments_gen, _info = model.transcribe(
                        block,
                        beam_size=5,
                        vad_filter=True,
                        language=self.language,
                    )

                    skip_before = chunk_start_sec + (
                        CHUNK_OVERLAP_SECONDS if chunk_index > 0 else 0.0
                    )

                    for seg in segments_gen:
                        if self._stop_flag:
                            raise _Cancelled()
                        if not self._wait_if_paused():
                            raise _Cancelled()

                        abs_start = chunk_start_sec + float(seg.start)
                        abs_end = chunk_start_sec + float(seg.end)

                        _emit_progress(min(total_dur, abs_end))

                        if abs_start + 0.05 < skip_before:
                            continue

                        out.append({
                            "start": abs_start,
                            "end": abs_end,
                            "text": (seg.text or "").strip(),
                        })
                        # Обновляем live-текст после каждого нового сегмента
                        _emit_partial(force=False)

                    _emit_progress(min(total_dur, chunk_start_sec + chunk_len_sec))
                    # По завершении чанка — точно отправим
                    _emit_partial(force=True)

                    chunk_index += 1
                    offset += step_frames
        finally:
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception:
                LOG.warning("Не удалось удалить tmp WAV: %s", wav_path)
            elapsed = monotonic() - t_start
            LOG.info("Файл %s обработан за %.1f сек, чанков: %d, сегментов: %d",
                     path, elapsed, chunk_index, len(out))

        return out


# =========================================================================
#                  QTHREAD WORKERS — ЗАГРУЗКИ
# =========================================================================


class ModelDownloadThread(QThread):
    sig_file = pyqtSignal(str, int, int)
    sig_progress = pyqtSignal(int, int, int)
    sig_done = pyqtSignal()
    sig_error = pyqtSignal(str)

    def __init__(self, repo: str, name: str):
        super().__init__()
        self.repo = repo
        self.name = name

    def run(self):
        try:
            download_model(
                self.repo, self.name,
                on_file=lambda rel, i, n: self.sig_file.emit(rel, i, n),
                on_progress=lambda p, d, t: self.sig_progress.emit(p, d, t),
            )
            self.sig_done.emit()
        except Exception as e:
            LOG.exception("download_model failed")
            self.sig_error.emit(str(e))


class CudaDownloadThread(QThread):
    sig_status = pyqtSignal(str)
    sig_progress = pyqtSignal(int, int, int)
    sig_done = pyqtSignal()
    sig_error = pyqtSignal(str)

    def run(self):
        try:
            download_cuda(
                on_status=lambda s: self.sig_status.emit(s),
                on_progress=lambda p, d, t: self.sig_progress.emit(p, d, t),
            )
            self.sig_done.emit()
        except Exception as e:
            LOG.exception("download_cuda failed")
            self.sig_error.emit(str(e))


# =========================================================================
#                              ЭКСПОРТ
# =========================================================================


def export_result(fmt: str, segments: list, path: str) -> None:
    fmt = fmt.upper()
    if fmt == "TXT":
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(s["text"] for s in segments))
    elif fmt == "SRT":
        with open(path, "w", encoding="utf-8") as f:
            for i, s in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{_fmt_srt(s['start'])} --> {_fmt_srt(s['end'])}\n")
                f.write(f"{s['text']}\n\n")
    elif fmt == "VTT":
        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, s in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{_fmt_vtt(s['start'])} --> {_fmt_vtt(s['end'])}\n")
                f.write(f"{s['text']}\n\n")
    elif fmt == "JSON":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    elif fmt == "TSV":
        with open(path, "w", encoding="utf-8") as f:
            for s in segments:
                f.write(f"{s['start']:.3f}\t{s['end']:.3f}\t{s['text']}\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")


# =========================================================================
#                          ЧЁРНАЯ ДЫРА
# =========================================================================


class BlackHoleWidget(QWidget):
    BH_REL        = 0.090
    DISK_IN_REL   = 0.120
    DISK_OUT_REL  = 0.500
    TILT          = 0.34

    PARTICLES      = 420
    ARC_SAMPLES    = 5
    TRAIL_MUL      = 1.05
    LINE_WIDTH_MUL = 0.55
    DENSITY_POWER  = 0.50
    FADE_IN_FRAMES = 76
    DEATH_FADE_REL = 0.055

    FORM_BOOST_REL = 0.96
    SPIN_MAX       = 12.0
    SPIN_DECAY_TAU = 1.6

    FORM_DURATION = 4.0
    DIS_DURATION  = 3.0
    FPS           = 60

    PAUSE_RAMP    = 0.35

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setMinimumHeight(170)
        self.setMaximumHeight(220)

        self._phase = "idle"
        self._phase_t = 0.0
        self._core_scale = 0.0
        self._form_progress = 0.0
        self._dis_progress = 0.0
        self._spin_boost = 1.0
        self._blur = 0.0

        self._time_scale = 1.0
        self._pause_target = 1.0

        self._last_t = monotonic()
        self._particles = []
        self._spawn_particles()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(int(1000 / self.FPS))
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._time_scale = 1.0
        self._pause_target = 1.0
        self._reset_timeline()
        self._phase = "forming"
        self._reset_particles()
        if not self._timer.isActive():
            self._last_t = monotonic()
            self._timer.start()
        self.show()

    def stop(self) -> None:
        if self._phase in ("idle", "disintegrating"):
            return
        for pt in self._particles:
            pt["dis_start_r"] = pt["r"]
            pt["dis_start_alpha"] = pt["p_alpha"]
        self._phase = "disintegrating"
        self._phase_t = 0.0
        self._pause_target = 1.0
        if not self._timer.isActive():
            self._last_t = monotonic()
            self._timer.start()

    def pause(self) -> None:
        if self._phase == "idle":
            return
        self._pause_target = 0.0
        if not self._timer.isActive():
            self._last_t = monotonic()
            self._timer.start()

    def resume(self) -> None:
        if self._phase == "idle":
            return
        self._pause_target = 1.0
        if not self._timer.isActive():
            self._last_t = monotonic()
            self._timer.start()

    def _reset_timeline(self) -> None:
        self._phase_t = 0.0
        self._core_scale = 0.01
        self._form_progress = 0.0
        self._dis_progress = 0.0
        self._spin_boost = 1.0
        self._blur = 0.0

    def _update_timeline(self, dt: float) -> None:
        self._phase_t += dt
        if self._phase == "forming":
            p = min(self._phase_t / self.FORM_DURATION, 1.0)
            self._form_progress = p
            self._dis_progress = 0.0
            self._core_scale = 1.0 - (1.0 - p) ** 3
            self._spin_boost = 1.0 + (self.SPIN_MAX - 1.0) * (p ** 1.5)
            self._blur = p * p
            if self._phase_t >= self.FORM_DURATION:
                self._phase = "stable"
                self._phase_t = 0.0
                self._core_scale = 1.0
                self._blur = 1.0
                for pt in self._particles:
                    pt["age"] = self.FADE_IN_FRAMES
        elif self._phase == "stable":
            self._core_scale = 1.0
            self._form_progress = 1.0
            self._dis_progress = 0.0
            self._blur = 1.0
            self._spin_boost += (1.0 - self._spin_boost) * (
                1.0 - math.exp(-dt / self.SPIN_DECAY_TAU))
            if self._spin_boost < 1.01:
                self._spin_boost = 1.0
        elif self._phase == "disintegrating":
            p = min(self._phase_t / self.DIS_DURATION, 1.0)
            self._dis_progress = p
            self._form_progress = 1.0
            self._core_scale = (1.0 - p) ** 1.8
            self._spin_boost = 1.0 + (self.SPIN_MAX - 1.0) * (p ** 1.5)
            self._blur = 1.0
            if self._phase_t >= self.DIS_DURATION:
                self._phase = "idle"
                self._core_scale = 0.0
                self._blur = 0.0

    def _pick_color(self, mix, rnd):
        centerness = 1.0 - mix
        roll = rnd.random()
        if roll < 0.10 and mix > 0.40:
            return (44.0 + rnd.random() * 16.0,
                    (62.0 + rnd.random() * 18.0) / 100.0 * 255.0,
                    0.55 + rnd.random() * 0.40)
        if roll < 0.28 and mix > 0.50:
            return (8.0 + rnd.random() * 16.0,
                    (50.0 + rnd.random() * 18.0) / 100.0 * 255.0,
                    0.50 + rnd.random() * 0.40)
        ranges = [(4.0, 22.0), (26.0, 44.0)]
        lo, hi = ranges[rnd.randrange(len(ranges))]
        edge_h = lo + rnd.random() * (hi - lo)
        h = edge_h * (1.0 - centerness * 0.65) + 48.0 * (centerness * 0.65)
        l = 52.0 + centerness * (96.0 - 52.0) + rnd.random() * 4.0
        if l > 97.0:
            l = 97.0
        a = 0.50 + centerness * 0.35 + rnd.random() * 0.15
        return (h, l / 100.0 * 255.0, a)

    def _make_particle(self, rnd, stagger=False):
        u = rnd.random()
        biased = u ** self.DENSITY_POWER
        natural_r = self.DISK_IN_REL + biased * (self.DISK_OUT_REL - self.DISK_IN_REL)
        r_norm = (natural_r - self.DISK_IN_REL) / (self.DISK_OUT_REL - self.DISK_IN_REL)
        hue, lum, calpha = self._pick_color(r_norm, rnd)
        return {
            "natural_r": natural_r, "r": natural_r,
            "angle": rnd.random() * math.tau,
            "speed_jitter": 0.8 + rnd.random() * 0.4, "speed": 0.0,
            "hue": hue, "lum": lum, "calpha": calpha,
            "qcolor": QColor.fromHslF((hue % 360.0) / 360.0, 0.95, lum / 255.0, calpha),
            "line_width": 0.6 + rnd.random() * 0.9,
            "trail": 0.6 + rnd.random() * 1.0,
            "arc_span": 0.3,
            "age": (rnd.random() * self.FADE_IN_FRAMES) if stagger else 0,
            "p_alpha": 1.0 if stagger else 0.0,
            "form_delay": r_norm * 0.40 + rnd.random() * 0.15,
            "dis_delay": (1.0 - r_norm) * 0.35 + rnd.random() * 0.15,
        }

    def _spawn_particles(self):
        rnd = random.Random(20260101)
        self._particles.clear()
        for _ in range(self.PARTICLES):
            self._particles.append(self._make_particle(rnd, stagger=True))

    def _reset_particles(self):
        rnd = random.Random()
        for i in range(len(self._particles)):
            self._particles[i] = self._make_particle(rnd, stagger=False)

    def _tick(self):
        now = monotonic()
        real_dt = max(0.0, min(0.08, now - self._last_t))
        self._last_t = now

        target = self._pause_target
        diff = target - self._time_scale
        step = real_dt / self.PAUSE_RAMP
        if abs(diff) <= step:
            self._time_scale = target
        else:
            self._time_scale += step if diff > 0 else -step

        if target == 0.0 and self._time_scale <= 0.001:
            self._time_scale = 0.0
            self._timer.stop()
            self.update()
            return

        dt = real_dt * self._time_scale
        self._update_timeline(dt)
        if self._phase != "idle":
            self._update_particles(dt)
            self.update()
        else:
            self.update()
            self._timer.stop()

    def _update_particles(self, dt):
        inner = self.DISK_IN_REL
        outer = self.DISK_OUT_REL
        forming = (self._phase == "forming")
        dis = (self._phase == "disintegrating")
        fp = self._form_progress
        dp = self._dis_progress
        spin_boost = self._spin_boost
        blur = self._blur

        for pt in self._particles:
            r_safe = max(pt["r"], 0.02)
            pt["speed"] = (1.5 / math.sqrt(r_safe)) * pt["speed_jitter"] * spin_boost
            pt["angle"] += pt["speed"] * dt * 1.5

            if forming:
                delay = pt["form_delay"]
                localP = 0.0 if delay >= 1.0 else max(
                    0.0, min(1.0, (fp - delay) / (1.0 - delay)))
                eased = 1.0 - (1.0 - localP) ** 3
                pt["r"] = pt["natural_r"] + (1.0 - eased) * self.FORM_BOOST_REL
                pt["p_alpha"] = localP * localP * (3.0 - 2.0 * localP)
            elif dis:
                delay = pt["dis_delay"]
                localP = 1.0 if delay >= 1.0 else max(
                    0.0, min(1.0, (dp - delay) / (1.0 - delay)))
                eased = localP * localP * (3.0 - 2.0 * localP)
                r_start = pt.get("dis_start_r", pt["natural_r"])
                pt["r"] = r_start + (inner - r_start) * eased
                a_start = pt.get("dis_start_alpha", 1.0)
                pt["p_alpha"] = a_start * (1.0 - localP)
            else:
                pt["r"] -= 0.016 * dt
                if pt["r"] < inner:
                    u = random.random()
                    biased = u ** self.DENSITY_POWER
                    new_r = inner + biased * (outer - inner)
                    pt["natural_r"] = new_r
                    pt["r"] = new_r
                    pt["angle"] = random.random() * math.tau
                    pt["age"] = 0
                    r_norm = (new_r - inner) / (outer - inner)
                    h, l, a = self._pick_color(r_norm, random)
                    pt["hue"] = h
                    pt["lum"] = l
                    pt["calpha"] = a
                    pt["qcolor"] = QColor.fromHslF(
                        (h % 360.0) / 360.0, 0.95, l / 255.0, a)
                    pt["form_delay"] = r_norm * 0.40 + random.random() * 0.15
                    pt["dis_delay"] = (1.0 - r_norm) * 0.35 + random.random() * 0.15
                pt["age"] += dt * 60.0
                a = min(1.0, pt["age"] / self.FADE_IN_FRAMES)
                a = a * a * (3.0 - 2.0 * a)
                alpha = a
                fade_start = inner + self.DEATH_FADE_REL
                if pt["r"] < fade_start:
                    f = (pt["r"] - inner) / self.DEATH_FADE_REL
                    if f < 0.0:
                        f = 0.0
                    f = f * f * (3.0 - 2.0 * f)
                    alpha *= f
                pt["p_alpha"] = alpha

            raw_len = (55.0 + pt["speed"] * 520.0 * pt["trail"]) * self.TRAIL_MUL
            visual_len = raw_len * blur
            if visual_len < 1.5:
                visual_len = 1.5
            span = visual_len / max(pt["r"] * 500.0, 5.0)
            if span > 1.05:
                span = 1.05
            pt["arc_span"] = span

    def paintEvent(self, _event):
        if self._phase == "idle":
            return
        w, h = self.width(), self.height()
        if w < 40 or h < 40:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cx, cy = w / 2.0, h / 2.0
        scale = min(w, h)
        bh_r_px = max(0.3, scale * self.BH_REL * self._core_scale)

        items = []
        for pt in self._particles:
            if pt["p_alpha"] <= 0.005:
                continue
            ang = pt["angle"]
            r_px = pt["r"] * scale
            wz = math.sin(ang) * r_px
            items.append((wz, pt, ang, r_px))
        items.sort(key=lambda t: t[0])

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for wz, pt, ang, r_px in items:
            if wz >= 0:
                break
            self._draw_particle(p, cx, cy, bh_r_px, pt, ang, r_px, wz)

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 255))
        if bh_r_px > 0.5:
            p.drawEllipse(QPointF(cx, cy), bh_r_px, bh_r_px)

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for wz, pt, ang, r_px in items:
            if wz < 0:
                continue
            self._draw_particle(p, cx, cy, bh_r_px, pt, ang, r_px, wz)

        if self._phase == "disintegrating":
            ring_a = 1.0 - self._dis_progress
        elif self._phase == "forming":
            ring_a = self._form_progress
        else:
            ring_a = 1.0
        ring_a = max(0.0, min(1.0, ring_a))
        if bh_r_px > 0.5:
            pen = QPen(QColor(255, 210, 160, int(56 * ring_a)))
            pen.setWidthF(1.4)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), bh_r_px + 1.2, bh_r_px + 1.2)
        p.end()

    def _draw_particle(self, p, cx, cy, bh_r_px, pt, ang, r_px, wz):
        base_alpha = pt["p_alpha"] * pt["calpha"]
        if base_alpha <= 0.005:
            return
        N = self.ARC_SAMPLES
        span = pt["arc_span"]
        col = QColor(pt["qcolor"])
        pen = QPen(col)
        pen.setWidthF(pt["line_width"] * self.LINE_WIDTH_MUL)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        if span < 0.004:
            x = math.cos(ang) * r_px
            z = math.sin(ang) * r_px
            px = cx + x
            py = cy + z * self.TILT
            radius = max(0.35, pt["line_width"] * self.LINE_WIDTH_MUL * 1.2)
            col.setAlpha(max(0, min(255, int(base_alpha * 255))))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawEllipse(QPointF(px, py), radius, radius)
            return

        a0 = ang - span
        step = span / N
        lens_ref = 1.10 * bh_r_px
        lens_amp = 1.25 * bh_r_px
        prev = None
        for s in range(N + 1):
            a = a0 + s * step
            x = math.cos(a) * r_px
            z = math.sin(a) * r_px
            px = cx + x
            py = cy + z * self.TILT
            if wz < 0.0 and bh_r_px > 0.5:
                dx = px - cx
                dy = py - cy
                d0 = math.hypot(dx, dy)
                if d0 > 0.001:
                    uu = d0 / lens_ref
                    push = lens_amp * math.exp(-uu * uu)
                    k = (d0 + push) / d0
                    px = cx + dx * k
                    py = cy + dy * k
            if prev is not None:
                t = s / N if N > 0 else 1.0
                a_seg = base_alpha * t * t
                col.setAlpha(max(0, min(255, int(a_seg * 255))))
                pen.setColor(col)
                p.setPen(pen)
                p.drawLine(QPointF(prev[0], prev[1]), QPointF(px, py))
            prev = (px, py)


# =========================================================================
#                              ПАНЕЛИ UI
# =========================================================================


class FilePanel(QGroupBox):
    COLUMNS = ("files.col.name", "files.col.size", "files.col.status")

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 18, 8, 8)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_add = QPushButton()
        self.btn_remove = QPushButton()
        self.btn_clear = QPushButton()
        for b in (self.btn_add, self.btn_remove, self.btn_clear):
            b.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        # Компактнее: минимум 90, максимум 130
        self.table.setMinimumHeight(90)
        self.table.setMaximumHeight(130)
        lay.addWidget(self.table)

        self.hint = QLabel()
        self.hint.setStyleSheet("color:#6f747c; font-size: 11px;")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.hint)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_clear.clicked.connect(self._on_clear)
        self.retranslate()

    def retranslate(self):
        self.setTitle(tr("files.group"))
        self.btn_add.setText(tr("files.add"))
        self.btn_remove.setText(tr("files.remove"))
        self.btn_clear.setText(tr("files.clear"))
        self.hint.setText(tr("files.drag_hint"))
        for i, key in enumerate(self.COLUMNS):
            self.table.setHorizontalHeaderItem(i, QTableWidgetItem(tr(key)))
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 2)
            if it:
                key = it.data(Qt.ItemDataRole.UserRole)
                if key:
                    it.setText(tr(key))

    def _on_add(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("files.add"), "",
            "Media (*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma *.opus "
            "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv *.m4v *.ts);;All (*.*)")
        for f in files:
            self._add_row(f)

    def _on_remove(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _on_clear(self):
        self.table.setRowCount(0)

    def _add_row(self, path):
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > MAX_FILE_BYTES:
            QMessageBox.warning(self, tr("dialog.warning"),
                                tr("files.too_big", name=os.path.basename(path)))
            return
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = [os.path.basename(path), _fmt_size_bytes(size), tr("files.status.pending")]
        keys = [None, None, "files.status.pending"]
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            it.setToolTip(path if c == 0 else v)
            if c == 0:
                it.setData(Qt.ItemDataRole.UserRole + 1, path)
            if keys[c]:
                it.setData(Qt.ItemDataRole.UserRole, keys[c])
            self.table.setItem(r, c, it)

    def set_status(self, row: int, key: str):
        it = self.table.item(row, 2)
        if not it:
            return
        it.setData(Qt.ItemDataRole.UserRole, key)
        it.setText(tr(key))

    def file_paths(self) -> list:
        paths = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                p = it.data(Qt.ItemDataRole.UserRole + 1)
                if p:
                    paths.append(p)
        return paths

    def row_for_path(self, path: str) -> int:
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole + 1) == path:
                return r
        return -1

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p:
                continue
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        if os.path.splitext(fn)[1].lower() in MEDIA_EXT:
                            self._add_row(os.path.join(root, fn))
            elif os.path.splitext(p)[1].lower() in MEDIA_EXT:
                self._add_row(p)
        e.acceptProposedAction()


# ---------------------------- Готовые файлы -------------------------------


class CompletedPanel(QGroupBox):
    sig_selected = pyqtSignal(str)
    sig_removed = pyqtSignal(list)

    COLUMNS = ("completed.col.name", "completed.col.size")

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 18, 8, 8)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_remove = QPushButton()
        self.btn_clear = QPushButton()
        for b in (self.btn_remove, self.btn_clear):
            b.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        # Компактнее: минимум 70, максимум 110
        self.table.setMinimumHeight(70)
        self.table.setMaximumHeight(110)
        lay.addWidget(self.table)

        self.hint = QLabel()
        self.hint.setStyleSheet("color:#6f747c; font-size: 11px;")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.hint)

        self.table.itemSelectionChanged.connect(self._on_select)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_clear.clicked.connect(self._on_clear)
        self.retranslate()

    def retranslate(self):
        self.setTitle(tr("completed.group"))
        self.btn_remove.setText(tr("files.remove"))
        self.btn_clear.setText(tr("files.clear"))
        for i, key in enumerate(self.COLUMNS):
            self.table.setHorizontalHeaderItem(i, QTableWidgetItem(tr(key)))
        self.hint.setText(tr("completed.hint"))

    def add(self, path: str) -> bool:
        if self.has_path(path):
            return False
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        r = self.table.rowCount()
        self.table.insertRow(r)
        it = QTableWidgetItem(os.path.basename(path))
        it.setData(Qt.ItemDataRole.UserRole, path)
        it.setToolTip(path)
        self.table.setItem(r, 0, it)
        self.table.setItem(r, 1, QTableWidgetItem(_fmt_size_bytes(size)))
        return True

    def has_path(self, path: str) -> bool:
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == path:
                return True
        return False

    def paths(self) -> list:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def selected_paths(self) -> list:
        rows = self.table.selectionModel().selectedRows()
        out = []
        for row in rows:
            it = self.table.item(row.row(), 0)
            if it:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def select_path(self, path: str):
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == path:
                self.table.selectRow(r)
                return

    def clear(self):
        self.table.setRowCount(0)

    def _on_select(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        it = self.table.item(rows[0].row(), 0)
        if it:
            p = it.data(Qt.ItemDataRole.UserRole)
            if p:
                self.sig_selected.emit(p)

    def _on_remove(self):
        paths = self.selected_paths()
        if not paths:
            QMessageBox.warning(self, tr("dialog.warning"),
                                tr("completed.remove_nothing"))
            return
        if QMessageBox.question(self, tr("completed.group"),
                                tr("completed.remove_confirm")) \
                != QMessageBox.StandardButton.Yes:
            return
        for p in paths:
            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it and it.data(Qt.ItemDataRole.UserRole) == p:
                    self.table.removeRow(r)
                    break
        self.sig_removed.emit(paths)

    def _on_clear(self):
        paths = self.paths()
        if not paths:
            return
        if QMessageBox.question(self, tr("completed.group"),
                                tr("completed.clear_confirm")) \
                != QMessageBox.StandardButton.Yes:
            return
        self.clear()
        self.sig_removed.emit(paths)


# ---------------------------- Модель --------------------------------------


class DownloadDialog(QDialog):
    def __init__(self, title: str, about: str, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        self.lbl_about = QLabel(about)
        self.lbl_about.setWordWrap(True)
        lay.addWidget(self.lbl_about)
        self.lbl_stage = QLabel("…")
        self.lbl_stage.setStyleSheet("color:#8a8f97; font-size: 11px;")
        self.lbl_stage.setWordWrap(True)
        lay.addWidget(self.lbl_stage)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        lay.addWidget(self.bar)
        self.lbl_bytes = QLabel("")
        self.lbl_bytes.setStyleSheet("color:#c9cdd3; font-size: 11px;")
        lay.addWidget(self.lbl_bytes)

    def set_stage(self, txt: str):
        self.lbl_stage.setText(txt)

    def set_progress(self, percent: int, done: int, total: int):
        self.bar.setValue(max(0, min(100, int(percent))))
        if total > 0:
            self.lbl_bytes.setText(
                f"{_fmt_bytes(done)} / {_fmt_bytes(total)}  "
                f"({int(percent)}%)")
        else:
            self.lbl_bytes.setText(_fmt_bytes(done))


class ModelPanel(QGroupBox):
    def __init__(self):
        super().__init__()
        self._dl_thread: ModelDownloadThread | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 18, 8, 8)
        lay.setSpacing(6)

        self.combo = QComboBox()
        self.combo.setMinimumHeight(28)
        lay.addWidget(self.combo)

        info = QFrame()
        info.setObjectName("modelInfo")
        info.setStyleSheet("""
            QFrame#modelInfo {
                background-color: #141619;
                border: 1px solid #2b2f36;
                border-radius: 6px;
            }
        """)
        g = QGridLayout(info)
        g.setContentsMargins(10, 8, 10, 8)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(4)

        self.k_size = QLabel(); self.v_size = QLabel()
        self.k_qual = QLabel(); self.v_qual = QLabel()
        self.k_ram  = QLabel(); self.v_ram  = QLabel()
        self.k_stat = QLabel(); self.v_stat = QLabel()
        for k in (self.k_size, self.k_qual, self.k_ram, self.k_stat):
            k.setStyleSheet("color:#8a8f97; font-size: 11px;")
        for v in (self.v_size, self.v_qual, self.v_ram, self.v_stat):
            v.setStyleSheet("color:#e6e6e6; font-size: 11px;")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        g.addWidget(self.k_size, 0, 0); g.addWidget(self.v_size, 0, 1)
        g.addWidget(self.k_qual, 0, 2); g.addWidget(self.v_qual, 0, 3)
        g.addWidget(self.k_ram,  1, 0); g.addWidget(self.v_ram,  1, 1)
        g.addWidget(self.k_stat, 1, 2); g.addWidget(self.v_stat, 1, 3)
        g.setColumnStretch(1, 1); g.setColumnStretch(3, 1)
        lay.addWidget(info)

        act = QHBoxLayout()
        act.setSpacing(4)
        self.btn_action = QPushButton()
        self.btn_open_dir = QPushButton()
        for b in (self.btn_action, self.btn_open_dir):
            b.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            act.addWidget(b)
        act.addStretch(1)
        lay.addLayout(act)

        custom = QHBoxLayout()
        custom.setSpacing(4)
        self.edit_custom = QLineEdit()
        self.edit_custom.setStyleSheet("padding: 3px 6px; font-size: 11px;")
        self.btn_custom = QPushButton()
        self.btn_custom.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        custom.addWidget(self.edit_custom, 1)
        custom.addWidget(self.btn_custom)
        lay.addLayout(custom)

        self.combo.currentIndexChanged.connect(self._on_changed)
        self.btn_action.clicked.connect(self._on_action)
        self.btn_open_dir.clicked.connect(self._open_models_folder)
        self.btn_custom.clicked.connect(self._on_custom_add)

        self.retranslate()
        self.refresh_models()

    def refresh_models(self):
        cur = self.combo.currentData()
        cur_name = cur.get("name") if cur else SETTINGS.last_model
        self.combo.blockSignals(True)
        self.combo.clear()
        for m in list_models():
            self.combo.addItem(_dot_icon(m["installed"]), m["name"], m)
        idx = 0
        for i in range(self.combo.count()):
            d = self.combo.itemData(i)
            if d and d.get("name") == cur_name:
                idx = i
                break
        self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)
        self._on_changed(idx)

    def _on_changed(self, _idx):
        d = self.combo.currentData()
        if not d:
            return
        SETTINGS.last_model = d["name"]
        SETTINGS.save()
        if d["size"] > 0:
            self.v_size.setText(f'{d["size"]} МБ' if TR.language() == "ru"
                                else f'{d["size"]} MB')
        else:
            self.v_size.setText("—")
        self.v_qual.setText(quality_label(d["qual"]))
        self.v_ram.setText(d["ram"])
        if d["installed"]:
            self.v_stat.setText(tr("models.downloaded"))
            self.v_stat.setStyleSheet("color:#4ade80; font-size: 11px;")
            self.btn_action.setText(tr("models.delete"))
        else:
            self.v_stat.setText(tr("models.not_downloaded"))
            self.v_stat.setStyleSheet("color:#9aa0a8; font-size: 11px;")
            self.btn_action.setText(tr("models.download"))

    def retranslate(self):
        self.setTitle(tr("models.group"))
        self.k_size.setText(tr("models.info.size"))
        self.k_qual.setText(tr("models.info.quality"))
        self.k_ram.setText(tr("models.info.ram"))
        self.k_stat.setText(tr("models.info.status"))
        self.btn_open_dir.setText(tr("models.open_folder"))
        self.edit_custom.setPlaceholderText(tr("models.custom_placeholder"))
        self.btn_custom.setText(tr("models.add_custom"))
        self._on_changed(self.combo.currentIndex())

    def _on_action(self):
        d = self.combo.currentData()
        if not d:
            return
        if d["installed"]:
            self._remove_current(d)
        else:
            self._download_current(d)

    def _download_current(self, d: dict):
        need_mb = int(d.get("size") or 0)
        if need_mb:
            try:
                free_mb = shutil.disk_usage(str(MODELS_DIR)).free // (1024 * 1024)
                if free_mb < need_mb + 200:
                    QMessageBox.warning(
                        self, tr("dialog.warning"),
                        tr("models.dl.no_space", need=need_mb, free=free_mb))
                    return
            except Exception:
                pass

        dlg = DownloadDialog(
            tr("models.dl.title"),
            tr("models.dl.about", name=d["name"]),
            self,
        )
        th = ModelDownloadThread(d["repo"], d["name"])
        th.sig_file.connect(
            lambda rel, i, n: dlg.set_stage(tr("dl.file", i=i, n=n, rel=rel)))
        th.sig_progress.connect(dlg.set_progress)

        def _done():
            dlg.accept()
            self.refresh_models()

        def _err(msg):
            dlg.reject()
            QMessageBox.critical(self, tr("dialog.error"),
                                 tr("models.dl.fail", err=msg))

        th.sig_done.connect(_done)
        th.sig_error.connect(_err)
        th.finished.connect(th.deleteLater)
        self._dl_thread = th
        th.start()
        dlg.exec()

    def _remove_current(self, d: dict):
        if QMessageBox.question(
                self, tr("models.group"),
                tr("models.rm.confirm", name=d["name"], size=d["size"])
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            remove_model(d["name"])
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"),
                                 tr("models.rm.fail", err=str(e)))
        self.refresh_models()

    def _open_models_folder(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            os.startfile(str(MODELS_DIR))
        except Exception:
            QMessageBox.information(self, tr("dialog.info"), str(MODELS_DIR))

    def _on_custom_add(self):
        repo = self.edit_custom.text().strip()
        if not repo or "/" not in repo:
            QMessageBox.warning(self, tr("dialog.warning"),
                                tr("models.custom.invalid"))
            return
        name = repo.split("/")[-1]
        for m in list_models():
            if m["name"] == name:
                QMessageBox.warning(self, tr("dialog.warning"),
                                    tr("models.custom.exists"))
                return
        SETTINGS.custom_models.append({
            "repo": repo, "name": name, "size": 0, "qual": 3.0, "ram": "?",
        })
        SETTINGS.save()
        self.edit_custom.clear()
        self.refresh_models()


# ---------------------------- Устройство ----------------------------------

_DEVICE_RADIO_QSS = """
QRadioButton {
    font-size: 12px;
    padding: 6px 10px;
    border: 1px solid #333a44;
    border-radius: 6px;
    background-color: #1e2126;
    color: #c9cdd3;
}
QRadioButton:hover { background-color: #262a30; }
QRadioButton:checked {
    background-color: #2b3844;
    border: 2px solid #d97a1f;
    color: #ffb46b;
    font-weight: 700;
}
QRadioButton::indicator { width: 12px; height: 12px; }
QRadioButton:disabled { color: #5f6571; background-color: #1a1c20; }
"""


class CudaDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dl_thread: CudaDownloadThread | None = None
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setWindowTitle(tr("device.cuda_dialog_title"))
        lay = QVBoxLayout(self)
        self.lbl = QLabel(tr("device.cuda_dialog_about"))
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#c9cdd3; font-size: 11px;")
        lay.addWidget(self.lbl_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)
        btns = QHBoxLayout()
        self.btn_dl = QPushButton(tr("device.cuda_download"))
        self.btn_ck = QPushButton(tr("device.cuda_check"))
        self.btn_rm = QPushButton(tr("device.cuda_remove"))
        btns.addWidget(self.btn_dl)
        btns.addWidget(self.btn_ck)
        btns.addWidget(self.btn_rm)
        btns.addStretch(1)
        lay.addLayout(btns)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        self.btn_dl.clicked.connect(self._on_download)
        self.btn_ck.clicked.connect(self._on_check)
        self.btn_rm.clicked.connect(self._on_remove)
        self._update_status()

    def _update_status(self):
        name, drv, drv_raw = query_gpu()
        if name:
            self.lbl_status.setText(f"GPU: {name} · driver {drv_raw}")
        else:
            self.lbl_status.setText(tr("device.gpu_not_found"))
        if cuda_installed():
            self.lbl_status.setText(self.lbl_status.text() + "  •  " +
                                    tr("device.cuda_installed"))

    def _on_download(self):
        try:
            usage = shutil.disk_usage(str(CUDA_DIR))
            if usage.free < 600 * 1024 * 1024:
                QMessageBox.warning(self, tr("dialog.warning"),
                                    tr("device.cuda.no_disk"))
                return
        except Exception:
            pass

        dlg = DownloadDialog(tr("device.cuda.dl.title"),
                             tr("device.cuda.dl.about"), self)
        th = CudaDownloadThread()
        th.sig_status.connect(dlg.set_stage)
        th.sig_progress.connect(dlg.set_progress)

        def _done():
            dlg.accept()
            self._update_status()

        def _err(msg):
            dlg.reject()
            QMessageBox.critical(self, tr("dialog.error"),
                                 tr("device.cuda.dl.fail", err=msg))

        th.sig_done.connect(_done)
        th.sig_error.connect(_err)
        th.finished.connect(th.deleteLater)
        self._dl_thread = th
        th.start()
        dlg.exec()

    def _on_check(self):
        ok, info = cuda_check_load()
        if not ok:
            QMessageBox.warning(self, tr("dialog.warning"),
                                tr("device.cuda.check.fail", err=info))

    def _on_remove(self):
        remove_cuda()
        self._update_status()


class DevicePanel(QGroupBox):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 18, 8, 8)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.radio_cpu = QRadioButton()
        self.radio_gpu = QRadioButton()
        self.radio_cpu.setStyleSheet(_DEVICE_RADIO_QSS)
        self.radio_gpu.setStyleSheet(_DEVICE_RADIO_QSS)
        grp = QButtonGroup(self)
        grp.addButton(self.radio_cpu, 0)
        grp.addButton(self.radio_gpu, 1)
        self.radio_cpu.setChecked(True)
        row.addWidget(self.radio_cpu, 1)
        row.addWidget(self.radio_gpu, 1)
        lay.addLayout(row)

        self.lbl_state = QLabel()
        self.lbl_state.setStyleSheet("color:#e07a4a; font-size: 11px;")
        self.lbl_state.setWordWrap(True)
        lay.addWidget(self.lbl_state)

        btn = QHBoxLayout()
        btn.setSpacing(4)
        self.btn_dl = QPushButton()
        self.btn_ck = QPushButton()
        self.btn_rm = QPushButton()
        for b in (self.btn_dl, self.btn_ck, self.btn_rm):
            b.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            btn.addWidget(b)
        btn.addStretch(1)
        lay.addLayout(btn)

        self.btn_dl.clicked.connect(self._open_cuda_dialog)
        self.btn_ck.clicked.connect(self._on_check)
        self.btn_rm.clicked.connect(self._on_remove)

        self.retranslate()
        self.refresh_state()

    def retranslate(self):
        self.setTitle(tr("device.group"))
        self.radio_cpu.setText(tr("device.cpu"))
        self.radio_gpu.setText(tr("device.gpu"))
        self.btn_dl.setText(tr("device.cuda_download"))
        self.btn_ck.setText(tr("device.cuda_check"))
        self.btn_rm.setText(tr("device.cuda_remove"))
        self.refresh_state()

    def refresh_state(self):
        name, drv, drv_raw = query_gpu()
        installed = cuda_installed()

        if name is None:
            self.radio_gpu.setEnabled(False)
            self.radio_gpu.setToolTip(tr("device.gpu_not_found"))
            self.lbl_state.setText(tr("device.gpu_not_found"))
            self.radio_cpu.setChecked(True)
        elif drv and drv < CUDA_MIN_DRIVER:
            self.radio_gpu.setEnabled(False)
            self.radio_gpu.setToolTip(tr("device.gpu_driver_old", v=drv_raw))
            self.lbl_state.setText(tr("device.gpu_driver_old", v=drv_raw))
            self.radio_cpu.setChecked(True)
            SETTINGS.cuda_may_not_work = True
            SETTINGS.save()
        else:
            self.radio_gpu.setEnabled(True)
            self.radio_gpu.setToolTip("")
            if installed:
                self.lbl_state.setText(tr("device.status.gpu_ok",
                                          name=name, drv=drv_raw))
            else:
                self.lbl_state.setText(tr("device.cuda_not_installed"))
            if SETTINGS.device == "cuda":
                self.radio_gpu.setChecked(True)
        if SETTINGS.device == "cpu":
            self.radio_cpu.setChecked(True)

    def current_device(self) -> str:
        return "cuda" if self.radio_gpu.isChecked() else "cpu"

    def _open_cuda_dialog(self):
        dlg = CudaDialog(self)
        dlg.exec()
        self.refresh_state()

    def _on_check(self):
        ok, info = cuda_check_load()
        if not ok:
            QMessageBox.warning(self, tr("dialog.warning"),
                                tr("device.cuda.check.fail", err=info))

    def _on_remove(self):
        remove_cuda()
        self.refresh_state()


# ---------------------------- Процесс -------------------------------------


class CurrentFileBox(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("currentFileBox")
        self.setStyleSheet("""
            QFrame#currentFileBox {
                background-color: rgba(217, 122, 31, 0.08);
                border: 1px solid rgba(217, 122, 31, 0.35);
                border-radius: 6px;
            }
            QFrame#currentFileBox QLabel {
                background: transparent;
                color: #ffb46b;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        self.lbl_index = QLabel()
        self.lbl_index.setStyleSheet("color:#ffb46b; font-size: 11px; font-weight: 600;")
        self.lbl_name = QLabel()
        self.lbl_name.setStyleSheet("color:#ffdba8; font-size: 13px; font-weight: 600;")
        lay.addWidget(self.lbl_index, 0)
        lay.addWidget(self.lbl_name, 1)
        self.set_current(0, 0, "")

    def set_current(self, n, m, name):
        if n == 0 and not name:
            self.lbl_index.setText(tr("progress.current_sub"))
            self.lbl_name.setText(tr("progress.current_empty"))
            self.lbl_name.setStyleSheet(
                "color:#6f747c; font-size: 13px; font-weight: 500;")
        else:
            self.lbl_index.setText(tr("progress.file", n=n, m=m))
            self.lbl_name.setText(name)
            self.lbl_name.setStyleSheet(
                "color:#ffdba8; font-size: 13px; font-weight: 600;")


class ProgressPanel(QGroupBox):
    sig_start = pyqtSignal()
    sig_stop = pyqtSignal()
    sig_pause = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 20, 10, 10)
        lay.setSpacing(8)

        self.current_box = CurrentFileBox()
        lay.addWidget(self.current_box)

        self.black_hole = BlackHoleWidget()
        lay.addWidget(self.black_hole)

        self.bar_file = QProgressBar()
        self.bar_file.setObjectName("flameBar")
        self.bar_file.setRange(0, 1000)
        self.bar_queue = QProgressBar()
        self.bar_queue.setObjectName("flameBar")
        self.bar_queue.setRange(0, 1000)
        lay.addWidget(self.bar_file)
        lay.addWidget(self.bar_queue)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        self.lbl_stage = QLabel()
        self.lbl_audio = QLabel()
        self.lbl_elapsed = QLabel()
        self.lbl_eta = QLabel()
        self.lbl_speed = QLabel()
        for w in (self.lbl_stage, self.lbl_audio,
                  self.lbl_elapsed, self.lbl_eta, self.lbl_speed):
            w.setStyleSheet("color:#c9cdd3; font-size: 11px;")
        grid.addWidget(self.lbl_stage, 0, 0, 1, 2)
        grid.addWidget(self.lbl_audio, 1, 0, 1, 2)
        grid.addWidget(self.lbl_elapsed, 2, 0)
        grid.addWidget(self.lbl_eta, 2, 1)
        grid.addWidget(self.lbl_speed, 3, 0, 1, 2)
        lay.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_start = QPushButton()
        self.btn_pause = QPushButton()
        self.btn_stop = QPushButton()
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        lay.addLayout(row)

        self._pause_effect = QGraphicsOpacityEffect(self.btn_pause)
        self._pause_effect.setOpacity(1.0)
        self.btn_pause.setGraphicsEffect(self._pause_effect)
        self._pause_anim = QPropertyAnimation(self._pause_effect, b"opacity", self)
        self._pause_anim.setDuration(900)
        self._pause_anim.setStartValue(1.0)
        self._pause_anim.setKeyValueAt(0.5, 0.35)
        self._pause_anim.setEndValue(1.0)
        self._pause_anim.setLoopCount(-1)
        self._pause_anim.setEasingCurve(QEasingCurve.Type.InOutSine)

        self._paused = False
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._pause_toggle)
        self.btn_stop.clicked.connect(self._stop)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._elapsed_start = 0.0
        self._running = False

        self.retranslate()

    def retranslate(self):
        self.setTitle(tr("progress.group"))
        self.btn_start.setText(tr("progress.start"))
        self.btn_stop.setText(tr("progress.stop"))
        self.btn_pause.setText(tr("progress.resume") if self._paused
                               else tr("progress.pause"))
        self.lbl_stage.setText(tr("progress.stage",
                                  stage=tr("progress.stage.idle")))
        self.lbl_audio.setText(tr("progress.audio",
                                  cur="00:00:00", total="00:00:00"))
        self.lbl_elapsed.setText(tr("progress.elapsed", t="00:00:00"))
        self.lbl_eta.setText(tr("progress.eta", t="--:--:--"))
        self.lbl_speed.setText(tr("progress.speed", x="0.00"))
        self.current_box.set_current(0, 0, "")

    def set_current(self, n: int, m: int, name: str):
        self.current_box.set_current(n, m, name)

    def set_stage(self, stage_key: str):
        self.lbl_stage.setText(tr("progress.stage", stage=tr(stage_key)))

    def set_file_progress(self, frac: float):
        self.bar_file.setValue(int(max(0.0, min(1.0, float(frac))) * 1000))

    def set_queue_progress(self, frac: float):
        self.bar_queue.setValue(int(max(0.0, min(1.0, frac)) * 1000))

    def set_audio_time(self, cur: float, total: float):
        self.lbl_audio.setText(tr("progress.audio",
                                  cur=_fmt_hms(cur), total=_fmt_hms(total)))

    def set_speed_eta(self, speed_x: float, eta_sec: float):
        self.lbl_speed.setText(tr("progress.speed", x=f"{speed_x:.2f}"))
        self.lbl_eta.setText(tr("progress.eta",
                                t=_fmt_hms(eta_sec) if eta_sec >= 0 else "--:--:--"))

    def _start(self):
        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._paused = False
        self.btn_pause.setText(tr("progress.pause"))
        self._pause_anim.stop()
        self._pause_effect.setOpacity(1.0)
        self.bar_file.setValue(0)
        self.bar_queue.setValue(0)
        self._elapsed_start = monotonic()
        self._elapsed_timer.start()
        self.black_hole.start()
        self.sig_start.emit()

    def _pause_toggle(self):
        self._paused = not self._paused
        if self._paused:
            self.black_hole.pause()
            self.btn_pause.setText(tr("progress.resume"))
            self._pause_anim.start()
            self.sig_pause.emit(True)
        else:
            self.black_hole.resume()
            self.btn_pause.setText(tr("progress.pause"))
            self._pause_anim.stop()
            self._pause_effect.setOpacity(1.0)
            self.sig_pause.emit(False)

    def _stop(self):
        self._running = False
        self._elapsed_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self._paused = False
        self.btn_pause.setText(tr("progress.pause"))
        self._pause_anim.stop()
        self._pause_effect.setOpacity(1.0)
        self.black_hole.stop()
        self.sig_stop.emit()

    def force_stop_ui(self):
        self._running = False
        self._elapsed_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self._paused = False
        self.btn_pause.setText(tr("progress.pause"))
        self._pause_anim.stop()
        self._pause_effect.setOpacity(1.0)
        self.black_hole.stop()

    def _tick_elapsed(self):
        if not self._running:
            return
        t = monotonic() - self._elapsed_start
        self.lbl_elapsed.setText(tr("progress.elapsed", t=_fmt_hms(t)))


# ---------------------------- Результат -----------------------------------


class ResultPanel(QGroupBox):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 18, 8, 8)
        self.edit = QPlainTextEdit()
        self.edit.setMinimumHeight(200)
        lay.addWidget(self.edit, 1)
        row = QHBoxLayout()
        self.chk = QCheckBox()
        self.btn_copy = QPushButton()
        row.addWidget(self.chk)
        row.addStretch(1)
        row.addWidget(self.btn_copy)
        lay.addLayout(row)
        self.btn_copy.clicked.connect(self._copy)
        self._segments: list = []
        self.retranslate()

    def retranslate(self):
        self.setTitle(tr("result.group"))
        self.chk.setText(tr("result.show_timecodes"))
        self.btn_copy.setText(tr("result.copy_all"))
        self.edit.setPlaceholderText(tr("result.placeholder"))

    def set_text_from_segments(self, segments: list):
        self._segments = list(segments)
        self._rebuild()

    def set_plain_text(self, txt: str):
        self.edit.setPlainText(txt)

    def clear(self):
        self._segments = []
        self.edit.clear()

    def _rebuild(self):
        if self.chk.isChecked():
            txt = "\n".join(
                f"[{_fmt_hms(s['start'])} → {_fmt_hms(s['end'])}] {s['text']}"
                for s in self._segments
            )
        else:
            txt = "\n".join(s["text"] for s in self._segments)
        self.edit.setPlainText(txt)

    def _copy(self):
        QGuiApplication.clipboard().setText(self.edit.toPlainText())


# ---------------------------- Экспорт -------------------------------------


class ExportPanel(QGroupBox):
    FORMATS = ["TXT", "SRT", "VTT", "JSON", "TSV"]

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 18, 10, 8)
        lay.setSpacing(8)

        self.lbl_fmt = QLabel()
        self.combo = QComboBox()
        self.combo.addItems(self.FORMATS)
        self.combo.setStyleSheet("padding: 3px 6px; font-size: 11px;")
        self.combo.setMinimumWidth(80)
        lay.addWidget(self.lbl_fmt)
        lay.addWidget(self.combo)

        self.btn_save = QPushButton()
        self.btn_save_all = QPushButton()
        for b in (self.btn_save, self.btn_save_all):
            b.setStyleSheet("padding: 4px 10px; font-size: 11px;")
            lay.addWidget(b)
        lay.addStretch(1)

        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_all.clicked.connect(self._on_save_all)
        self.retranslate()

    def retranslate(self):
        self.setTitle(tr("export.group"))
        self.lbl_fmt.setText(tr("export.format"))
        self.btn_save.setText(tr("export.save_as"))
        self.btn_save_all.setText(tr("export.save_all"))

    def _main(self) -> "MainWindow":
        return self.window()  # type: ignore

    @staticmethod
    def _unique_path(folder: Path, stem: str, ext: str) -> Path:
        p = folder / f"{stem}.{ext}"
        if not p.exists():
            return p
        i = 2
        while True:
            p = folder / f"{stem} ({i}).{ext}"
            if not p.exists():
                return p
            i += 1

    @staticmethod
    def _save_one(fmt: str, segs: list, path: Path):
        try:
            export_result(fmt, segs, str(path))
            return None
        except Exception as e:
            LOG.exception("export failed: %s", path)
            return f"{path.name}: {e}"

    def _on_save(self):
        main = self._main()
        main._flush_current_edit()

        path = main.current_result_path
        if path is None:
            QMessageBox.warning(self, tr("dialog.warning"), tr("export.no_text"))
            return
        segs = main.get_export_segments(path)
        if not segs:
            QMessageBox.warning(self, tr("dialog.warning"), tr("export.no_text"))
            return

        fmt = self.combo.currentText()
        default_dir = Path(SETTINGS.export_dir or OUTPUT_DIR)
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"{Path(path).stem}.{fmt.lower()}"

        out_path, _ = QFileDialog.getSaveFileName(
            self, tr("export.save_as"),
            str(default_dir / default_name),
            f"{fmt} (*.{fmt.lower()})",
        )
        if not out_path:
            return
        err = self._save_one(fmt, segs, Path(out_path))
        if err is None:
            SETTINGS.export_dir = str(Path(out_path).parent)
            SETTINGS.save()
        else:
            QMessageBox.critical(self, tr("dialog.error"), err)

    def _on_save_all(self):
        main = self._main()
        main._flush_current_edit()

        paths = main.completed_panel.paths()
        if not paths:
            QMessageBox.warning(self, tr("dialog.warning"), tr("export.no_text"))
            return
        d = QFileDialog.getExistingDirectory(
            self, tr("export.choose_dir"),
            SETTINGS.export_dir or str(OUTPUT_DIR))
        if not d:
            return
        folder = Path(d)
        fmt = self.combo.currentText()
        errors = []
        for p in paths:
            segs = main.get_export_segments(p)
            if not segs:
                continue
            out = self._unique_path(folder, Path(p).stem, fmt.lower())
            err = self._save_one(fmt, segs, out)
            if err:
                errors.append(err)
        SETTINGS.export_dir = d
        SETTINGS.save()
        if errors:
            QMessageBox.critical(self, tr("dialog.error"), "\n".join(errors))


# =========================================================================
#                         ДИАЛОГ ПАПКИ ДАННЫХ
# =========================================================================


class RootFolderDialog(QDialog):
    def __init__(self, default_dir: Path, parent=None):
        super().__init__(parent)
        self._path: Path = Path(default_dir)
        self.setModal(True)
        self.setMinimumWidth(580)
        self.setWindowTitle(tr("root.dialog.title"))
        lay = QVBoxLayout(self)

        lbl = QLabel(tr("root.dialog.about"))
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        row = QHBoxLayout()
        self.edit = QLineEdit(str(default_dir))
        self.edit.setReadOnly(True)
        btn_browse = QPushButton(tr("root.dialog.browse"))
        btn_browse.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(btn_browse)
        lay.addLayout(row)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._on_ok)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, tr("root.dialog.title"), self.edit.text())
        if d:
            self.edit.setText(d)

    def _on_ok(self):
        p = Path(self.edit.text().strip())
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, tr("dialog.error"), str(e))
            return
        if not os.access(p, os.W_OK):
            QMessageBox.warning(self, tr("dialog.warning"), tr("root.invalid"))
            return
        self._path = p
        self.accept()

    def chosen_path(self) -> Path:
        return self._path


def ask_root_dialog(default_dir: Path):
    dlg = RootFolderDialog(default_dir)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        return dlg.chosen_path()
    return None


# =========================================================================
#                            DARK QSS
# =========================================================================

DARK_QSS = """
QWidget { background-color: #1b1d21; color: #e6e6e6; font-size: 12px; }
QMainWindow, QDialog { background-color: #16181c; }
QGroupBox {
    border: 1px solid #2b2f36; border-radius: 6px;
    margin-top: 14px; padding-top: 8px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px;
    padding: 0 4px; color: #ffb46b;
}
QPushButton {
    background-color: #262a30; border: 1px solid #333a44;
    border-radius: 4px; padding: 6px 12px;
}
QPushButton:hover { background-color: #30353d; }
QPushButton:pressed { background-color: #20242a; }
QPushButton:disabled { color: #7a7f87; background-color: #20242a; }
QTableWidget, QPlainTextEdit, QComboBox, QLineEdit {
    background-color: #141619; border: 1px solid #2b2f36;
    border-radius: 4px; selection-background-color: #3a4657;
}
QComboBox { padding: 4px 8px; }
QComboBox::drop-down { border: 0; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #141619;
    border: 1px solid #2b2f36;
    selection-background-color: #3a4657;
    padding: 4px;
}
QHeaderView::section {
    background-color: #20242a; border: 0;
    border-bottom: 1px solid #2b2f36;
    padding: 4px 6px; color: #c9cdd3;
}
QProgressBar {
    background-color: #141619; border: 1px solid #2b2f36;
    border-radius: 4px; text-align: center; color: #ffffff;
    min-height: 16px;
}
QProgressBar#flameBar::chunk {
    background-color: qlineargradient(x1:1, y1:0, x2:0, y2:0,
        stop:0    #ffe89a,
        stop:0.18 #ffd35a,
        stop:0.42 #ff8c1a,
        stop:0.70 #e04a00,
        stop:1    #7a1a00);
    border-radius: 3px;
}
QToolBar { background-color: #16181c; border: 0; spacing: 6px; padding: 4px; }
QMenuBar, QMenu { background-color: #16181c; }
QMenuBar::item:selected, QMenu::item:selected { background-color: #2b2f36; }
QSplitter::handle { background-color: #2b2f36; }
QScrollBar:vertical { background: #141619; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #333a44; border-radius: 4px; min-height: 24px; }
QScrollBar:horizontal { background: #141619; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #333a44; border-radius: 4px; min-width: 24px; }
QScrollArea { border: 0; background: transparent; }
"""


def _wrap_scroll(widget):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    sa.setWidget(widget)
    return sa


# =========================================================================
#                            ГЛАВНОЕ ОКНО
# =========================================================================


def _segments_to_text(segments: list, with_tc: bool) -> str:
    if with_tc:
        return "\n".join(
            f"[{_fmt_hms(s['start'])} → {_fmt_hms(s['end'])}] {s['text']}"
            for s in segments
        )
    return "\n".join(s["text"] for s in segments)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1420, 900)
        self.worker: TranscribeWorker | None = None

        self.per_file_segments: dict = {}
        self.per_file_text: dict = {}
        self.per_file_edited: dict = {}
        self.current_result_path: str | None = None
        self._worker_files: list = []
        # путь файла, который сейчас транскрибируется (для live-текста)
        self._live_path: str | None = None

        tb = QToolBar("main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        self.menu_lang = QMenu(tr("menu.language"), self)
        self.act_ru = QAction(tr("menu.language.ru"), self, checkable=True)
        self.act_en = QAction(tr("menu.language.en"), self, checkable=True)
        grp = QActionGroup(self)
        grp.addAction(self.act_ru)
        grp.addAction(self.act_en)
        grp.setExclusive(True)
        self.menu_lang.addAction(self.act_ru)
        self.menu_lang.addAction(self.act_en)
        (self.act_en if TR.language() == "en" else self.act_ru).setChecked(True)
        self.act_ru.triggered.connect(lambda: TR.set_language("ru"))
        self.act_en.triggered.connect(lambda: TR.set_language("en"))

        btn_lang = QPushButton("🌐")
        btn_lang.setMenu(self.menu_lang)
        tb.addWidget(btn_lang)

        self.menu_root = QMenu(self)
        self.act_change_root = QAction(tr("menu.change_root"), self)
        self.act_change_root.triggered.connect(self._on_change_root)
        self.menu_root.addAction(self.act_change_root)
        btn_root = QPushButton("📁")
        btn_root.setToolTip(str(ROOT))
        btn_root.setMenu(self.menu_root)
        tb.addWidget(btn_root)
        self._btn_root = btn_root

        tb.addSeparator()

        self.lbl_title = QLabel()
        self.lbl_title.setStyleSheet("color:#9aa0a8; padding-left:8px;")
        tb.addWidget(self.lbl_title)

        self.file_panel = FilePanel()
        self.completed_panel = CompletedPanel()
        self.model_panel = ModelPanel()
        self.device_panel = DevicePanel()
        self.progress_panel = ProgressPanel()
        self.result_panel = ResultPanel()
        self.export_panel = ExportPanel()

        left_content = QWidget()
        ll = QVBoxLayout(left_content)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        ll.addWidget(self.file_panel)
        ll.addWidget(self.completed_panel)
        ll.addWidget(self.model_panel)
        ll.addWidget(self.device_panel)
        ll.addStretch(1)
        left_scroll = _wrap_scroll(left_content)
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(400)

        right_content = QWidget()
        rl = QVBoxLayout(right_content)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        rl.addWidget(self.progress_panel, 0)
        rl.addWidget(self.result_panel, 1)
        rl.addWidget(self.export_panel, 0)
        right_scroll = _wrap_scroll(right_content)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_scroll)
        split.addWidget(right_scroll)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 1090])
        self.setCentralWidget(split)

        TR.language_changed.connect(self._retranslate)
        self.progress_panel.sig_start.connect(self._on_start)
        self.progress_panel.sig_stop.connect(self._on_stop)
        self.progress_panel.sig_pause.connect(self._on_pause)
        self.completed_panel.sig_selected.connect(self._show_result_for)
        self.completed_panel.sig_removed.connect(self._on_completed_removed)
        self.result_panel.chk.toggled.connect(self._on_timecodes_toggled)

        self._retranslate()

    def _retranslate(self, *_):
        self.setWindowTitle(tr("app.title"))
        self.lbl_title.setText(tr("app.title"))
        self.statusBar().showMessage(tr("app.title"))
        self.menu_lang.setTitle(tr("menu.language"))
        self.act_ru.setText(tr("menu.language.ru"))
        self.act_en.setText(tr("menu.language.en"))
        self.act_change_root.setText(tr("menu.change_root"))
        for p in (self.file_panel, self.completed_panel, self.model_panel,
                  self.device_panel, self.progress_panel,
                  self.result_panel, self.export_panel):
            if hasattr(p, "retranslate"):
                p.retranslate()

    def _on_change_root(self):
        cur = Path(ROOT)
        chosen = ask_root_dialog(cur)
        if chosen is None or chosen == cur:
            return
        save_bootstrap_root(chosen)
        QMessageBox.information(self, tr("dialog.info"),
                                tr("root.change_restart"))

    # -------------------- работа с текстом --------------------
    def _auto_text_for(self, path: str) -> str:
        segs = self.per_file_segments.get(path, [])
        return _segments_to_text(segs, self.result_panel.chk.isChecked())

    def _flush_current_edit(self):
        p = self.current_result_path
        if p is None:
            return
        # не перезаписываем live-текст, который сейчас растёт в редакторе
        if p == self._live_path:
            return
        cur = self.result_panel.edit.toPlainText()
        auto = self._auto_text_for(p)
        self.per_file_text[p] = cur
        self.per_file_edited[p] = (cur != auto)

    def _show_result_for(self, path: str):
        self._flush_current_edit()
        self.current_result_path = path
        if self.per_file_edited.get(path, False):
            self.result_panel.set_plain_text(self.per_file_text.get(path, ""))
        else:
            segs = self.per_file_segments.get(path, [])
            self.result_panel.set_text_from_segments(segs)

    def _on_timecodes_toggled(self, checked: bool):
        p = self.current_result_path
        if p is None:
            return
        if self.per_file_edited.get(p, False):
            ans = QMessageBox.question(
                self, tr("dialog.warning"), tr("result.edited_warn"))
            if ans != QMessageBox.StandardButton.Yes:
                self.result_panel.chk.blockSignals(True)
                self.result_panel.chk.setChecked(not checked)
                self.result_panel.chk.blockSignals(False)
                return
        segs = self.per_file_segments.get(p, [])
        self.result_panel.set_text_from_segments(segs)
        self.per_file_edited[p] = False
        self.per_file_text[p] = _segments_to_text(segs, checked)

    def _on_completed_removed(self, paths: list):
        removed_current = self.current_result_path in paths
        for p in paths:
            self.per_file_segments.pop(p, None)
            self.per_file_text.pop(p, None)
            self.per_file_edited.pop(p, None)
        if removed_current:
            self.current_result_path = None
            self.result_panel.clear()
            remaining = self.completed_panel.paths()
            if remaining:
                self.completed_panel.select_path(remaining[0])

    def get_export_segments(self, path: str) -> list:
        if self.per_file_edited.get(path, False):
            txt = self.per_file_text.get(path, "")
            lines = [ln for ln in txt.splitlines() if ln.strip()]
            if not lines:
                return []
            orig = self.per_file_segments.get(path, [])
            end = orig[-1]["end"] if orig else float(len(lines))
            n = len(lines)
            return [
                {
                    "start": end * i / n,
                    "end": end * (i + 1) / n,
                    "text": ln,
                }
                for i, ln in enumerate(lines)
            ]
        return list(self.per_file_segments.get(path, []))

    # -------------------- запуск воркера --------------------
    def _on_start(self):
        if self.worker and self.worker.isRunning():
            return
        files = self.file_panel.file_paths()
        if not files:
            QMessageBox.warning(self, tr("dialog.warning"), tr("queue.empty"))
            self.progress_panel.force_stop_ui()
            return

        d = self.model_panel.combo.currentData()
        if not d or not d["installed"]:
            QMessageBox.warning(self, tr("dialog.warning"),
                                tr("queue.no_installed_model"))
            self.progress_panel.force_stop_ui()
            return

        device = self.device_panel.current_device()
        if device == "cuda":
            if not cuda_installed():
                ans = QMessageBox.question(
                    self, tr("queue.cuda_missing_title"),
                    tr("queue.cuda_missing"),
                )
                if ans == QMessageBox.StandardButton.Yes:
                    dlg = CudaDialog(self)
                    dlg.exec()
                    if not cuda_installed():
                        self.device_panel.refresh_state()
                        self.progress_panel.force_stop_ui()
                        return
                else:
                    self.device_panel.radio_cpu.setChecked(True)
                    device = "cpu"
            activate_cuda()

        compute_type = "int8" if device == "cpu" else "float16"

        for r in range(self.file_panel.table.rowCount()):
            self.file_panel.set_status(r, "files.status.pending")

        self._worker_files = list(files)

        self.worker = TranscribeWorker(
            files=files,
            model_path=str(model_dir(d["name"])),
            device=device,
            compute_type=compute_type,
            language=None,
        )
        self.worker.sig_status.connect(self._on_status)
        self.worker.sig_stage.connect(self.progress_panel.set_stage)
        self.worker.sig_file_progress.connect(self.progress_panel.set_file_progress)
        self.worker.sig_queue_progress.connect(self.progress_panel.set_queue_progress)
        self.worker.sig_file_time.connect(self._on_file_time)
        self.worker.sig_speed.connect(self.progress_panel.set_speed_eta)
        self.worker.sig_partial_segments.connect(self._on_partial_segments)
        self.worker.sig_file_done.connect(self._on_file_done)
        self.worker.sig_file_error.connect(self._on_file_error)
        self.worker.sig_finished.connect(self._on_worker_finished)
        self.worker.sig_log.connect(LOG.info)

        self.worker.start()

    def _on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def _on_pause(self, paused: bool):
        if not self.worker:
            return
        if paused:
            self.worker.pause()
        else:
            self.worker.resume()

    def _path_from_idx(self, idx: int):
        if 0 <= idx < len(self._worker_files):
            return self._worker_files[idx]
        return None

    def _on_status(self, idx: int, key: str):
        path = self._path_from_idx(idx)
        if path is None:
            return
        row = self.file_panel.row_for_path(path)
        if row < 0:
            # файл мог быть удалён из очереди, но мы всё равно продолжаем
            pass
        else:
            self.file_panel.set_status(row, key)
        if key == "files.status.transcribing":
            name = os.path.basename(path)
            self.progress_panel.set_current(idx + 1,
                                            max(len(self._worker_files), 1),
                                            name)
            # Показываем live-текст этого файла в редакторе
            self._live_path = path
            self.current_result_path = path
            self.result_panel.clear()

    def _on_partial_segments(self, idx: int, segments: list):
        path = self._path_from_idx(idx)
        if path is None:
            return
        # Обновляем только если в редакторе открыт именно этот файл
        if self.current_result_path != path:
            return
        if self._live_path != path:
            return
        self.result_panel.set_text_from_segments(segments)

    def _on_file_time(self, idx: int, cur: float, total: float):
        self.progress_panel.set_audio_time(cur, total)

    def _on_file_done(self, idx: int, path: str, segments: list):
        # Финальные сегменты перетирают live-версию
        self.per_file_segments[path] = segments
        self.per_file_edited[path] = False
        self.per_file_text[path] = _segments_to_text(
            segments, self.result_panel.chk.isChecked())

        # Если это последний live-файл — оставляем его отображённым
        if self._live_path == path:
            self._live_path = None
            self.current_result_path = path
            self.result_panel.set_text_from_segments(segments)

        row = self.file_panel.row_for_path(path)
        if row >= 0:
            self.file_panel.table.removeRow(row)

        was_empty = (self.completed_panel.table.rowCount() == 0)
        self.completed_panel.add(path)

        if was_empty or self.current_result_path is None:
            self.completed_panel.select_path(path)

    def _on_file_error(self, idx: int, path: str, err: str):
        row = self.file_panel.row_for_path(path)
        if row >= 0:
            self.file_panel.set_status(row, "files.status.error")
        if self._live_path == path:
            self._live_path = None
        LOG.error("Ошибка файла %s: %s", path, err)
        QMessageBox.critical(self, tr("dialog.error"),
                             f"{Path(path).name}: {err}")

    def _on_worker_finished(self):
        self.progress_panel.force_stop_ui()
        self.progress_panel.set_queue_progress(1.0)
        self.progress_panel.set_current(0, 0, "")
        self._live_path = None

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(4000):
                self.worker.terminate()
                self.worker.wait(1500)

        for th in (getattr(self.model_panel, "_dl_thread", None),):
            if th is not None and th.isRunning():
                th.terminate()
                th.wait(1000)

        try:
            SETTINGS.device = self.device_panel.current_device()
            SETTINGS.save()
        except Exception:
            pass

        super().closeEvent(e)

        os._exit(0)


# =========================================================================
#                              MAIN
# =========================================================================


def main() -> int:
    # Windows: AppUserModelID — без него таскбар игнорирует иконку окна.
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "WhisperGUI.whispergui.1")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("WhisperGUI")
    app.setApplicationDisplayName("WhisperGUI")
    app.setStyleSheet(DARK_QSS)

    ic = load_app_icon()
    if not ic.isNull():
        app.setWindowIcon(ic)

    root = load_bootstrap_root()
    if root is None:
        chosen = ask_root_dialog(app_dir())
        if chosen is None:
            chosen = app_dir()
        save_bootstrap_root(chosen)
        root = chosen
    apply_root(root, create=True)

    setup_logging()
    SETTINGS.load()
    TR._lang = SETTINGS.language or "ru"
    LOG.info("Корневая папка данных: %s", ROOT)

    w = MainWindow()
    if not ic.isNull():
        w.setWindowIcon(ic)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())