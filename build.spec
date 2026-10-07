# build.spec — сборка WhisperGUI (onedir, без CUDA)
# Запуск:  pyinstaller build.spec

import sys
from pathlib import Path
import imageio_ffmpeg
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
ROOT = Path(SPECPATH)

ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

ICON_FILE = ROOT / "whispergui.ico"
icon_arg = str(ICON_FILE) if ICON_FILE.exists() else None

# datas: VAD-модель faster-whisper + иконка
datas = []
datas += collect_data_files("faster_whisper")
if ICON_FILE.exists():
    datas.append((str(ICON_FILE), "."))

a = Analysis(
    ["whispergui.py"],
    pathex=[str(ROOT)],
    binaries=[(ffmpeg_bin, ".")],
    datas=datas,
    hiddenimports=[
        "faster_whisper", "faster_whisper.assets",
        "ctranslate2", "tokenizers",
        "huggingface_hub", "soundfile", "imageio_ffmpeg", "av",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "pandas", "scipy",
        "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="WhisperGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon=icon_arg,                 # ← иконка самого exe
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="WhisperGUI",
)