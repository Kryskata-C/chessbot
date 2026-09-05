# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Windows: builds dist/ChessVision/ChessVision.exe
(one-folder build; zip the folder to distribute).

Run through packaging/build_win.ps1, which stages packaging/stage/stockfish.exe
first. The exe is windowed (no console); prints go to
%LOCALAPPDATA%\\Chess Vision\\chess-vision.log.
"""
import os

HERE = os.path.abspath(os.getcwd())
STAGE = os.path.join(HERE, "packaging", "stage")
stockfish = os.path.join(STAGE, "stockfish.exe")
if not os.path.isfile(stockfish):
    raise SystemExit("packaging/stage/stockfish.exe missing — run packaging/build_win.ps1")

hidden = [
    # supabase pulls these lazily
    "supabase", "supabase_auth", "supabase_functions", "postgrest", "storage3", "realtime",
    "httpx", "h11", "h2", "websockets", "websockets.legacy", "websockets.legacy.client",
    "deprecation", "pydantic", "pydantic_core",
    # Windows credential store
    "keyring", "keyring.backends", "keyring.backends.Windows", "win32ctypes", "win32ctypes.pywin32",
]
# Optional Windows OCR (requirements-win-ocr.txt); only bundled when installed.
try:
    import winrt.windows.media.ocr  # noqa: F401
    hidden += ["winrt", "winrt.windows.foundation", "winrt.windows.globalization",
               "winrt.windows.graphics.imaging", "winrt.windows.media.ocr",
               "winrt.windows.storage.streams"]
except Exception:
    pass

a = Analysis(
    ["main.py"],
    pathex=[HERE],
    binaries=[(stockfish, "bin")],  # not "." — it would shadow the `stockfish` Python package
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "jupyter",
              "Vision", "Quartz", "Foundation", "objc", "AppKit",
              "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets", "PyQt6.QtQml",
              "PyQt6.QtQuick", "PyQt6.QtMultimedia", "PyQt6.Qt3DCore",
              "PyQt6.QtBluetooth", "PyQt6.QtNfc", "PyQt6.QtPositioning",
              "PyQt6.QtSensors", "PyQt6.QtSerialPort", "PyQt6.QtSql",
              "PyQt6.QtTest", "PyQt6.QtXml", "PyQt6.QtDesigner", "PyQt6.QtHelp",
              "PyQt6.QtPdf", "PyQt6.QtPdfWidgets", "PyQt6.QtRemoteObjects",
              "PyQt6.QtSpatialAudio", "PyQt6.QtTextToSpeech", "PyQt6.QtWebChannel",
              "PyQt6.QtWebSockets", "PyQt6.QtOpenGL", "PyQt6.QtOpenGLWidgets",
              "PyQt6.QtSvgWidgets", "PyQt6.QtPrintSupport", "PyQt6.QtNetworkAuth",
              "PyQt6.QtDBus", "PyQt6.QtStateMachine", "PyQt6.QtGraphs", "PyQt6.QtGraphsWidgets",
              "PyQt6.QtHttpServer", "PyQt6.QtQuick3D", "PyQt6.QtQuickWidgets",
              "PyQt6.QtLocation", "PyQt6.QtCharts", "PyQt6.QtDataVisualization"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChessVision",
    icon="assets/ChessVision.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ChessVision")
