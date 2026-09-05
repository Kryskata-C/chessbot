# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: builds dist/Chess Vision.app.

Run through packaging/build_app.sh, which stages the Stockfish binary into
packaging/stage/ first. The app is windowed (no terminal); prints go to
~/Library/Application Support/Chess Vision/chess-vision.log.
"""
import os

HERE = os.path.abspath(os.getcwd())
STAGE = os.path.join(HERE, "packaging", "stage")
stockfish = os.path.join(STAGE, "stockfish")
if not os.path.isfile(stockfish):
    raise SystemExit("packaging/stage/stockfish missing — run packaging/build_app.sh")

a = Analysis(
    ["main.py"],
    pathex=[HERE],
    binaries=[(stockfish, "bin")],  # not "." — it would shadow the `stockfish` Python package
    datas=[],
    hiddenimports=[
        # supabase pulls these lazily
        "supabase", "supabase_auth", "supabase_functions", "postgrest", "storage3", "realtime",
        "httpx", "h11", "h2", "websockets", "websockets.legacy", "websockets.legacy.client",
        "deprecation", "pydantic", "pydantic_core",
        # macOS OCR
        "Vision", "Quartz", "Foundation", "objc",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "jupyter",
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
    name="Chess Vision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Chess Vision")
app = BUNDLE(
    coll,
    name="Chess Vision.app",
    icon="assets/ChessVision.icns",
    bundle_identifier="com.chessvision.app",
    info_plist={
        "CFBundleName": "Chess Vision",
        "CFBundleDisplayName": "Chess Vision",
        "CFBundleShortVersionString": os.environ.get("CV_VERSION", "1.0.0"),
        "CFBundleVersion": os.environ.get("CV_BUILD", "1"),
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "LSApplicationCategoryType": "public.app-category.games",
        "NSHumanReadableCopyright": "© 2026 Chess Vision",
        # Shown by macOS next to the Screen Recording permission request
        "NSScreenCaptureUsageDescription":
            "Chess Vision reads the chess board on your screen to suggest moves.",
    },
)
