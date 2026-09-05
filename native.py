"""The thin OS-specific layer: everything the app needs from macOS or
Windows that Qt does not cover, behind one set of functions.

    overlay pinning      pin_overlay(), overlay_needs_repin(), exclude_from_capture()
    card pinning         pin_card()
    screen permission    screen_capture_granted(), request_screen_capture(),
                         open_screen_capture_settings(), relaunch()
    secrets              secret_get(), secret_set(), secret_delete()
    OCR                  ocr_text()                  (None when unavailable)
    process              configure_qt_env(), quiet_subprocesses()

macOS: Cocoa through objc_msgSend (no pyobjc needed for the overlay), Vision
for OCR, the `security` keychain CLI. Windows: Win32 through ctypes
(layered click-through topmost window excluded from capture with
SetWindowDisplayAffinity), Windows.Media.Ocr through the winrt packages when
installed, Credential Manager through `keyring`. Anything else: no-ops that
keep the app running with plain Qt windows.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"


# =============================================================== macOS: Cocoa

_ns: dict | None = None


def _cocoa() -> dict | None:
    """Typed objc_msgSend wrappers (required for the arm64 ABI), cached."""
    global _ns
    if _ns is not None or not IS_MAC:
        return _ns
    try:
        lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        cast = ctypes.cast
        _ns = {
            "sel": lib.sel_registerName,
            "send": cast(lib.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)),
            "send_long": cast(lib.objc_msgSend, ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long)),
            "send_bool": cast(lib.objc_msgSend, ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)),
            "get_long": cast(lib.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)),
        }
    except Exception as e:
        print(f"cocoa bridge warning: {e}")
        _ns = {}
    return _ns


def _qt_platform() -> str:
    try:
        from PyQt6.QtWidgets import QApplication
        return QApplication.platformName()
    except Exception:
        return ""


def _nswindow(widget) -> int:
    """The NSWindow behind a Qt widget; 0 unless Qt runs on Cocoa (under the
    offscreen test platform winId() is not an NSView and objc calls crash)."""
    ns = _cocoa()
    if not ns or _qt_platform() != "cocoa":
        return 0
    try:
        return ns["send"](int(widget.winId()), ns["sel"](b"window")) or 0
    except Exception:
        return 0


# ============================================================ Windows: Win32

if IS_WIN:
    _user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    _kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    _GWL_EXSTYLE = -20
    _WS_EX_TOPMOST = 0x00000008
    _WS_EX_TRANSPARENT = 0x00000020
    _WS_EX_TOOLWINDOW = 0x00000080
    _WS_EX_LAYERED = 0x00080000
    _WS_EX_NOACTIVATE = 0x08000000
    _HWND_TOPMOST = ctypes.c_void_p(-1 & 0xFFFFFFFFFFFFFFFF)
    _SWP_NOSIZE, _SWP_NOMOVE, _SWP_NOACTIVATE, _SWP_SHOWWINDOW = 0x1, 0x2, 0x10, 0x40
    _WDA_MONITOR, _WDA_EXCLUDEFROMCAPTURE = 0x1, 0x11
    _user32.GetWindowLongPtrW.restype = ctypes.c_longlong
    _user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _user32.SetWindowLongPtrW.restype = ctypes.c_longlong
    _user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]
    _user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    _user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _user32.GetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]


def _hwnd(widget) -> int:
    """The HWND behind a Qt widget; 0 unless Qt runs on the windows platform."""
    if _qt_platform() != "windows":
        return 0
    try:
        return int(widget.winId())
    except Exception:
        return 0


def _win_exclude(hwnd: int) -> bool:
    """Hide the window from screen capture (GDI/DXGI grabs, so also mss).
    WDA_EXCLUDEFROMCAPTURE needs Windows 10 2004+; older builds get
    WDA_MONITOR, which blanks the window in captures — equally fine."""
    if _user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAPTURE):
        return True
    return bool(_user32.SetWindowDisplayAffinity(hwnd, _WDA_MONITOR))


# ================================================================= overlays

def pin_overlay(widget) -> bool:
    """Make the full-screen overlay truly always-on-top, click-through and
    invisible to screen capture. Idempotent: call again after a move to
    another display. Returns False when the native window is not ready."""
    if IS_MAC:
        ns = _cocoa()
        w = _nswindow(widget)
        if not ns or not w:
            return False
        sel = ns["sel"]
        ns["send_long"](w, sel(b"setLevel:"), 1000)                  # NSScreenSaverWindowLevel
        ns["send_bool"](w, sel(b"setIgnoresMouseEvents:"), True)
        # canJoinAllSpaces | stationary | fullScreenAuxiliary
        ns["send_long"](w, sel(b"setCollectionBehavior:"), (1 << 0) | (1 << 4) | (1 << 8))
        ns["send_long"](w, sel(b"setSharingType:"), 0)               # NSWindowSharingNone
        ns["send"](w, sel(b"orderFrontRegardless"))
        return True
    if IS_WIN:
        h = _hwnd(widget)
        if not h:
            return False
        ex = _user32.GetWindowLongPtrW(h, _GWL_EXSTYLE)
        ex |= _WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE | _WS_EX_TOPMOST
        _user32.SetWindowLongPtrW(h, _GWL_EXSTYLE, ex)
        _user32.SetWindowPos(h, _HWND_TOPMOST, 0, 0, 0, 0,
                             _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_SHOWWINDOW)
        _win_exclude(h)
        return True
    return False


def overlay_needs_repin(widget) -> bool:
    """Cheap once-a-second check: did the OS knock the overlay off?"""
    if IS_MAC:
        ns = _cocoa()
        w = _nswindow(widget)
        if not ns or not w:
            return False
        return not ns["get_long"](w, ns["sel"](b"isOnActiveSpace"))
    if IS_WIN:
        h = _hwnd(widget)
        if not h:
            return False
        ex = _user32.GetWindowLongPtrW(h, _GWL_EXSTYLE)
        if not (ex & _WS_EX_TOPMOST) or not (ex & _WS_EX_TRANSPARENT):
            return True
        aff = ctypes.c_uint(0)
        if _user32.GetWindowDisplayAffinity(h, ctypes.byref(aff)) and aff.value == 0:
            return True
        return False
    return False


def exclude_from_capture(widget) -> None:
    """Keep a small helper window (debug board) out of our own screen grabs."""
    if IS_MAC:
        ns = _cocoa()
        w = _nswindow(widget)
        if ns and w:
            ns["send_long"](w, ns["sel"](b"setSharingType:"), 0)
    elif IS_WIN:
        h = _hwnd(widget)
        if h:
            _win_exclude(h)


def pin_card(widget) -> None:
    """Let a dialog card float over a fullscreen app on its display.
    Windows: WindowStaysOnTopHint already does this."""
    if IS_MAC:
        ns = _cocoa()
        w = _nswindow(widget)
        if not ns or not w:
            return
        sel = ns["sel"]
        ns["send_long"](w, sel(b"setCollectionBehavior:"), (1 << 0) | (1 << 8))
        ns["send_long"](w, sel(b"setLevel:"), 3)                     # NSFloatingWindowLevel
        ns["send"](w, sel(b"orderFrontRegardless"))


# ======================================================= screen permission

def screen_capture_granted() -> bool:
    if IS_MAC:
        try:
            import Quartz
            return bool(Quartz.CGPreflightScreenCaptureAccess())
        except Exception:
            return True  # no pyobjc: can't tell, don't block
    return True  # Windows has no screen-recording permission


def request_screen_capture() -> bool:
    if IS_MAC:
        try:
            import Quartz
            return bool(Quartz.CGRequestScreenCaptureAccess())
        except Exception:
            return True
    return True


def open_screen_capture_settings() -> None:
    if IS_MAC:
        subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"])


def relaunch() -> None:
    """Start a fresh copy of the app; the caller quits this one."""
    exe = sys.executable
    frozen = bool(getattr(sys, "frozen", False))
    if IS_WIN:
        args = [exe] if frozen else [exe, *sys.argv]
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(args, creationflags=flags, close_fds=True)
        return
    if frozen:
        bundle = os.path.abspath(os.path.join(os.path.dirname(exe), "..", ".."))
        if bundle.endswith(".app"):
            subprocess.Popen(["/bin/sh", "-c", f'sleep 0.7; open -n "{bundle}"'])
        else:
            subprocess.Popen(["/bin/sh", "-c", f'sleep 0.7; "{exe}"'])
    else:
        subprocess.Popen(["/bin/sh", "-c", "sleep 0.7; " + " ".join(f'"{a}"' for a in [exe, *sys.argv])])


# ================================================================== secrets

def secret_get(service: str, account: str) -> str | None:
    if IS_MAC:
        r = subprocess.run(["security", "find-generic-password", "-a", account, "-s", service, "-w"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    if IS_WIN:
        try:
            import keyring
            return keyring.get_password(service, account) or None
        except Exception as e:
            print(f"credential store warning: {e}")
            return None
    return _file_secret(service, account)


def secret_set(service: str, account: str, value: str) -> None:
    if IS_MAC:
        subprocess.run(["security", "add-generic-password", "-a", account, "-s", service, "-w", value, "-U"],
                       capture_output=True, text=True)
        return
    if IS_WIN:
        try:
            import keyring
            keyring.set_password(service, account, value)
        except Exception as e:
            print(f"credential store warning: {e}")
        return
    _file_secret(service, account, value)


def secret_delete(service: str, account: str) -> None:
    if IS_MAC:
        subprocess.run(["security", "delete-generic-password", "-a", account, "-s", service],
                       capture_output=True, text=True)
        return
    if IS_WIN:
        try:
            import keyring
            keyring.delete_password(service, account)
        except Exception:
            pass
        return
    _file_secret(service, account, "")


def _file_secret(service: str, account: str, value: str | None = None) -> str | None:
    """Linux / unknown: a plain file in the data dir (dev only)."""
    from paths import data_path
    path = data_path(f".{service}-{account}")
    if value is None:
        try:
            return open(path).read().strip() or None
        except OSError:
            return None
    if value == "":
        try:
            os.remove(path)
        except OSError:
            pass
    else:
        with open(path, "w") as f:
            f.write(value)
    return None


# ====================================================================== OCR

def ocr_available() -> bool:
    if IS_MAC:
        try:
            import Vision, Quartz  # noqa: F401
            return True
        except Exception:
            return False
    if IS_WIN:
        try:
            from winrt.windows.media.ocr import OcrEngine  # noqa: F401
            return True
        except Exception:
            return False
    return False


def ocr_text(gray) -> str | None:
    """Recognise text in a grayscale numpy image with the OS's own OCR.
    None when no OS OCR is available (callers may try tesseract)."""
    if IS_MAC:
        return _ocr_vision(gray)
    if IS_WIN:
        return _ocr_winrt(gray)
    return None


def _ocr_vision(gray) -> str | None:
    try:
        import cv2
        import Quartz
        import Vision
        from Foundation import NSData
        ok, png = cv2.imencode(".png", gray)
        if not ok:
            return ""
        data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if img is None:
            return ""
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
        handler.performRequests_error_([req], None)
        lines = []
        for r in req.results() or []:
            cands = r.topCandidates_(1)
            if cands:
                lines.append(str(cands[0].string()))
        return " ".join(lines)
    except Exception as e:
        print(f"vision ocr warning: {e}")
        return None


def _ocr_winrt(gray) -> str | None:
    """Windows.Media.Ocr through the `winrt-*` packages (optional install)."""
    try:
        import asyncio
        import numpy as np
        from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataWriter

        h, w = gray.shape
        bgra = np.dstack([gray, gray, gray, np.full_like(gray, 255)]).astype(np.uint8)
        writer = DataWriter()
        writer.write_bytes(bgra.tobytes())
        buf = writer.detach_buffer()
        bmp = SoftwareBitmap.create_copy_from_buffer(buf, BitmapPixelFormat.BGRA8, w, h, BitmapAlphaMode.IGNORE)
        engine = OcrEngine.try_create_from_user_profile_languages() or OcrEngine.try_create_from_language(None)
        if engine is None:
            return None

        async def run():
            return await engine.recognize_async(bmp)

        result = asyncio.run(run())
        return str(result.text) if result is not None else ""
    except Exception as e:
        print(f"windows ocr warning: {e}")
        return None


# ================================================================== process

def configure_qt_env() -> None:
    """Call before QApplication exists.

    Windows: the whole pipeline works in physical screen pixels (mss grabs
    them, the board detector measures them, the overlay paints at those
    coordinates). Qt's high-DPI scaling would put its windows in logical
    coordinates instead, so it is switched off: Qt then sees the same
    pixel grid as mss on every display, whatever the scaling setting.
    """
    if IS_WIN:
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
        os.environ.setdefault("QT_SCALE_FACTOR", "1")


def quiet_subprocesses() -> None:
    """Windows: a windowed (no console) app spawning Stockfish or tesseract
    would flash a console window for each child; default them to hidden."""
    if not IS_WIN:
        return
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    orig = subprocess.Popen

    class QuietPopen(orig):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            kw.setdefault("creationflags", 0)
            kw["creationflags"] |= flag
            super().__init__(*a, **kw)

    subprocess.Popen = QuietPopen  # type: ignore[assignment]
