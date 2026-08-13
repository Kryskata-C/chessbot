"""Screen capture using mss."""

import numpy as np
import mss


def capture_screen(monitor_index: int = 1,
                   region: dict | None = None) -> np.ndarray:
    """Capture the screen and return as a BGR numpy array.

    Args:
        monitor_index: Which monitor to capture (1 = primary).
        region: Optional sub-region {left, top, width, height} to grab
            instead of the whole monitor (much cheaper).

    Returns:
        Screenshot as BGR numpy array suitable for OpenCV.
    """
    with mss.mss() as sct:
        monitor = region if region is not None else sct.monitors[monitor_index]
        screenshot = sct.grab(monitor)
        # mss returns BGRA, convert to BGR for OpenCV
        frame = np.array(screenshot)
        return frame[:, :, :3].copy()


def get_monitor_info(monitor_index: int = 1) -> dict:
    """Return monitor position and size."""
    with mss.mss() as sct:
        return dict(sct.monitors[monitor_index])
