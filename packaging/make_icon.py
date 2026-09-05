"""Draw the app icon (tournament-noir palette) and build assets/ChessVision.icns."""
import os, subprocess, sys, tempfile
from PyQt6.QtGui import QImage, QPainter, QColor, QPainterPath, QPen, QGuiApplication
from PyQt6.QtCore import QRectF, Qt

app = QGuiApplication(sys.argv)
BG, BONE, GREEN, DARK = QColor("#0e0d0b"), QColor("#efe9dd"), QColor("#b8ff3a"), QColor("#1c1a17")

def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size
    path = QPainterPath(); path.addRoundedRect(QRectF(s*0.04, s*0.04, s*0.92, s*0.92), s*0.2, s*0.2)
    p.fillPath(path, BG)
    # 4x4 board tile, slightly inset, bone/dark squares
    ox, oy, sq = s*0.22, s*0.22, s*0.14
    for r in range(4):
        for c in range(4):
            p.fillRect(QRectF(ox + c*sq, oy + r*sq, sq, sq), BONE if (r + c) % 2 == 0 else DARK)
    # the "vision" arrow: a green move arrow across the board
    pen = QPen(GREEN, s*0.045, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    x0, y0 = ox + 0.5*sq, oy + 3.5*sq
    x1, y1 = ox + 2.5*sq, oy + 1.5*sq
    p.drawLine(int(x0), int(y0), int(x1), int(y1))
    import math
    dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy); dx, dy = dx / L, dy / L
    px, py = -dy, dx  # perpendicular
    ln, wd = s*0.13, s*0.075
    head = QPainterPath(); head.moveTo(x1 + dx*s*0.02, y1 + dy*s*0.02)
    head.lineTo(x1 - dx*ln + px*wd, y1 - dy*ln + py*wd)
    head.lineTo(x1 - dx*ln - px*wd, y1 - dy*ln - py*wd); head.closeSubpath()
    p.fillPath(head, GREEN)
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(GREEN)
    p.drawEllipse(QRectF(x0 - s*0.045, y0 - s*0.045, s*0.09, s*0.09))
    p.end()
    return img

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(out, exist_ok=True)
draw(1024).save(os.path.join(out, "icon-1024.png"))
with tempfile.TemporaryDirectory() as td:
    iconset = os.path.join(td, "ChessVision.iconset"); os.makedirs(iconset)
    for base in (16, 32, 128, 256, 512):
        draw(base).save(os.path.join(iconset, f"icon_{base}x{base}.png"))
        draw(base*2).save(os.path.join(iconset, f"icon_{base}x{base}@2x.png"))
    subprocess.check_call(["iconutil", "-c", "icns", iconset, "-o", os.path.join(out, "ChessVision.icns")])
print("wrote", os.path.join(out, "ChessVision.icns"))
