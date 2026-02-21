"""One-time calibration tool to extract piece templates from a chess.com screenshot.

Usage:
    1. Open chess.com in your browser with pieces at starting position
    2. Run: python calibrate.py
    3. Templates will be saved to templates/ folder
"""

import os
import sys
import time
import cv2

from capture import capture_screen
from board_detector import detect_board


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
TEMPLATE_SIZE = 80

# Starting position layout (row 0 = top of screen = rank 8 for white-bottom)
# Standard: black pieces on top, white on bottom
STARTING_POSITION = [
    # Row 0 (rank 8): black back rank
    ["black_rook", "black_knight", "black_bishop", "black_queen",
     "black_king", "black_bishop", "black_knight", "black_rook"],
    # Row 1 (rank 7): black pawns
    ["black_pawn"] * 8,
    # Rows 2-5: empty
    [None] * 8,
    [None] * 8,
    [None] * 8,
    [None] * 8,
    # Row 6 (rank 2): white pawns
    ["white_pawn"] * 8,
    # Row 7 (rank 1): white back rank
    ["white_rook", "white_knight", "white_bishop", "white_queen",
     "white_king", "white_bishop", "white_knight", "white_rook"],
]


def calibrate():
    print("Chess Vision - Calibration Tool")
    print("=" * 40)
    print("Make sure chess.com is open with the starting position visible.")
    print("The board should have standard green/beige colors.")
    print()

    input("Press Enter when ready to capture...")
    time.sleep(0.5)

    print("Capturing screen...")
    screenshot = capture_screen()

    print("Detecting board...")
    board = detect_board(screenshot)

    if board is None:
        print("ERROR: Could not detect chess board on screen!")
        print("Tips:")
        print("  - Make sure the chess.com board is fully visible")
        print("  - Use the default green/beige theme")
        print("  - Grant screen recording permission to your terminal")
        sys.exit(1)

    print(f"Board found at ({board['x']}, {board['y']}) "
          f"size {board['width']}x{board['height']}")
    print(f"Square size: {board['square_size']:.1f}px")

    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    sq = board["square_size"]
    saved = 0
    seen = set()

    for row in range(8):
        for col in range(8):
            piece_name = STARTING_POSITION[row][col]
            if piece_name is None:
                continue
            if piece_name in seen:
                continue  # Only save one template per piece type
            seen.add(piece_name)

            x = int(board["x"] + col * sq)
            y = int(board["y"] + row * sq)
            w = int(sq)
            h = int(sq)

            square_img = screenshot[y : y + h, x : x + w]
            if square_img.size == 0:
                print(f"  WARNING: Empty crop for {piece_name} at ({row},{col})")
                continue

            # Resize to standard template size
            template = cv2.resize(square_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            path = os.path.join(TEMPLATE_DIR, f"{piece_name}.png")
            cv2.imwrite(path, template)
            print(f"  Saved {piece_name}.png")
            saved += 1

    print()
    if saved == 12:
        print(f"SUCCESS: All {saved} templates saved to templates/")
    else:
        print(f"WARNING: Only {saved}/12 templates saved. Some pieces may be missing.")
        print("You may need to adjust board detection parameters.")


if __name__ == "__main__":
    calibrate()
