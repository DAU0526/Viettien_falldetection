from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict_video import DEFAULT_INPUT, DEFAULT_MODEL, DEFAULT_OUTPUT, predict_video


if __name__ == "__main__":
    print("Chay demo phat hien te nga")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Video dau vao: {DEFAULT_INPUT}")
    print(f"Video dau ra: {DEFAULT_OUTPUT}")
    predict_video(DEFAULT_MODEL, DEFAULT_INPUT, DEFAULT_OUTPUT)
