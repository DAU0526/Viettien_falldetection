from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {model.names}")

    class_values = [str(name).lower() for name in model.names.values()]
    if "fall" in class_values:
        print("OK: Model co class fall")
    else:
        print("CANH BAO: Chua thay class fall trong model.names")


if __name__ == "__main__":
    main()
