from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _empty_result(error: str) -> dict[str, Any]:
    return {
        "fall_start": 0,
        "fall_end": 0,
        "has_fall": False,
        "bboxes": [],
        "error": error,
    }


def parse_annotation(ann_path: str | Path) -> dict[str, Any]:
    """Doc 1 file annotation LE2I.

    Cau truc file:
    - Dong 0: frame bat dau te nga, 0 neu video khong co te nga
    - Dong 1: frame ket thuc te nga, 0 neu video khong co te nga
    - Dong 2 tro di: frame_no height width center_x center_y
    """
    ann_path = Path(ann_path)

    if not ann_path.exists():
        return _empty_result(f"File khong ton tai: {ann_path}")
    if not ann_path.is_file():
        return _empty_result(f"Duong dan khong phai file: {ann_path}")

    try:
        lines = [line.strip() for line in ann_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        lines = [line.strip() for line in ann_path.read_text(encoding="latin-1").splitlines() if line.strip()]
    except OSError as exc:
        return _empty_result(f"Khong doc duoc file: {exc}")

    if len(lines) < 2:
        return _empty_result("File annotation phai co it nhat 2 dong")

    try:
        fall_start = int(lines[0].split()[0])
        fall_end = int(lines[1].split()[0])
    except (ValueError, IndexError):
        return _empty_result("fall_start hoac fall_end khong hop le")

    if fall_start < 0 or fall_end < 0:
        return _empty_result("fall_start/fall_end khong duoc la gia tri am")
    if fall_start == 0 and fall_end != 0:
        return _empty_result("fall_start = 0 nhung fall_end khac 0")
    if fall_start != 0 and fall_end == 0:
        return _empty_result("fall_end = 0 nhung fall_start khac 0")
    if fall_start > fall_end:
        return _empty_result("fall_start lon hon fall_end")

    bboxes: list[dict[str, int]] = []
    errors: list[str] = []

    for line_no, line in enumerate(lines[2:], start=3):
        # Format chuan theo yeu cau: frame_no height width center_x center_y
        # Mot so file LE2I thuc te dung CSV: frame_no, flag, x1, y1, x2, y2
        parts = line.replace(",", " ").split()

        try:
            values = list(map(int, parts))
        except ValueError:
            errors.append(f"Dong {line_no}: bbox chua gia tri khong phai so nguyen")
            continue

        if len(values) == 5:
            frame_no, height, width, center_x, center_y = values
        elif len(values) == 6:
            frame_no, _flag, x1, y1, x2, y2 = values
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
        else:
            errors.append(f"Dong {line_no}: can 5 hoac 6 gia tri, nhan {len(values)}")
            continue

        if min(frame_no, height, width, center_x, center_y) < 0:
            errors.append(f"Dong {line_no}: bbox co gia tri am")
            continue

        bboxes.append(
            {
                "frame_no": frame_no,
                "height": height,
                "width": width,
                "center_x": center_x,
                "center_y": center_y,
            }
        )

    has_fall = fall_start > 0 and fall_end > 0
    error = "; ".join(errors) if errors else None

    return {
        "fall_start": fall_start,
        "fall_end": fall_end,
        "has_fall": has_fall,
        "bboxes": bboxes,
        "error": error,
    }


def label_frames(total_frames: int, fall_start: int, fall_end: int) -> list[int]:
    """Tao nhan 0/1 cho tung frame: 1 la frame nam trong doan te nga."""
    if total_frames < 0:
        raise ValueError("total_frames khong duoc am")
    if fall_start < 0 or fall_end < 0:
        raise ValueError("fall_start/fall_end khong duoc am")
    if fall_start > fall_end:
        raise ValueError("fall_start khong duoc lon hon fall_end")

    labels = [0] * total_frames

    if fall_start == 0 and fall_end == 0:
        return labels
    if fall_start == 0 or fall_end == 0:
        raise ValueError("fall_start va fall_end phai cung bang 0 neu video khong co te nga")

    start_idx = max(fall_start, 0)
    end_idx = min(fall_end, total_frames - 1)

    for frame_idx in range(start_idx, end_idx + 1):
        labels[frame_idx] = 1

    return labels


def scan_dataset(ann_dir: str | Path) -> dict[str, int]:
    """Quet tat ca file .txt trong thu muc annotation va log thong ke."""
    ann_dir = Path(ann_dir)
    stats = {
        "total": 0,
        "fall": 0,
        "no_fall": 0,
        "error": 0,
    }

    if not ann_dir.exists():
        logger.error("Thu muc annotation khong ton tai: %s", ann_dir)
        stats["error"] = 1
        return stats
    if not ann_dir.is_dir():
        logger.error("Duong dan khong phai thu muc: %s", ann_dir)
        stats["error"] = 1
        return stats

    ann_files = sorted(ann_dir.rglob("*.txt"))
    logger.info("Tim thay %d file annotation trong %s", len(ann_files), ann_dir)

    for ann_file in ann_files:
        result = parse_annotation(ann_file)
        stats["total"] += 1

        if result["error"]:
            stats["error"] += 1
            logger.warning("Loi annotation %s: %s", ann_file, result["error"])
        elif result["has_fall"]:
            stats["fall"] += 1
        else:
            stats["no_fall"] += 1

    logger.info(
        "Thong ke LE2I: total=%d, fall=%d, no_fall=%d, error=%d",
        stats["total"],
        stats["fall"],
        stats["no_fall"],
        stats["error"],
    )
    return stats


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Quet annotation files cua dataset LE2I")
    parser.add_argument("ann_dir", type=Path, help="Thu muc chua cac file annotation .txt")
    args = parser.parse_args()

    scan_dataset(args.ann_dir)

