from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from parse_le2i import label_frames, parse_annotation


logger = logging.getLogger(__name__)

WINDOW_SIZE = 75
STEP_SIZE = 15
FALL_FRAME_THRESHOLD = 8
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mkv")


def find_video_file(video_dir: str | Path, base_name: str) -> Path | None:
    """Tim video theo ten annotation, thu lan luot .avi, .mp4, .mkv."""
    video_dir = Path(video_dir)

    for ext in VIDEO_EXTENSIONS:
        video_path = video_dir / f"{base_name}{ext}"
        if video_path.exists():
            return video_path

    return None


def _resolve_input_dirs(data_root: Path) -> tuple[Path, Path]:
    """Tim dung thu muc Videos va Annotation_files cua Coffee_room_01."""
    direct_video_dir = data_root / "lezzi" / "Coffee_room_01" / "Videos"
    direct_ann_dir = data_root / "lezzi" / "Coffee_room_01" / "Annotation_files"

    if direct_video_dir.exists() and direct_ann_dir.exists():
        return direct_video_dir, direct_ann_dir

    nested_video_dir = data_root / "lezzi" / "Coffee_room_01" / "Coffee_room_01" / "Videos"
    nested_ann_dir = data_root / "lezzi" / "Coffee_room_01" / "Coffee_room_01" / "Annotation_files"

    if nested_video_dir.exists() and nested_ann_dir.exists():
        return nested_video_dir, nested_ann_dir

    raise FileNotFoundError(
        "Khong tim thay thu muc Videos/Annotation_files cua Coffee_room_01 trong data_root"
    )


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    """Doc toan bo frame video va resize ve dung kich thuoc LE2I 320x240."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    frames: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        frames.append(frame)

    cap.release()
    return frames


def _save_clip(clip_frames: list[np.ndarray], clip_path: Path) -> None:
    """Luu clip thanh file .npy shape (75, 240, 320, 3)."""
    clip_array = np.stack(clip_frames, axis=0)

    if clip_array.shape != (WINDOW_SIZE, FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise ValueError(f"Shape clip khong hop le: {clip_array.shape}")

    np.save(clip_path, clip_array)


def run(data_root: str | Path, output_dir: str | Path, csv_out: str | Path) -> None:
    """Entry point chinh de cat clip Coffee_room tu dataset LE2I."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    csv_out = Path(csv_out)

    video_dir, ann_dir = _resolve_input_dirs(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(ann_dir.glob("*.txt"))
    rows: list[dict[str, str | int]] = []

    logger.info("Video dir: %s", video_dir)
    logger.info("Annotation dir: %s", ann_dir)
    logger.info("Tim thay %d annotation files", len(annotation_files))

    for ann_path in tqdm(annotation_files, desc="Extract Coffee_room clips"):
        parsed = parse_annotation(ann_path)
        if parsed["error"]:
            logger.warning("Bo qua annotation loi %s: %s", ann_path, parsed["error"])
            continue

        video_path = find_video_file(video_dir, ann_path.stem)
        if video_path is None:
            logger.warning("Khong tim thay video cho annotation: %s", ann_path.name)
            continue

        frames = _read_video_frames(video_path)
        total_frames = len(frames)
        if total_frames < WINDOW_SIZE:
            logger.warning("Bo qua video ngan hon %d frame: %s", WINDOW_SIZE, video_path.name)
            continue

        frame_labels = label_frames(total_frames, parsed["fall_start"], parsed["fall_end"])

        for clip_start in range(0, total_frames - WINDOW_SIZE + 1, STEP_SIZE):
            clip_end = clip_start + WINDOW_SIZE
            fall_frames = sum(frame_labels[clip_start:clip_end])
            clip_label = 1 if fall_frames >= FALL_FRAME_THRESHOLD else 0

            clip_file = f"{video_path.stem}_start_{clip_start:05d}_label_{clip_label}.npy"
            clip_path = output_dir / clip_file

            _save_clip(frames[clip_start:clip_end], clip_path)

            rows.append(
                {
                    "file": str(clip_path),
                    "label": clip_label,
                    "source_video": str(video_path),
                    "fall_start": parsed["fall_start"],
                    "fall_end": parsed["fall_end"],
                    "clip_start_frame": clip_start,
                    "fall_frames_in_clip": fall_frames,
                }
            )

    fieldnames = [
        "file",
        "label",
        "source_video",
        "fall_start",
        "fall_end",
        "clip_start_frame",
        "fall_frames_in_clip",
    ]

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Da luu %d clips vao %s", len(rows), output_dir)
    logger.info("Da luu metadata vao %s", csv_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cat video LE2I Coffee_room thanh clip .npy")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Thu muc goc chua folder lezzi/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "clips" / "coffee_room",
        help="Thu muc luu clip .npy",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data") / "labels_coffee_room.csv",
        help="File CSV metadata dau ra",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args.data_root, args.output_dir, args.csv_out)


if __name__ == "__main__":
    main()
