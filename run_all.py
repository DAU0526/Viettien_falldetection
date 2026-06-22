from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable

from extract_clips import run as run_extract_clips
from parse_le2i import scan_dataset
from split_dataset import split as split_dataset


logger = logging.getLogger(__name__)


def resolve_annotation_dir(data_root: Path) -> Path:
    """Tim thu muc Annotation_files tu data_root Coffee_room_01."""
    candidates = [
        data_root / "Annotation_files",
        data_root / "Coffee_room_01" / "Annotation_files",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Khong tim thay Annotation_files trong: {data_root}")


def resolve_project_root(data_root: Path) -> Path:
    """Lay project root de goi extract_clips.run theo entry point hien co."""
    if data_root.name == "Coffee_room_01" and data_root.parent.name == "lezzi":
        return data_root.parent.parent

    if data_root.name == "Coffee_room_01" and data_root.parent.name == "Coffee_room_01":
        return data_root.parent.parent.parent

    return Path(".")


def run_keypoint_step(clip_dir: Path, kp_dir: Path, min_det_rate: float) -> None:
    from extract_keypoints import run as run_extract_keypoints

    run_extract_keypoints(
        clip_dir=clip_dir,
        output_dir=kp_dir,
        normalize=True,
        min_det_rate=min_det_rate,
    )

def timed_step(name: str, func: Callable[[], None]) -> float:
    """Chay mot buoc va in thoi gian thuc thi."""
    print(f"\n=== {name} ===")
    start = time.perf_counter()
    func()
    elapsed = time.perf_counter() - start
    print(f"{name} done in {elapsed:.2f}s")
    return elapsed


def print_outputs(clip_dir: Path, kp_dir: Path, splits_dir: Path, labels_csv: Path) -> None:
    print("\n=== Output structure ===")
    print(f"Clips     : {clip_dir}")
    print(f"Keypoints : {kp_dir}")
    print(f"Splits    : {splits_dir}")
    print(f"Labels CSV: {labels_csv}")
    print(f"Train CSV : {splits_dir / 'train.csv'}")
    print(f"Val CSV   : {splits_dir / 'val.csv'}")
    print(f"Test CSV  : {splits_dir / 'test.csv'}")
    print(f"Weights   : {splits_dir / 'class_weights.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chay toan bo pipeline xu ly LE2I Coffee_room")
    parser.add_argument("--data_root", type=Path, default=Path("lezzi") / "Coffee_room_01")
    parser.add_argument("--clip_dir", type=Path, default=Path("data") / "clips" / "coffee_room")
    parser.add_argument("--kp_dir", type=Path, default=Path("data") / "keypoints" / "coffee_room")
    parser.add_argument("--splits_dir", type=Path, default=Path("data") / "splits")
    parser.add_argument("--labels_csv", type=Path, default=Path("data") / "labels_coffee_room.csv")
    parser.add_argument("--min_det_rate", type=float, default=0.5)
    parser.add_argument("--skip_clips", action="store_true", help="Bo qua buoc cat clip neu da co clip")
    parser.add_argument("--skip_kp", action="store_true", help="Bo qua buoc trich keypoint neu da co keypoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    total_start = time.perf_counter()
    step_times: list[tuple[str, float]] = []

    ann_dir = resolve_annotation_dir(args.data_root)
    extract_clips_root = resolve_project_root(args.data_root)

    step_times.append(("1. scan_dataset", timed_step("1. scan_dataset", lambda: scan_dataset(ann_dir))))

    if args.skip_clips:
        print("\n=== 2. extract_clips ===")
        print("Skipped by --skip_clips")
    else:
        step_times.append(
            (
                "2. extract_clips",
                timed_step(
                    "2. extract_clips",
                    lambda: run_extract_clips(extract_clips_root, args.clip_dir, args.labels_csv),
                ),
            )
        )

    if args.skip_kp:
        print("\n=== 3. extract_keypoints ===")
        print("Skipped by --skip_kp")
    else:
        step_times.append(
            (
                "3. extract_keypoints",
                timed_step(
                    "3. extract_keypoints",
                    lambda: run_keypoint_step(args.clip_dir, args.kp_dir, args.min_det_rate),
                ),
            )
        )

    step_times.append(
        (
            "4. split_dataset",
            timed_step(
                "4. split_dataset",
                lambda: split_dataset(args.labels_csv, args.kp_dir, args.splits_dir),
            ),
        )
    )

    total_elapsed = time.perf_counter() - total_start

    print("\n=== Timing summary ===")
    for name, elapsed in step_times:
        print(f"{name}: {elapsed:.2f}s")
    print(f"Total: {total_elapsed:.2f}s")

    print_outputs(args.clip_dir, args.kp_dir, args.splits_dir, args.labels_csv)


if __name__ == "__main__":
    main()

