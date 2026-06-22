from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

CLIP_SHAPE = (75, 240, 320, 3)
KEYPOINT_SHAPE = (75, 33, 4)
LOG_DIR = Path("logs")
SKIPPED_JSON = LOG_DIR / "skipped_low_detection.json"
LOG_FILE = LOG_DIR / "keypoint_extraction.log"


def get_pose_module():
    """Lay module MediaPipe Pose, ho tro cac ban mediapipe khac nhau."""
    first_error = None

    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return mp.solutions.pose
    except Exception as exc:
        first_error = exc

    try:
        return importlib.import_module("mediapipe.python.solutions.pose")
    except Exception as exc:
        message = (
            "Khong import duoc MediaPipe Pose. Hay cai lai mediapipe/protobuf, vi moi truong "
            "hien tai co the dang bi lech phien ban. Thu chay: "
            "pip install --upgrade protobuf mediapipe"
        )
        if first_error is not None:
            raise ImportError(message) from first_error
        raise ImportError(message) from exc

def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_skipped_log() -> list[dict]:
    if not SKIPPED_JSON.exists():
        return []

    try:
        return json.loads(SKIPPED_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("File skipped log bi loi JSON, tao lai file moi: %s", SKIPPED_JSON)
        return []


def save_skipped_log(records: list[dict]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SKIPPED_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_by_person_bbox(keypoints: np.ndarray) -> np.ndarray:
    """Normalize x, y theo bbox cua nguoi trong tung frame."""
    normalized = keypoints.copy()

    for frame_idx in range(normalized.shape[0]):
        frame_points = normalized[frame_idx]

        # Frame khong co detection thi bo qua, giu toan bo bang 0.
        if not np.any(frame_points):
            continue

        x_values = frame_points[:, 0]
        y_values = frame_points[:, 1]

        min_x = float(np.min(x_values))
        max_x = float(np.max(x_values))
        min_y = float(np.min(y_values))
        max_y = float(np.max(y_values))

        box_w = max_x - min_x
        box_h = max_y - min_y

        if box_w <= 1e-6 or box_h <= 1e-6:
            continue

        normalized[frame_idx, :, 0] = (x_values - min_x) / box_w
        normalized[frame_idx, :, 1] = (y_values - min_y) / box_h

    return normalized


def interpolate_missing_frames(keypoints: np.ndarray, detected_mask: np.ndarray) -> np.ndarray:
    """Noi suy tuyen tinh cac frame khong detect duoc nguoi."""
    interpolated = keypoints.copy()
    frame_ids = np.arange(keypoints.shape[0])
    valid_ids = frame_ids[detected_mask]

    if len(valid_ids) == 0:
        return interpolated

    for landmark_idx in range(keypoints.shape[1]):
        for value_idx in range(keypoints.shape[2]):
            valid_values = keypoints[detected_mask, landmark_idx, value_idx]
            interpolated[:, landmark_idx, value_idx] = np.interp(frame_ids, valid_ids, valid_values)

    return interpolated


def extract_clip_keypoints(clip: np.ndarray, pose, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Trich keypoint MediaPipe Pose cho 1 clip."""
    keypoints = np.zeros(KEYPOINT_SHAPE, dtype=np.float32)
    detected_mask = np.zeros(KEYPOINT_SHAPE[0], dtype=bool)

    for frame_idx, frame in enumerate(clip):
        # Clip tao bang OpenCV thuong la BGR, MediaPipe can RGB.
        rgb_frame = frame[:, :, ::-1]
        rgb_frame = np.ascontiguousarray(rgb_frame)

        result = pose.process(rgb_frame)
        if result.pose_landmarks is None:
            continue

        detected_mask[frame_idx] = True
        for landmark_idx, landmark in enumerate(result.pose_landmarks.landmark):
            keypoints[frame_idx, landmark_idx] = [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
            ]

    if normalize:
        keypoints = normalize_by_person_bbox(keypoints)

    keypoints = interpolate_missing_frames(keypoints, detected_mask)
    return keypoints.astype(np.float32), detected_mask


def run(
    clip_dir: str | Path,
    output_dir: str | Path,
    normalize: bool = True,
    min_det_rate: float = 0.5,
) -> None:
    """Entry point chinh de trich keypoint tu tat ca clip .npy."""
    clip_dir = Path(clip_dir)
    output_dir = Path(output_dir)

    if not clip_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc clip: {clip_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    skipped_records = load_skipped_log()
    clip_files = sorted(clip_dir.glob("*.npy"))

    logger.info("Tim thay %d clip trong %s", len(clip_files), clip_dir)
    logger.info("Output keypoints: %s", output_dir)

    mp_pose = get_pose_module()

    # Khoi tao Pose 1 lan duy nhat va tai su dung cho tat ca clip.
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for clip_path in tqdm(clip_files, desc="Extract MediaPipe keypoints"):
            output_path = output_dir / clip_path.name

            # Resume: clip nao da co output thi bo qua.
            if output_path.exists():
                logger.info("Skip clip da co output: %s", clip_path.name)
                continue

            clip = np.load(clip_path)
            if clip.shape != CLIP_SHAPE:
                logger.warning("Skip clip sai shape %s: %s", clip.shape, clip_path.name)
                continue

            keypoints, detected_mask = extract_clip_keypoints(clip, pose, normalize=normalize)

            detected_frames = int(detected_mask.sum())
            missed_frames = int(len(detected_mask) - detected_frames)
            detection_rate = detected_frames / len(detected_mask)

            logger.info(
                "%s: missed_frames=%d, detection_rate=%.3f",
                clip_path.name,
                missed_frames,
                detection_rate,
            )

            if detection_rate < min_det_rate:
                skipped_records.append(
                    {
                        "file": str(clip_path),
                        "detected_frames": detected_frames,
                        "total_frames": int(len(detected_mask)),
                        "detection_rate": detection_rate,
                    }
                )
                save_skipped_log(skipped_records)
                logger.warning("Skip low detection clip: %s", clip_path.name)
                continue

            if keypoints.shape != KEYPOINT_SHAPE:
                raise ValueError(f"Shape keypoint khong hop le {keypoints.shape}: {clip_path}")

            np.save(output_path, keypoints)

    save_skipped_log(skipped_records)
    logger.info("Hoan tat trich keypoint")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trich keypoint MediaPipe Pose tu clip LE2I")
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=Path("data") / "clips" / "coffee_room",
        help="Thu muc input chua clip .npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "keypoints" / "coffee_room",
        help="Thu muc output chua keypoint .npy",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Tat normalize x, y theo bbox nguoi")
    parser.add_argument("--min-det-rate", type=float, default=0.5, help="Nguong detection_rate toi thieu")
    args = parser.parse_args()

    setup_logging()
    run(
        clip_dir=args.clip_dir,
        output_dir=args.output_dir,
        normalize=not args.no_normalize,
        min_det_rate=args.min_det_rate,
    )


if __name__ == "__main__":
    main()


