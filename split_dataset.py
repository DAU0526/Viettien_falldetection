from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def _keypoint_path(row_file: str, keypoint_dir: Path) -> Path:
    """Lay duong dan keypoint tu ten file clip trong labels CSV."""
    return keypoint_dir / Path(row_file).name


def _print_split_stats(name: str, df: pd.DataFrame) -> None:
    """In so clip va ti le fall/no-fall cua mot split."""
    total = len(df)
    counts = df["label"].value_counts().to_dict()
    no_fall = int(counts.get(0, 0))
    fall = int(counts.get(1, 0))

    no_fall_rate = no_fall / total if total else 0
    fall_rate = fall / total if total else 0

    print(
        f"{name}: total={total}, "
        f"no_fall={no_fall} ({no_fall_rate:.2%}), "
        f"fall={fall} ({fall_rate:.2%})"
    )


def _compute_class_weights(df: pd.DataFrame) -> dict[str, float]:
    """Tinh class weight theo cong thuc total / (2 * count[class])."""
    total = len(df)
    counts = df["label"].value_counts().to_dict()
    weights: dict[str, float] = {}

    for class_id in [0, 1]:
        count = int(counts.get(class_id, 0))
        if count == 0:
            raise ValueError(f"Khong co mau nao cho class {class_id}, khong tinh duoc class weight")
        weights[str(class_id)] = total / (2 * count)

    return weights


def split(labels_csv: str | Path, keypoint_dir: str | Path, splits_dir: str | Path) -> None:
    """Entry point chinh de chia dataset thanh train/val/test."""
    labels_csv = Path(labels_csv)
    keypoint_dir = Path(keypoint_dir)
    splits_dir = Path(splits_dir)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Khong tim thay labels CSV: {labels_csv}")
    if not keypoint_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc keypoint: {keypoint_dir}")

    df = pd.read_csv(labels_csv)
    if "file" not in df.columns or "label" not in df.columns:
        raise ValueError("labels CSV phai co cot 'file' va 'label'")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["keypoint_file"] = df["file"].apply(lambda value: _keypoint_path(str(value), keypoint_dir))

    # Chi giu clip da co keypoint output, bo clip bi skip trong buoc extract_keypoints.
    df = df[df["keypoint_file"].apply(lambda path: path.exists())].reset_index(drop=True)
    if df.empty:
        raise ValueError("Khong co clip nao co file keypoint .npy tuong ung")

    class_counts = df["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Can co ca 2 class label 0 va 1 de stratified split")
    if class_counts.min() < 3:
        raise ValueError("Moi class can it nhat 3 clip de chia train/val/test co stratify")

    # Output CSV nen tro toi file keypoint .npy that su ton tai.
    df["file"] = df["keypoint_file"].astype(str)
    df = df.drop(columns=["keypoint_file"])

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    class_weights = _compute_class_weights(train_df)
    (splits_dir / "class_weights.json").write_text(
        json.dumps(class_weights, indent=2),
        encoding="utf-8",
    )

    _print_split_stats("train", train_df)
    _print_split_stats("val", val_df)
    _print_split_stats("test", test_df)
    print(f"class_weights: {class_weights}")
    print(f"Da luu splits vao: {splits_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chia dataset LE2I keypoints thanh train/val/test")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data") / "labels_coffee_room.csv",
        help="File CSV metadata tu buoc extract_clips",
    )
    parser.add_argument(
        "--keypoint-dir",
        type=Path,
        default=Path("data") / "keypoints" / "coffee_room",
        help="Thu muc chua keypoint .npy da extract",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data") / "splits",
        help="Thu muc output train.csv, val.csv, test.csv, class_weights.json",
    )
    args = parser.parse_args()

    split(args.labels_csv, args.keypoint_dir, args.splits_dir)


if __name__ == "__main__":
    main()
