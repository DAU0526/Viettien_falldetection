from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


KEYPOINT_SHAPE = (75, 33, 4)
FLATTEN_SHAPE = (75, 132)


class FallDataset(Dataset):
    def __init__(self, csv_path: str | Path, keypoint_dir: str | Path, augment: bool = False):
        self.csv_path = Path(csv_path)
        self.keypoint_dir = Path(keypoint_dir)
        self.augment = augment

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Khong tim thay CSV: {self.csv_path}")
        if not self.keypoint_dir.exists():
            raise FileNotFoundError(f"Khong tim thay thu muc keypoint: {self.keypoint_dir}")

        self.df = pd.read_csv(self.csv_path)
        if "file" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV phai co cot 'file' va 'label'")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_keypoint_path(self, file_value: str) -> Path:
        path = Path(file_value)
        if path.exists():
            return path
        return self.keypoint_dir / path.name

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        keypoint_path = self._resolve_keypoint_path(str(row["file"]))
        label = int(row["label"])

        keypoints = np.load(keypoint_path).astype(np.float32)
        if keypoints.shape != KEYPOINT_SHAPE:
            raise ValueError(f"Keypoint sai shape {keypoints.shape}: {keypoint_path}")

        if self.augment:
            # Them Gaussian noise nho vao keypoint.
            noise = np.random.normal(loc=0.0, scale=0.01, size=keypoints.shape).astype(np.float32)
            keypoints = keypoints + noise

            # Dao nguoc thu tu thoi gian voi xac suat 0.3.
            if random.random() < 0.3:
                keypoints = np.flip(keypoints, axis=0).copy()

        keypoints = keypoints.reshape(FLATTEN_SHAPE)

        keypoints_tensor = torch.from_numpy(keypoints).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return keypoints_tensor, label_tensor


def get_class_weights(csv_path: str | Path) -> torch.Tensor:
    """Doc data/splits/class_weights.json va tra ve tensor [w0, w1]."""
    csv_path = Path(csv_path)
    weights_path = csv_path.parent / "class_weights.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Khong tim thay class_weights.json: {weights_path}")

    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    return torch.tensor([float(weights["0"]), float(weights["1"])], dtype=torch.float32)
