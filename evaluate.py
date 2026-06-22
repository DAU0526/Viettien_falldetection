from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader

from dataset import FallDataset
from model_bilstm import FallBiLSTM


TEST_CSV = Path("data") / "splits" / "test.csv"
KEYPOINT_DIR = Path("data") / "keypoints" / "coffee_room"
CHECKPOINT_PATH = Path("models") / "best_model.pth"
LOG_DIR = Path("logs")
CONFUSION_MATRIX_PATH = LOG_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = LOG_DIR / "roc_curve.png"
REPORT_PATH = LOG_DIR / "evaluation_report.json"
BATCH_SIZE = 32
CLASS_NAMES = ["no_fall", "fall"]


def load_model(device: torch.device) -> FallBiLSTM:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint.get("config", {})

    model = FallBiLSTM(
        input_size=config.get("input_size", 132),
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model: FallBiLSTM, dataloader: DataLoader, device: torch.device):
    all_labels = []
    all_preds = []
    all_probs = []
    all_files = []

    with torch.no_grad():
        for batch_idx, (keypoints, labels) in enumerate(dataloader):
            keypoints = keypoints.to(device)
            logits = model(keypoints)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            start = batch_idx * dataloader.batch_size
            end = start + labels.size(0)
            batch_files = dataloader.dataset.df.iloc[start:end]["file"].astype(str).tolist()
            all_files.extend(batch_files)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_files,
    )


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, fall_probs: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, fall_probs)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close(fig)
    return float(auc_score)


def find_worst_mistakes(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, files: list[str]):
    false_positives = []
    false_negatives = []

    for true_label, pred_label, prob, file_path in zip(y_true, y_pred, probs, files):
        confidence = float(np.max(prob))
        record = {
            "file": file_path,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "confidence": confidence,
            "prob_no_fall": float(prob[0]),
            "prob_fall": float(prob[1]),
        }

        if true_label == 0 and pred_label == 1:
            false_positives.append(record)
        elif true_label == 1 and pred_label == 0:
            false_negatives.append(record)

    false_positives.sort(key=lambda item: item["confidence"], reverse=True)
    false_negatives.sort(key=lambda item: item["confidence"], reverse=True)
    return false_positives[:5], false_negatives[:5]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = FallDataset(TEST_CSV, KEYPOINT_DIR, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(device)
    y_true, y_pred, probs, files = predict(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    save_confusion_matrix(y_true, y_pred)
    auc_score = save_roc_curve(y_true, probs[:, 1])
    false_positives, false_negatives = find_worst_mistakes(y_true, y_pred, probs, files)

    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision_pc[idx]),
            "recall": float(recall_pc[idx]),
            "f1": float(f1_pc[idx]),
            "support": int(support_pc[idx]),
        }

    report = {
        "accuracy": float(accuracy),
        "weighted": {
            "precision": float(precision_w),
            "recall": float(recall_w),
            "f1": float(f1_w),
        },
        "per_class": per_class,
        "auc": auc_score,
        "false_positives_top5": false_positives,
        "false_negatives_top5": false_negatives,
        "confusion_matrix_png": str(CONFUSION_MATRIX_PATH),
        "roc_curve_png": str(ROC_CURVE_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision weighted: {precision_w:.4f}")
    print(f"Recall weighted: {recall_w:.4f}")
    print(f"F1 weighted: {f1_w:.4f}")
    print(f"AUC: {auc_score:.4f}")

    print("\nPer-class metrics:")
    for name, metrics in per_class.items():
        print(
            f"{name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )

    print("\nTop 5 false positives:")
    for item in false_positives:
        print(f"{item['confidence']:.4f} | {item['file']}")

    print("\nTop 5 false negatives:")
    for item in false_negatives:
        print(f"{item['confidence']:.4f} | {item['file']}")

    print(f"\nSaved confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Saved ROC curve: {ROC_CURVE_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
