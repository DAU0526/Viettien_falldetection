from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import FallDataset, get_class_weights
from model_bilstm import FallBiLSTM


TRAIN_CSV = Path("data") / "splits" / "train.csv"
VAL_CSV = Path("data") / "splits" / "val.csv"
KEYPOINT_DIR = Path("data") / "keypoints" / "coffee_room"
BEST_MODEL_PATH = Path("models") / "best_model.pth"

CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
    "input_size": 132,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for keypoints, labels in dataloader:
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(keypoints)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for keypoints, labels in dataloader:
            keypoints = keypoints.to(device)
            labels = labels.to(device)

            logits = model(keypoints)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / total_samples
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    return val_loss, val_accuracy, val_f1


def save_checkpoint(model, epoch, val_f1, val_accuracy):
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_f1": val_f1,
        "val_accuracy": val_accuracy,
        "config": CONFIG,
    }
    torch.save(checkpoint, BEST_MODEL_PATH)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = FallDataset(TRAIN_CSV, KEYPOINT_DIR, augment=True)
    val_dataset = FallDataset(VAL_CSV, KEYPOINT_DIR, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    model = FallBiLSTM(
        input_size=CONFIG["input_size"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    class_weights = get_class_weights(TRAIN_CSV).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
    )

    best_val_f1 = -1.0
    epochs_without_improvement = 0

    print(f"Train clips: {len(train_dataset)}")
    print(f"Val clips  : {len(val_dataset)}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_accuracy:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            save_checkpoint(model, epoch, val_f1, val_accuracy)
            print(f"Saved best checkpoint: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
            print("Early stopping: val F1 khong cai thien sau 10 epoch lien tiep")
            break

    print(f"Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
