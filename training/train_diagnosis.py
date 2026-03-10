"""
Training Loop — Diagnosis CNN
==============================
Multi-task training for corruption type + severity classification.
Includes learning rate scheduling, early stopping, and gradient clipping.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config.config import ExperimentConfig
from data.dataset import create_dataloaders
from models.diagnosis_cnn import DiagnosisCNN, DiagnosisLoss


def train_diagnosis_cnn(
    config: ExperimentConfig,
    image_dir: str,
    max_images: int | None = None,
):
    """
    Full training pipeline for the Diagnosis CNN.

    Args:
        config: experiment configuration
        image_dir: path to directory of clean satellite images
        max_images: optional cap on number of clean images to load
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Training Diagnosis CNN on {device}")

    # ---- Data ----
    train_loader, val_loader = create_dataloaders(
        image_dir=image_dir,
        dataset_type="diagnosis",
        corruption_config=config.corruption,
        batch_size=config.diagnosis.batch_size,
        num_workers=config.num_workers,
        target_size=config.corruption.image_size,
        max_images=max_images,
    )
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")

    # ---- Model ----
    model = DiagnosisCNN(config.diagnosis).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=config.diagnosis.learning_rate,
        weight_decay=config.diagnosis.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.diagnosis.epochs, eta_min=1e-6
    )

    # ---- Loss ----
    criterion = DiagnosisLoss(config.diagnosis).to(device)

    # ---- Training Loop ----
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    checkpoint_dir = config.paths.checkpoint_dir / "diagnosis_cnn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.diagnosis.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_type_correct = 0
        train_sev_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.diagnosis.epochs}")
        for images, type_labels, sev_labels in pbar:
            images = images.to(device)
            type_labels = type_labels.to(device)
            sev_labels = sev_labels.to(device)

            optimizer.zero_grad()
            type_logits, sev_logits = model(images)
            total_loss, type_loss, sev_loss = criterion(
                type_logits, sev_logits, type_labels, sev_labels
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * images.size(0)
            train_type_correct += (type_logits.argmax(1) == type_labels).sum().item()
            train_sev_correct += (sev_logits.argmax(1) == sev_labels).sum().item()
            train_total += images.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "type_acc": f"{train_type_correct/train_total:.3f}",
                "sev_acc": f"{train_sev_correct/train_total:.3f}",
            })

        scheduler.step()

        train_loss /= train_total
        train_type_acc = train_type_correct / train_total
        train_sev_acc = train_sev_correct / train_total

        # --- Validate ---
        val_metrics = validate_diagnosis(model, val_loader, criterion, device)

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Type Acc: {train_type_acc:.3f} | Sev Acc: {train_sev_acc:.3f}"
        )
        print(
            f"  Val Loss:   {val_metrics['loss']:.4f} | "
            f"Type Acc: {val_metrics['type_acc']:.3f} | "
            f"Sev Acc: {val_metrics['sev_acc']:.3f}"
        )

        # --- Checkpointing ---
        combined_acc = (val_metrics["type_acc"] + val_metrics["sev_acc"]) / 2
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config.diagnosis,
            }, checkpoint_dir / "best_model.pt")
            print(f"  Saved best model (combined acc: {combined_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.3f}")
    return model


def validate_diagnosis(model, val_loader, criterion, device):
    """Run validation and return metrics."""
    model.eval()
    val_loss = 0.0
    type_correct = 0
    sev_correct = 0
    total = 0

    with torch.no_grad():
        for images, type_labels, sev_labels in val_loader:
            images = images.to(device)
            type_labels = type_labels.to(device)
            sev_labels = sev_labels.to(device)

            type_logits, sev_logits = model(images)
            loss, _, _ = criterion(
                type_logits, sev_logits, type_labels, sev_labels
            )

            val_loss += loss.item() * images.size(0)
            type_correct += (type_logits.argmax(1) == type_labels).sum().item()
            sev_correct += (sev_logits.argmax(1) == sev_labels).sum().item()
            total += images.size(0)

    return {
        "loss": val_loss / total,
        "type_acc": type_correct / total,
        "sev_acc": sev_correct / total,
    }
