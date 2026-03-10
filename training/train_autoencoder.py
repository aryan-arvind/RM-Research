"""
Training Loop — Hardening Autoencoder
======================================
Trains the U-Net + Kick decoder to restore corrupted satellite images.
Uses L1 + Perceptual + SSIM composite loss.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from config.config import ExperimentConfig
from data.dataset import create_dataloaders
from models.autoencoder import HardeningAutoencoder, HardeningLoss
from utils.metrics import compute_psnr, compute_ssim_value


def train_autoencoder(
    config: ExperimentConfig,
    image_dir: str,
    max_images: int | None = None,
):
    """
    Full training pipeline for the Hardening Autoencoder.

    Args:
        config: experiment configuration
        image_dir: path to directory of clean satellite images
        max_images: optional cap on number of clean images to load
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Training Hardening Autoencoder on {device}")

    # ---- Data ----
    train_loader, val_loader = create_dataloaders(
        image_dir=image_dir,
        dataset_type="autoencoder",
        corruption_config=config.corruption,
        batch_size=config.autoencoder.batch_size,
        num_workers=config.num_workers,
        target_size=config.corruption.image_size,
        max_images=max_images,
    )
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")

    # ---- Model ----
    model = HardeningAutoencoder(config.autoencoder).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=config.autoencoder.learning_rate,
        weight_decay=config.autoencoder.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ---- Loss ----
    criterion = HardeningLoss(config.autoencoder).to(device)

    # ---- Training Loop ----
    best_val_psnr = 0.0
    patience_counter = 0
    patience = 15

    checkpoint_dir = config.paths.checkpoint_dir / "autoencoder"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.autoencoder.epochs):
        # --- Train ---
        model.train()
        train_metrics = {"loss": 0, "l1": 0, "perceptual": 0, "ssim": 0}
        train_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.autoencoder.epochs}",
        )
        for corrupted, clean, ctype, severity in pbar:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            restored, residual = model(corrupted)
            total_loss, l1_loss, perc_loss, ssim_loss = criterion(restored, clean)

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = corrupted.size(0)
            train_metrics["loss"] += total_loss.item() * bs
            train_metrics["l1"] += l1_loss.item() * bs
            train_metrics["perceptual"] += perc_loss.item() * bs
            train_metrics["ssim"] += ssim_loss.item() * bs
            train_total += bs

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "l1": f"{l1_loss.item():.4f}",
            })

        scheduler.step()

        for k in train_metrics:
            train_metrics[k] /= train_total

        # --- Validate ---
        val_metrics = validate_autoencoder(model, val_loader, criterion, device)

        print(
            f"  Train — Loss: {train_metrics['loss']:.4f} | "
            f"L1: {train_metrics['l1']:.4f} | "
            f"SSIM Loss: {train_metrics['ssim']:.4f}"
        )
        print(
            f"  Val   — Loss: {val_metrics['loss']:.4f} | "
            f"PSNR: {val_metrics['psnr']:.2f} dB | "
            f"SSIM: {val_metrics['ssim_value']:.4f}"
        )

        # --- Checkpointing ---
        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config.autoencoder,
            }, checkpoint_dir / "best_model.pt")
            print(f"  Saved best model (PSNR: {best_val_psnr:.2f} dB)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best validation PSNR: {best_val_psnr:.2f} dB")
    return model


def validate_autoencoder(model, val_loader, criterion, device):
    """Run validation and return metrics including PSNR and SSIM."""
    model.eval()
    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total = 0

    with torch.no_grad():
        for corrupted, clean, ctype, severity in val_loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            restored, residual = model(corrupted)
            loss, _, _, _ = criterion(restored, clean)

            bs = corrupted.size(0)
            val_loss += loss.item() * bs

            # Per-image quality metrics
            for i in range(bs):
                psnr = compute_psnr(
                    restored[i].cpu().numpy(),
                    clean[i].cpu().numpy(),
                )
                ssim_val = compute_ssim_value(
                    restored[i].cpu().numpy(),
                    clean[i].cpu().numpy(),
                )
                total_psnr += psnr
                total_ssim += ssim_val
            total += bs

    return {
        "loss": val_loss / total,
        "psnr": total_psnr / total,
        "ssim_value": total_ssim / total,
    }
