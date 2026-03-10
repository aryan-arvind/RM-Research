"""
End-to-End Evaluation Pipeline
================================
Runs the full SatCorrupt-Bench evaluation:
1. Generate corrupted images at all severity levels
2. Diagnose corruption with the Diagnosis CNN
3. Restore with the Hardening Autoencoder
4. Evaluate with YOLOv8 (before and after hardening)
5. Compute stress curves and recovery metrics
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config.config import ExperimentConfig
from corruption.injector import CorruptionInjector, CorruptionBenchmark
from models.diagnosis_cnn import DiagnosisCNN
from models.autoencoder import HardeningAutoencoder
from models.yolo_evaluator import YOLOv8Evaluator
from data.dataset import load_images_from_directory
from utils.metrics import (
    compute_psnr,
    compute_ssim_value,
    compute_detection_recovery_rate,
)


def run_full_evaluation(
    config: ExperimentConfig,
    image_dir: str,
    diagnosis_ckpt: str,
    autoencoder_ckpt: str,
    yolo_weights: str = None,
    max_images: int = 100,
) -> dict:
    """
    Run the complete SatCorrupt-Bench evaluation pipeline.

    Returns comprehensive results dictionary with:
    - Diagnosis accuracy per corruption type
    - PSNR/SSIM restoration quality per corruption type × severity
    - Detection count before/after hardening (stress curves)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # ---- Load models ----
    print("Loading models...")

    # Diagnosis CNN
    diagnosis_model = DiagnosisCNN(config.diagnosis).to(device)
    diag_ckpt = torch.load(diagnosis_ckpt, map_location=device, weights_only=False)
    diagnosis_model.load_state_dict(diag_ckpt["model_state_dict"])
    diagnosis_model.eval()

    # Hardening Autoencoder
    hardener = HardeningAutoencoder(config.autoencoder).to(device)
    ae_ckpt = torch.load(autoencoder_ckpt, map_location=device, weights_only=False)
    hardener.load_state_dict(ae_ckpt["model_state_dict"])
    hardener.eval()

    # YOLOv8 Evaluator
    yolo_eval = YOLOv8Evaluator(config.yolo)
    yolo_eval.load_model(yolo_weights)

    # Corruption injector
    injector = CorruptionInjector(config.corruption)
    corruption_names = config.corruption.corruption_names

    # ---- Load test images ----
    print(f"Loading images from {image_dir}...")
    raw = load_images_from_directory(
        image_dir, target_size=config.corruption.image_size,
        max_images=max_images,
    )
    clean_images = [img for img, _ in raw]
    filenames = [name for _, name in raw]
    print(f"Loaded {len(clean_images)} test images")

    # ---- Evaluate ----
    results = {
        "diagnosis": {},
        "restoration": {},
        "detection": {"clean": [], "corrupted": {}, "hardened": {}},
    }

    # Clean detection baseline
    print("Evaluating clean images...")
    for img in tqdm(clean_images, desc="Clean detection"):
        dets = yolo_eval.predict_single(img)
        results["detection"]["clean"].append(len(dets))

    # Per corruption type × severity
    for ctype in range(1, 7):
        cname = corruption_names[ctype]
        results["diagnosis"][cname] = {"correct": 0, "total": 0, "per_severity": {}}
        results["restoration"][cname] = {}

        for sev in range(3):
            key = f"{cname}_sev{sev}"
            psnr_list, ssim_list = [], []
            det_corrupted, det_hardened = [], []
            diag_correct = 0

            print(f"\nEvaluating {cname} severity {sev}...")

            for img in tqdm(clean_images, desc=f"{cname}_s{sev}"):
                corrupted = injector.inject(img, ctype, sev)

                # --- Diagnosis ---
                c_tensor = (
                    torch.from_numpy(corrupted).permute(2, 0, 1).float() / 255.0
                )
                c_tensor = c_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    type_logits, sev_logits = diagnosis_model(c_tensor)
                pred_type = type_logits.argmax(1).item()
                if pred_type == ctype:
                    diag_correct += 1

                # --- Restoration ---
                with torch.no_grad():
                    restored_tensor, _ = hardener(c_tensor)
                restored_np = (
                    restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                )
                clean_norm = img.astype(np.float32) / 255.0

                psnr_list.append(compute_psnr(
                    restored_np.transpose(2, 0, 1),
                    clean_norm.transpose(2, 0, 1),
                ))
                ssim_list.append(compute_ssim_value(
                    restored_np.transpose(2, 0, 1),
                    clean_norm.transpose(2, 0, 1),
                ))

                # --- Detection ---
                det_c = yolo_eval.predict_single(corrupted)
                det_corrupted.append(len(det_c))

                restored_uint8 = (restored_np * 255).clip(0, 255).astype(np.uint8)
                det_h = yolo_eval.predict_single(restored_uint8)
                det_hardened.append(len(det_h))

            # Store results
            n = len(clean_images)
            results["diagnosis"][cname]["correct"] += diag_correct
            results["diagnosis"][cname]["total"] += n
            results["diagnosis"][cname]["per_severity"][sev] = {
                "accuracy": diag_correct / n,
            }

            results["restoration"][key] = {
                "psnr_mean": float(np.mean(psnr_list)),
                "psnr_std": float(np.std(psnr_list)),
                "ssim_mean": float(np.mean(ssim_list)),
                "ssim_std": float(np.std(ssim_list)),
            }

            results["detection"]["corrupted"][key] = det_corrupted
            results["detection"]["hardened"][key] = det_hardened

    # ---- Compute summary statistics ----
    results["summary"] = compute_summary(results, corruption_names)

    return results


def compute_summary(results: dict, corruption_names: tuple) -> dict:
    """Compute high-level summary metrics from evaluation results."""
    clean_avg = np.mean(results["detection"]["clean"])

    summary = {
        "clean_avg_detections": float(clean_avg),
        "per_corruption": {},
    }

    for ctype in range(1, 7):
        cname = corruption_names[ctype]
        diag_info = results["diagnosis"][cname]
        diag_acc = (diag_info["correct"] / diag_info["total"]
                    if diag_info["total"] > 0 else 0)

        corruption_summary = {
            "diagnosis_accuracy": diag_acc,
            "severity_curves": {},
        }

        for sev in range(3):
            key = f"{cname}_sev{sev}"
            rest = results["restoration"].get(key, {})
            det_corr = results["detection"]["corrupted"].get(key, [])
            det_hard = results["detection"]["hardened"].get(key, [])

            avg_corr = np.mean(det_corr) if det_corr else 0
            avg_hard = np.mean(det_hard) if det_hard else 0

            corruption_summary["severity_curves"][sev] = {
                "psnr": rest.get("psnr_mean", 0),
                "ssim": rest.get("ssim_mean", 0),
                "det_corrupted_avg": float(avg_corr),
                "det_hardened_avg": float(avg_hard),
                "det_drop_pct": float(
                    (1 - avg_corr / clean_avg) * 100 if clean_avg > 0 else 0
                ),
                "recovery_rate": float(
                    compute_detection_recovery_rate(
                        clean_avg, avg_corr, avg_hard
                    )
                ),
            }

        summary["per_corruption"][cname] = corruption_summary

    return summary
