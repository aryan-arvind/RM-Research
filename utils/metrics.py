"""
Metrics — Quality & Performance Measurement
=============================================
PSNR, SSIM, detection recovery rate, and stress curve computation.
"""

import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Higher = better reconstruction.

    Args:
        pred:   (C, H, W) predicted image in [0, max_val]
        target: (C, H, W) ground truth image in [0, max_val]

    Returns:
        PSNR in dB
    """
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0  # essentially perfect
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim_value(
    pred: np.ndarray, target: np.ndarray, max_val: float = 1.0
) -> float:
    """
    Structural Similarity Index.
    Higher = better structural preservation.

    Args:
        pred:   (C, H, W) predicted image
        target: (C, H, W) ground truth image

    Returns:
        SSIM value in [0, 1]
    """
    # skimage expects (H, W, C)
    if pred.ndim == 3 and pred.shape[0] <= 4:
        pred = pred.transpose(1, 2, 0)
        target = target.transpose(1, 2, 0)

    # Determine win_size based on smallest dimension
    min_dim = min(pred.shape[0], pred.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    return float(structural_similarity(
        pred, target,
        data_range=max_val,
        channel_axis=2 if pred.ndim == 3 else None,
        win_size=win_size,
    ))


def compute_detection_recovery_rate(
    clean_count: float,
    corrupted_count: float,
    hardened_count: float,
) -> float:
    """
    Detection Recovery Rate: how much of the detection loss does the
    hardener recover?

    Recovery = (hardened - corrupted) / (clean - corrupted)

    - 100% means full recovery to clean performance
    - 0% means hardening did nothing
    - >100% means hardening improved beyond clean baseline (rare)
    """
    drop = clean_count - corrupted_count
    if abs(drop) < 1e-8:
        return 100.0  # no drop, no recovery needed
    recovery = (hardened_count - corrupted_count) / drop
    return float(recovery * 100)


def compute_stress_curves(results: dict) -> dict:
    """
    Compute stress curves from evaluation results.

    A stress curve shows how detection performance degrades as
    corruption severity increases, and how much the hardener recovers.

    Returns per-corruption-type arrays of:
        severity → (mAP_clean, mAP_corrupted, mAP_hardened, recovery%)
    """
    clean_avg = np.mean(results["detection"]["clean"])
    corruption_names = [
        "haze", "blur", "jpeg", "noise",
        "radiometric_drift", "band_misalignment",
    ]

    curves = {}
    for cname in corruption_names:
        curve = []
        for sev in range(3):
            key = f"{cname}_sev{sev}"
            corr_dets = results["detection"]["corrupted"].get(key, [])
            hard_dets = results["detection"]["hardened"].get(key, [])

            avg_corr = np.mean(corr_dets) if corr_dets else 0
            avg_hard = np.mean(hard_dets) if hard_dets else 0

            curve.append({
                "severity": sev,
                "clean_avg": float(clean_avg),
                "corrupted_avg": float(avg_corr),
                "hardened_avg": float(avg_hard),
                "drop_pct": float(
                    (1 - avg_corr / clean_avg) * 100 if clean_avg > 0 else 0
                ),
                "recovery_pct": float(
                    compute_detection_recovery_rate(clean_avg, avg_corr, avg_hard)
                ),
            })
        curves[cname] = curve

    return curves


def compute_confusion_matrix(
    predicted: list[int], actual: list[int], num_classes: int
) -> np.ndarray:
    """Compute confusion matrix for corruption type classification."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, a in zip(predicted, actual):
        if 0 <= p < num_classes and 0 <= a < num_classes:
            cm[a, p] += 1
    return cm


def compute_per_class_metrics(confusion_matrix: np.ndarray) -> dict:
    """Compute precision, recall, F1 per class from confusion matrix."""
    num_classes = confusion_matrix.shape[0]
    metrics = {}
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        metrics[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return metrics
