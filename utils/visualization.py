"""
Visualization Utilities
========================
Publication-quality figures for:
  - Corruption comparison grids
  - Stress curves (mAP vs severity)
  - Grad-CAM heatmaps for corruption localization
  - Restoration before/after comparisons
  - Confusion matrices for diagnosis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


CORRUPTION_NAMES = [
    "Clean", "Haze", "Blur", "JPEG", "Noise",
    "Rad. Drift", "Band Misalign"
]
SEVERITY_LABELS = ["Low", "Medium", "High"]


def plot_corruption_grid(
    clean_image: np.ndarray,
    corrupted_images: dict,
    save_path: str = None,
    figsize: tuple = (20, 10),
):
    """
    Plot a grid showing clean image + all corruptions at all severities.
    corrupted_images: {(corruption_type, severity): image_array}
    """
    fig, axes = plt.subplots(3, 7, figsize=figsize)
    fig.suptitle("Corruption Injection Grid — SatCorrupt-Bench", fontsize=16, y=0.98)

    for sev in range(3):
        axes[sev, 0].imshow(clean_image)
        axes[sev, 0].set_title(f"Clean\n{SEVERITY_LABELS[sev]}", fontsize=9)
        axes[sev, 0].axis("off")

        for ctype in range(1, 7):
            key = (ctype, sev)
            img = corrupted_images.get(key, clean_image)
            axes[sev, ctype].imshow(img)
            axes[sev, ctype].set_title(
                f"{CORRUPTION_NAMES[ctype]}\n{SEVERITY_LABELS[sev]}",
                fontsize=9,
            )
            axes[sev, ctype].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stress_curves(
    stress_data: dict,
    save_path: str = None,
    figsize: tuple = (14, 8),
):
    """
    Plot stress curves: detection count vs corruption severity.
    Shows clean baseline, corrupted degradation, and hardened recovery.

    stress_data: output of compute_stress_curves()
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Detection Stress Curves — mAP vs Corruption Severity",
                 fontsize=14, y=1.0)

    corruption_list = list(stress_data.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for idx, cname in enumerate(corruption_list):
        ax = axes[idx // 3, idx % 3]
        curve = stress_data[cname]

        severities = [c["severity"] for c in curve]
        clean_vals = [c["clean_avg"] for c in curve]
        corr_vals = [c["corrupted_avg"] for c in curve]
        hard_vals = [c["hardened_avg"] for c in curve]

        ax.plot(severities, clean_vals, "g--", marker="o", label="Clean Baseline",
                linewidth=2)
        ax.plot(severities, corr_vals, "r-", marker="s", label="Corrupted",
                linewidth=2)
        ax.plot(severities, hard_vals, "b-", marker="^", label="Hardened",
                linewidth=2)

        # Shade recovery region
        ax.fill_between(
            severities, corr_vals, hard_vals,
            alpha=0.2, color="blue", label="Recovery",
        )

        ax.set_title(cname.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Severity Level")
        ax.set_ylabel("Avg Detections")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(SEVERITY_LABELS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_grad_cam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM",
    save_path: str = None,
    alpha: float = 0.4,
):
    """
    Overlay Grad-CAM heatmap on image to show corruption localization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_restoration_comparison(
    clean: np.ndarray,
    corrupted: np.ndarray,
    restored: np.ndarray,
    residual: np.ndarray = None,
    corruption_name: str = "",
    severity: int = 0,
    psnr: float = None,
    ssim: float = None,
    save_path: str = None,
):
    """
    Side-by-side comparison of clean → corrupted → restored images.
    """
    n_cols = 4 if residual is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(clean)
    axes[0].set_title("Clean (Ground Truth)")
    axes[0].axis("off")

    axes[1].imshow(corrupted)
    axes[1].set_title(f"Corrupted\n{corruption_name} (sev={severity})")
    axes[1].axis("off")

    title = "Restored (Hardened)"
    if psnr is not None:
        title += f"\nPSNR: {psnr:.2f} dB"
    if ssim is not None:
        title += f" | SSIM: {ssim:.4f}"
    axes[2].imshow(restored)
    axes[2].set_title(title)
    axes[2].axis("off")

    if residual is not None:
        # Normalize residual for visualization
        res_vis = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        axes[3].imshow(res_vis, cmap="RdBu_r")
        axes[3].set_title("Corruption Residual")
        axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = None,
    title: str = "Corruption Type Confusion Matrix",
    save_path: str = None,
    figsize: tuple = (10, 8),
):
    """Plot confusion matrix with values and color coding."""
    if class_names is None:
        class_names = CORRUPTION_NAMES[: cm.shape[0]]

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize by row for color
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Corruption",
        xlabel="Predicted Corruption",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate with counts
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_summary_dashboard(
    summary: dict,
    save_path: str = None,
    figsize: tuple = (18, 12),
):
    """
    Create a comprehensive dashboard showing all key results.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("SatCorrupt-Bench — Evaluation Dashboard", fontsize=16, y=0.98)

    corruption_list = list(summary.get("per_corruption", {}).keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(corruption_list)))

    # 1. Diagnosis Accuracy per corruption
    ax1 = fig.add_subplot(gs[0, 0])
    accs = [
        summary["per_corruption"][c]["diagnosis_accuracy"]
        for c in corruption_list
    ]
    bars = ax1.bar(range(len(corruption_list)), accs, color=colors)
    ax1.set_xticks(range(len(corruption_list)))
    ax1.set_xticklabels(
        [c.replace("_", "\n") for c in corruption_list], fontsize=8
    )
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Diagnosis Accuracy per Type")
    ax1.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{acc:.1%}", ha="center", fontsize=8)

    # 2. PSNR by corruption and severity
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    width = 0.12
    for i, cname in enumerate(corruption_list):
        psnrs = [
            summary["per_corruption"][cname]["severity_curves"][s]["psnr"]
            for s in range(3)
        ]
        ax2.bar(x + i * width, psnrs, width, label=cname.replace("_", " "),
                color=colors[i])
    ax2.set_xticks(x + width * len(corruption_list) / 2)
    ax2.set_xticklabels(SEVERITY_LABELS)
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Restoration PSNR")
    ax2.legend(fontsize=6, ncol=2)

    # 3. Detection drop % by severity
    ax3 = fig.add_subplot(gs[0, 2])
    for i, cname in enumerate(corruption_list):
        drops = [
            summary["per_corruption"][cname]["severity_curves"][s]["det_drop_pct"]
            for s in range(3)
        ]
        ax3.plot([0, 1, 2], drops, marker="o", label=cname.replace("_", " "),
                 color=colors[i], linewidth=2)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(SEVERITY_LABELS)
    ax3.set_ylabel("Detection Drop (%)")
    ax3.set_title("Detection Degradation")
    ax3.legend(fontsize=6, ncol=2)
    ax3.grid(True, alpha=0.3)

    # 4. Recovery rate by corruption
    ax4 = fig.add_subplot(gs[1, 0])
    for i, cname in enumerate(corruption_list):
        recoveries = [
            summary["per_corruption"][cname]["severity_curves"][s]["recovery_rate"]
            for s in range(3)
        ]
        ax4.bar(
            x + i * width, recoveries, width,
            label=cname.replace("_", " "), color=colors[i],
        )
    ax4.set_xticks(x + width * len(corruption_list) / 2)
    ax4.set_xticklabels(SEVERITY_LABELS)
    ax4.set_ylabel("Recovery Rate (%)")
    ax4.set_title("Hardening Recovery Rate")
    ax4.legend(fontsize=6, ncol=2)
    ax4.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Full Recovery")

    # 5. SSIM by corruption and severity
    ax5 = fig.add_subplot(gs[1, 1])
    for i, cname in enumerate(corruption_list):
        ssims = [
            summary["per_corruption"][cname]["severity_curves"][s]["ssim"]
            for s in range(3)
        ]
        ax5.plot([0, 1, 2], ssims, marker="s", label=cname.replace("_", " "),
                 color=colors[i], linewidth=2)
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(SEVERITY_LABELS)
    ax5.set_ylabel("SSIM")
    ax5.set_title("Structural Similarity (Restored)")
    ax5.legend(fontsize=6, ncol=2)
    ax5.grid(True, alpha=0.3)

    # 6. Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    clean_avg = summary.get("clean_avg_detections", 0)
    text = f"Clean Baseline: {clean_avg:.1f} avg detections\n\n"
    text += "Best Recovery Rates:\n"
    for cname in corruption_list:
        best_rec = max(
            summary["per_corruption"][cname]["severity_curves"][s]["recovery_rate"]
            for s in range(3)
        )
        text += f"  {cname}: {best_rec:.1f}%\n"
    ax6.text(0.1, 0.9, text, transform=ax6.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
    ax6.set_title("Summary")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
