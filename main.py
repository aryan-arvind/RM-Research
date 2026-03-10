"""
Adaptive Preprocessing Hardening for Satellite Imagery
=======================================================
Main orchestrator: end-to-end pipeline from raw data to benchmark results.

Usage:
    # Train diagnosis CNN
    python main.py --mode train_diagnosis --image_dir ./datasets/DOTA/train/images

    # Train hardening autoencoder
    python main.py --mode train_autoencoder --image_dir ./datasets/DOTA/train/images

    # Fine-tune YOLOv8 on clean data
    python main.py --mode train_yolo --dataset_yaml ./datasets/DOTA/data.yaml

    # Run full SatCorrupt-Bench evaluation
    python main.py --mode evaluate --image_dir ./datasets/DOTA/val/images

    # Generate corruption samples for visualization
    python main.py --mode visualize --image_dir ./datasets/DOTA/val/images

    # Run everything end-to-end
    python main.py --mode full_pipeline --image_dir ./datasets/DOTA/train/images
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import get_config, ExperimentConfig
from corruption.injector import CorruptionInjector
from preprocessing.pipeline import SatellitePreprocessor
from data.dataset import load_images_from_directory
from models.diagnosis_cnn import DiagnosisCNN
from models.autoencoder import HardeningAutoencoder
from models.yolo_evaluator import YOLOv8Evaluator, prepare_yolo_dataset
from training.train_diagnosis import train_diagnosis_cnn
from training.train_autoencoder import train_autoencoder
from training.evaluate import run_full_evaluation
from utils.metrics import compute_stress_curves
from utils.visualization import (
    plot_corruption_grid,
    plot_stress_curves,
    plot_restoration_comparison,
    plot_summary_dashboard,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mode_train_diagnosis(config: ExperimentConfig, args):
    """Train the Diagnosis CNN."""
    print("=" * 60)
    print("PHASE 1: Training Diagnosis CNN")
    print("=" * 60)
    train_diagnosis_cnn(config, args.image_dir, max_images=args.max_images)


def mode_train_autoencoder(config: ExperimentConfig, args):
    """Train the Hardening Autoencoder."""
    print("=" * 60)
    print("PHASE 2: Training Hardening Autoencoder")
    print("=" * 60)
    train_autoencoder(config, args.image_dir, max_images=args.max_images)


def mode_train_yolo(config: ExperimentConfig, args):
    """Fine-tune YOLOv8 on clean satellite data."""
    print("=" * 60)
    print("PHASE 3: Fine-tuning YOLOv8")
    print("=" * 60)

    if not args.dataset_yaml:
        print("ERROR: --dataset_yaml required for YOLO training")
        return

    evaluator = YOLOv8Evaluator(config.yolo)
    evaluator.load_model()
    evaluator.fine_tune(args.dataset_yaml)


def mode_evaluate(config: ExperimentConfig, args):
    """Run full SatCorrupt-Bench evaluation."""
    print("=" * 60)
    print("PHASE 4: Running SatCorrupt-Bench Evaluation")
    print("=" * 60)

    diag_ckpt = str(config.paths.checkpoint_dir / "diagnosis_cnn" / "best_model.pt")
    ae_ckpt = str(config.paths.checkpoint_dir / "autoencoder" / "best_model.pt")

    results = run_full_evaluation(
        config=config,
        image_dir=args.image_dir,
        diagnosis_ckpt=diag_ckpt,
        autoencoder_ckpt=ae_ckpt,
        yolo_weights=args.yolo_weights,
        max_images=args.max_images,
    )

    # Save results
    output_dir = config.paths.output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    # Figures
    if "summary" in results:
        plot_summary_dashboard(
            results["summary"],
            save_path=str(output_dir / "dashboard.png"),
        )

        stress_curves = compute_stress_curves(results)
        plot_stress_curves(
            stress_curves,
            save_path=str(output_dir / "stress_curves.png"),
        )

    print(f"\nResults saved to {output_dir}")
    return results


def mode_visualize(config: ExperimentConfig, args):
    """Generate corruption visualization samples."""
    print("=" * 60)
    print("Generating Corruption Visualizations")
    print("=" * 60)

    output_dir = config.paths.output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    injector = CorruptionInjector(config.corruption)

    # Load a few images
    raw = load_images_from_directory(
        args.image_dir,
        target_size=config.corruption.image_size,
        max_images=5,
    )

    for img, fname in raw:
        print(f"Processing {fname}...")

        # Generate all corruptions
        corrupted_dict = {}
        for ctype in range(1, 7):
            for sev in range(3):
                corrupted = injector.inject(img, ctype, sev)
                corrupted_dict[(ctype, sev)] = corrupted

        # Corruption grid
        plot_corruption_grid(
            img, corrupted_dict,
            save_path=str(output_dir / f"{Path(fname).stem}_corruption_grid.png"),
        )

    print(f"Visualizations saved to {output_dir}")


def mode_full_pipeline(config: ExperimentConfig, args):
    """Run the complete pipeline end-to-end."""
    print("=" * 60)
    print("FULL PIPELINE — Adaptive Preprocessing Hardening")
    print("=" * 60)
    print()

    # Phase 1: Train Diagnosis CNN
    mode_train_diagnosis(config, args)

    # Phase 2: Train Autoencoder
    mode_train_autoencoder(config, args)

    # Phase 3: Fine-tune YOLO (if dataset_yaml provided)
    if args.dataset_yaml:
        mode_train_yolo(config, args)

    # Phase 4: Evaluate
    results = mode_evaluate(config, args)

    # Phase 5: Visualize
    mode_visualize(config, args)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    if results and "summary" in results:
        summary = results["summary"]
        print(f"\nClean baseline: {summary['clean_avg_detections']:.1f} avg detections")
        for cname, data in summary.get("per_corruption", {}).items():
            diag_acc = data["diagnosis_accuracy"]
            best_rec = max(
                data["severity_curves"][s]["recovery_rate"] for s in range(3)
            )
            print(f"  {cname}: diagnosis acc={diag_acc:.1%}, "
                  f"best recovery={best_rec:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Preprocessing Hardening for Satellite Imagery"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train_diagnosis", "train_autoencoder", "train_yolo",
            "evaluate", "visualize", "full_pipeline",
        ],
        help="Pipeline mode to run",
    )
    parser.add_argument("--image_dir", type=str, help="Path to satellite images")
    parser.add_argument("--dataset_yaml", type=str, help="YOLO dataset YAML path")
    parser.add_argument("--yolo_weights", type=str, help="YOLOv8 weights path")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Max images for evaluation")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)

    args = parser.parse_args()

    # Build config
    config = get_config()
    config.device = args.device
    config.seed = args.seed
    config.corruption.image_size = args.image_size

    if args.batch_size:
        config.diagnosis.batch_size = args.batch_size
        config.autoencoder.batch_size = args.batch_size
    if args.epochs:
        config.diagnosis.epochs = args.epochs
        config.autoencoder.epochs = args.epochs
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    elif os.name == "nt":
        # Safer default on Windows where multiprocessing workers can be brittle.
        config.num_workers = 0

    set_seed(config.seed)

    # Validate
    if args.mode != "train_yolo" and not args.image_dir:
        parser.error(f"--image_dir is required for mode '{args.mode}'")

    # Dispatch
    modes = {
        "train_diagnosis": mode_train_diagnosis,
        "train_autoencoder": mode_train_autoencoder,
        "train_yolo": mode_train_yolo,
        "evaluate": mode_evaluate,
        "visualize": mode_visualize,
        "full_pipeline": mode_full_pipeline,
    }
    modes[args.mode](config, args)


if __name__ == "__main__":
    main()
