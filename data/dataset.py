"""
Dataset Loaders for Satellite Imagery Benchmarks
=================================================
Supports: DOTA, xView, EuroSAT, and custom satellite image directories.
Each loader produces clean images that the corruption injector can process.
Also provides PyTorch Dataset classes for training the Diagnosis CNN
and the Hardening Autoencoder.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from corruption.injector import CorruptionInjector
from config.config import CorruptionConfig


# ---------------------------------------------------------------
# Generic satellite image loader (works for any image directory)
# ---------------------------------------------------------------


def load_images_from_directory(
    image_dir: str,
    target_size: int = 512,
    max_images: Optional[int] = None,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> list[tuple[np.ndarray, str]]:
    """
    Load images from a flat directory.
    Returns list of (image_array, filename) tuples.
    """
    image_dir = Path(image_dir)
    # Search recursively to handle datasets with class subdirectories (e.g. EuroSAT)
    files = sorted(
        f for f in image_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )
    if max_images:
        files = files[:max_images]

    images = []
    for fpath in files:
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if target_size:
            img = cv2.resize(img, (target_size, target_size))
        images.append((img, fpath.name))
    return images


# ---------------------------------------------------------------
# DOTA Dataset Loader
# ---------------------------------------------------------------


class DOTADataset:
    """
    Loader for DOTA (Dataset for Object Detection in Aerial Images).
    Expected structure:
        dota_root/
            images/
                P0001.png
                ...
            labelTxt/
                P0001.txt
                ...
    DOTA labels: x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    """

    def __init__(self, root_dir: str, target_size: int = 512, split: str = "train"):
        self.root = Path(root_dir)
        self.image_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labelTxt"
        self.target_size = target_size
        self.image_files = sorted(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_files)

    def load_image(self, idx: int) -> tuple[np.ndarray, list[dict]]:
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Parse labels
        label_path = self.label_dir / (img_path.stem + ".txt")
        labels = self._parse_dota_labels(label_path, img.shape)

        img = cv2.resize(img, (self.target_size, self.target_size))
        return img, labels

    @staticmethod
    def _parse_dota_labels(label_path: Path, img_shape: tuple) -> list[dict]:
        labels = []
        if not label_path.exists():
            return labels
        h, w = img_shape[:2]
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(x) for x in parts[:8]]
                except ValueError:
                    continue
                category = parts[8]
                difficulty = int(parts[9]) if len(parts) > 9 else 0
                # Convert oriented bbox to axis-aligned for YOLO
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs) / w, max(xs) / w
                y_min, y_max = min(ys) / h, max(ys) / h
                labels.append({
                    "category": category,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "difficulty": difficulty,
                })
        return labels


# ---------------------------------------------------------------
# xView Dataset Loader
# ---------------------------------------------------------------


class XViewDataset:
    """
    Loader for xView satellite dataset (used in military detection papers).
    Expected: images directory + geojson or csv annotations.
    """

    def __init__(self, root_dir: str, target_size: int = 512):
        self.root = Path(root_dir)
        self.image_dir = self.root / "images"
        self.target_size = target_size
        self.image_files = sorted(
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in (".tif", ".png", ".jpg")
        )

    def __len__(self):
        return len(self.image_files)

    def load_image(self, idx: int) -> np.ndarray:
        img = cv2.imread(str(self.image_files[idx]))
        if img is None:
            raise ValueError(f"Failed to load {self.image_files[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.target_size, self.target_size))


# ---------------------------------------------------------------
# EuroSAT Dataset Loader
# ---------------------------------------------------------------


class EuroSATDataset:
    """
    Loader for EuroSAT dataset (land-use classification from Sentinel-2).
    Expected structure:
        eurosat_root/
            AnnualCrop/
            Forest/
            HerbaceousVegetation/
            ...
    """

    CLASS_NAMES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]

    def __init__(self, root_dir: str, target_size: int = 64):
        self.root = Path(root_dir)
        self.target_size = target_size
        self.samples = []
        for cls_idx, cls_name in enumerate(self.CLASS_NAMES):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for fpath in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((fpath, cls_idx))

    def __len__(self):
        return len(self.samples)

    def load_image(self, idx: int) -> tuple[np.ndarray, int]:
        fpath, cls_idx = self.samples[idx]
        img = cv2.imread(str(fpath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))
        return img, cls_idx


# ---------------------------------------------------------------
# PyTorch Dataset: Diagnosis CNN Training
# ---------------------------------------------------------------


class CorruptionDiagnosisDataset(Dataset):
    """
    Generates corrupted images on-the-fly from clean images.
    Each sample: (corrupted_image_tensor, corruption_type, severity)
    Used for training the Diagnosis CNN.
    """

    def __init__(
        self,
        image_dir: str,
        corruption_config: Optional[CorruptionConfig] = None,
        target_size: int = 512,
        max_images: Optional[int] = None,
        include_clean: bool = True,
    ):
        self.target_size = target_size
        self.config = corruption_config or CorruptionConfig()
        self.injector = CorruptionInjector(self.config)
        self.include_clean = include_clean

        # Load all clean images
        raw = load_images_from_directory(image_dir, target_size, max_images)
        self.clean_images = [img for img, _ in raw]
        self.filenames = [name for _, name in raw]

        # Pre-compute index mapping:
        # Each clean image → 7 types × 3 severities (or 6×3 + 1 with clean)
        num_types = 7 if include_clean else 6
        self.samples_per_image = num_types * 3 if not include_clean else 6 * 3 + 1
        self.total_samples = len(self.clean_images) * self.samples_per_image

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        variant_idx = idx % self.samples_per_image

        clean = self.clean_images[img_idx]

        if self.include_clean and variant_idx == 0:
            # Clean sample
            corruption_type = 0
            severity = 0
            image = clean.copy()
        else:
            # Determine corruption type and severity
            adj_idx = variant_idx - 1 if self.include_clean else variant_idx
            corruption_type = (adj_idx // 3) + 1  # 1..6
            severity = adj_idx % 3                 # 0..2
            image = self.injector.inject(clean, corruption_type, severity)

        # Convert to tensor: HWC uint8 → CHW float32
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return tensor, corruption_type, severity


# ---------------------------------------------------------------
# PyTorch Dataset: Autoencoder Training (Clean-Corrupt Pairs)
# ---------------------------------------------------------------


class AutoencoderPairDataset(Dataset):
    """
    Produces (corrupted_image, clean_image) pairs for training the hardener.
    Corruption is applied on-the-fly with random type and severity.
    """

    def __init__(
        self,
        image_dir: str,
        corruption_config: Optional[CorruptionConfig] = None,
        target_size: int = 512,
        max_images: Optional[int] = None,
        augment_factor: int = 6,
    ):
        self.target_size = target_size
        self.config = corruption_config or CorruptionConfig()
        self.injector = CorruptionInjector(self.config)
        self.augment_factor = augment_factor

        raw = load_images_from_directory(image_dir, target_size, max_images)
        self.clean_images = [img for img, _ in raw]

    def __len__(self):
        return len(self.clean_images) * self.augment_factor

    def __getitem__(self, idx):
        img_idx = idx // self.augment_factor
        clean = self.clean_images[img_idx]

        # Random corruption
        corrupted, ctype, severity = self.injector.inject_random(clean)

        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        corrupt_t = torch.from_numpy(corrupted).permute(2, 0, 1).float() / 255.0

        return corrupt_t, clean_t, ctype, severity


# ---------------------------------------------------------------
# PyTorch Dataset: YOLO Evaluation (with corruption support)
# ---------------------------------------------------------------


class YOLOEvalDataset(Dataset):
    """
    Wraps images for YOLO evaluation. Supports injecting corruption
    to measure mAP degradation.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str] = None,
        corruption_config: Optional[CorruptionConfig] = None,
        target_size: int = 640,
        corruption_type: int = 0,
        severity: int = 0,
    ):
        self.target_size = target_size
        self.config = corruption_config or CorruptionConfig()
        self.injector = CorruptionInjector(self.config)
        self.corruption_type = corruption_type
        self.severity = severity

        raw = load_images_from_directory(image_dir, target_size)
        self.images = [img for img, _ in raw]
        self.filenames = [name for _, name in raw]

        # Load YOLO-format labels if available
        self.labels = []
        if label_dir:
            label_path = Path(label_dir)
            for fname in self.filenames:
                lbl_file = label_path / (Path(fname).stem + ".txt")
                if lbl_file.exists():
                    self.labels.append(self._parse_yolo_label(lbl_file))
                else:
                    self.labels.append([])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.corruption_type > 0:
            img = self.injector.inject(img, self.corruption_type, self.severity)

        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels = self.labels[idx] if self.labels else []
        return tensor, labels, self.filenames[idx]

    @staticmethod
    def _parse_yolo_label(label_path: Path) -> list[list[float]]:
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    labels.append([cls_id] + coords)
        return labels


# ---------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------


def create_dataloaders(
    image_dir: str,
    dataset_type: str = "diagnosis",
    corruption_config: Optional[CorruptionConfig] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.8,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders for any dataset type.
    """
    config = corruption_config or CorruptionConfig()

    if dataset_type == "diagnosis":
        full_dataset = CorruptionDiagnosisDataset(
            image_dir, config, **kwargs
        )
    elif dataset_type == "autoencoder":
        full_dataset = AutoencoderPairDataset(
            image_dir, config, **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Train/val split
    n = len(full_dataset)
    n_train = int(n * train_split)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
