"""
Adaptive Preprocessing Hardening for Satellite Imagery
=======================================================
Central configuration for all pipeline components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    root: Path = Path(".")
    data_dir: Path = Path("./datasets")
    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        for p in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class CorruptionConfig:
    """Parameters for physically-grounded corruption injection."""
    image_size: int = 512

    # Atmospheric haze (ASM model: I(x) = J(x)*t(x) + A*(1-t(x)))
    haze_beta_range: tuple = (0.4, 0.8, 1.5)        # low, med, high
    haze_airlight_range: tuple = (0.7, 0.85, 0.95)

    # Gaussian blur (PSF modeling atmospheric turbulence)
    blur_sigma_levels: tuple = (1.0, 3.0, 5.0)

    # JPEG compression artifacts (DCT quantization)
    jpeg_quality_levels: tuple = (50, 30, 10)         # low, med, high corruption

    # Sensor noise (Gaussian + shot noise)
    gaussian_noise_std: tuple = (10.0, 25.0, 50.0)
    shot_noise_scale: tuple = (0.02, 0.05, 0.1)

    # Radiometric drift (gain/offset perturbation)
    gain_range: tuple = (0.05, 0.15, 0.3)
    offset_range: tuple = (5.0, 15.0, 30.0)

    # Band misalignment (sub-pixel shift)
    band_shift_pixels: tuple = (0.5, 1.0, 2.0)

    # Number of corruption types (clean + 6 corruptions)
    num_corruption_types: int = 7
    num_severity_levels: int = 3
    corruption_names: tuple = (
        "clean", "haze", "blur", "jpeg", "noise",
        "radiometric_drift", "band_misalignment"
    )


@dataclass
class DiagnosisCNNConfig:
    """Diagnosis CNN: multi-task classification of corruption type + severity."""
    backbone: str = "resnet18"
    in_channels: int = 3
    num_corruption_classes: int = 7
    num_severity_classes: int = 3
    dropout: float = 0.3
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 32
    type_loss_weight: float = 1.0
    severity_loss_weight: float = 0.5


@dataclass
class AutoencoderConfig:
    """Hardening autoencoder with Kick (Shift-N-Overlap) decoder."""
    in_channels: int = 3
    base_filters: int = 64
    latent_dim: int = 256
    num_encoder_blocks: int = 4
    use_skip_connections: bool = True
    # Kick decoder parameters (from Shift-N-Overlap paper)
    kick_num_branches: int = 3
    kick_shift_pixels: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 80
    batch_size: int = 16
    # Loss weights
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1
    ssim_weight: float = 0.5


@dataclass
class YOLOConfig:
    """YOLOv8 downstream evaluator configuration."""
    model_variant: str = "yolov8m.pt"
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    dataset_yaml: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    experiment_name: str = "adaptive_preprocessing_hardening"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    use_wandb: bool = False
    wandb_project: str = "sat-preprocessing-hardening"

    paths: PathConfig = field(default_factory=PathConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    diagnosis: DiagnosisCNNConfig = field(default_factory=DiagnosisCNNConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)


def get_config(**overrides) -> ExperimentConfig:
    """Create config with optional overrides."""
    config = ExperimentConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
