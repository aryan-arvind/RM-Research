"""
Corruption Injection Module — Physically-Grounded Preprocessing Failures
=========================================================================
Each corruption models a real physical phenomenon that occurs in satellite
image preprocessing pipelines. This is NOT random noise — each function is
a mathematical model of a specific failure mode.

Corruption Types:
    0: Clean (no corruption)
    1: Atmospheric Haze (ASM model)
    2: Gaussian Blur (PSF turbulence)
    3: JPEG Compression Artifacts (DCT quantization)
    4: Sensor Noise (Gaussian + Shot)
    5: Radiometric Drift (Gain/Offset perturbation)
    6: Band Misalignment (Sub-pixel channel shift)

Severity Levels: 0 (low), 1 (medium), 2 (high)
"""

import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import shift as ndshift


class CorruptionInjector:
    """
    Physically-grounded corruption generator for satellite imagery.
    Each corruption maps to a real preprocessing failure mode.
    """

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(seed=42)

        # Registry of corruption functions keyed by type index
        self._corruption_fns = {
            0: self._no_corruption,
            1: self._atmospheric_haze,
            2: self._gaussian_blur,
            3: self._jpeg_compression,
            4: self._sensor_noise,
            5: self._radiometric_drift,
            6: self._band_misalignment,
        }

    def inject(
        self,
        image: np.ndarray,
        corruption_type: int,
        severity: int,
    ) -> np.ndarray:
        """
        Apply a specific corruption at a given severity.

        Args:
            image: HxWxC uint8 image (0-255)
            corruption_type: int in [0..6]
            severity: int in [0..2] (low/med/high)

        Returns:
            Corrupted image as uint8 HxWxC
        """
        if corruption_type not in self._corruption_fns:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        if severity not in (0, 1, 2):
            raise ValueError(f"Severity must be 0, 1, or 2, got {severity}")

        fn = self._corruption_fns[corruption_type]
        corrupted = fn(image.copy(), severity)
        return np.clip(corrupted, 0, 255).astype(np.uint8)

    def inject_random(
        self, image: np.ndarray, exclude_clean: bool = True
    ) -> tuple[np.ndarray, int, int]:
        """
        Apply a random corruption at random severity.
        Returns: (corrupted_image, corruption_type, severity)
        """
        start = 1 if exclude_clean else 0
        ctype = int(self.rng.integers(start, 7))
        severity = int(self.rng.integers(0, 3))
        return self.inject(image, ctype, severity), ctype, severity

    def generate_all_corruptions(
        self, image: np.ndarray
    ) -> list[tuple[np.ndarray, int, int]]:
        """Generate all corruption×severity combinations for benchmarking."""
        results = []
        for ctype in range(1, 7):
            for sev in range(3):
                corrupted = self.inject(image, ctype, sev)
                results.append((corrupted, ctype, sev))
        return results

    # ------------------------------------------------------------------
    # Corruption implementations (physically motivated)
    # ------------------------------------------------------------------

    @staticmethod
    def _no_corruption(image: np.ndarray, severity: int) -> np.ndarray:
        return image

    def _atmospheric_haze(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        Atmospheric Scattering Model (ASM / Koschmieder):
            I(x) = J(x) * t(x) + A * (1 - t(x))
        where:
            J(x) = scene radiance (clean image)
            t(x) = transmission map ~ exp(-beta * d(x))
            A    = global atmospheric light (airlight)
            beta = scattering coefficient (controls density)

        This models incomplete atmospheric correction in the preprocessing
        pipeline — a common failure when Sen2Cor or similar tools fail to
        fully remove aerosol effects.
        """
        cfg = self.config
        beta = cfg.haze_beta_range[severity]
        airlight = cfg.haze_airlight_range[severity]

        h, w = image.shape[:2]
        img_float = image.astype(np.float64) / 255.0

        # Depth map: gradient from top to bottom simulating distance
        # In satellite imagery, haze varies spatially
        depth = np.linspace(0.2, 1.0, h).reshape(h, 1)
        depth = np.broadcast_to(depth, (h, w))

        # Add Perlin-like spatial variation
        noise = cv2.GaussianBlur(
            self.rng.standard_normal((h, w)).astype(np.float32),
            (0, 0), sigmaX=50
        )
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        depth = depth * 0.7 + noise * 0.3

        transmission = np.exp(-beta * depth)
        transmission = np.expand_dims(transmission, axis=-1)

        hazed = img_float * transmission + airlight * (1.0 - transmission)
        return (hazed * 255.0).astype(np.float64)

    def _gaussian_blur(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        Point Spread Function (PSF) modeling atmospheric turbulence.
        In satellite imaging, the PSF degrades spatial resolution. A Gaussian
        kernel approximates the turbulence-induced blur. This models
        failure in deconvolution/sharpening steps of the preprocessing pipeline.
        """
        sigma = self.config.blur_sigma_levels[severity]
        # Kernel size must be odd and large enough for the sigma
        ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
        return blurred.astype(np.float64)

    def _jpeg_compression(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        DCT-based compression artifacts. Satellite imagery is often compressed
        for transmission/storage. Low quality factors cause blocking artifacts
        in specific frequency bands. This models corruption from lossy
        compression in the data delivery pipeline.
        """
        quality = self.config.jpeg_quality_levels[severity]
        # Use PIL for proper JPEG encoding/decoding
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = np.array(Image.open(buffer))
        return compressed.astype(np.float64)

    def _sensor_noise(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        Combined Gaussian + Shot (Poisson) noise model.
        - Gaussian: thermal/read noise in the sensor electronics
        - Shot noise: photon counting statistics (signal-dependent)

        This models failure in noise reduction preprocessing or
        degraded sensor performance.
        """
        img_float = image.astype(np.float64)
        gauss_std = self.config.gaussian_noise_std[severity]
        shot_scale = self.config.shot_noise_scale[severity]

        # Gaussian component (signal-independent)
        gaussian_noise = self.rng.normal(0, gauss_std, image.shape)

        # Shot noise component (signal-dependent, Poisson-like)
        shot_noise = self.rng.normal(0, 1, image.shape) * np.sqrt(
            np.maximum(img_float * shot_scale, 0)
        )

        noisy = img_float + gaussian_noise + shot_noise
        return noisy

    def _radiometric_drift(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        Gain and offset perturbation per channel. Models miscalibration
        in radiometric correction — when the DN-to-reflectance conversion
        uses stale or incorrect calibration coefficients. This is a common
        silent failure when calibration LUTs are not updated.
        """
        gain_mag = self.config.gain_range[severity]
        offset_mag = self.config.offset_range[severity]
        img_float = image.astype(np.float64)

        num_channels = image.shape[2] if image.ndim == 3 else 1

        for c in range(num_channels):
            # Per-channel gain perturbation: multiplicative
            gain = 1.0 + self.rng.uniform(-gain_mag, gain_mag)
            # Per-channel offset perturbation: additive
            offset = self.rng.uniform(-offset_mag, offset_mag)
            if image.ndim == 3:
                img_float[:, :, c] = img_float[:, :, c] * gain + offset
            else:
                img_float = img_float * gain + offset

        return img_float

    def _band_misalignment(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        Sub-pixel shift between spectral bands. In multi-spectral sensors,
        each band is captured by a different detector array. Geometric
        co-registration aligns them, but residual sub-pixel shifts remain
        if orthorectification is imperfect. This creates color fringing
        and affects spectral indices.
        """
        shift_px = self.config.band_shift_pixels[severity]
        img_float = image.astype(np.float64)

        if image.ndim == 2:
            return img_float

        num_channels = image.shape[2]
        # Apply random sub-pixel shifts to each channel independently
        # Channel 0 stays fixed as reference
        for c in range(1, num_channels):
            dx = self.rng.uniform(-shift_px, shift_px)
            dy = self.rng.uniform(-shift_px, shift_px)
            img_float[:, :, c] = ndshift(
                img_float[:, :, c], shift=(dy, dx), order=3, mode="reflect"
            )

        return img_float


class CorruptionBenchmark:
    """
    Generates the full SatCorrupt-Bench evaluation set from clean images.
    For each clean image, produces 6 corruption types × 3 severities = 18 variants.
    """

    def __init__(self, config):
        self.injector = CorruptionInjector(config)
        self.config = config

    def generate_benchmark_set(
        self,
        clean_images: list[np.ndarray],
        image_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Generate the full benchmark dataset.

        Returns list of dicts:
            {
                "image_id": str,
                "clean": np.ndarray,
                "corrupted": np.ndarray,
                "corruption_type": int,
                "corruption_name": str,
                "severity": int,
            }
        """
        if image_ids is None:
            image_ids = [f"img_{i:05d}" for i in range(len(clean_images))]

        records = []
        for img, img_id in zip(clean_images, image_ids):
            for ctype in range(1, self.config.num_corruption_types):
                for sev in range(self.config.num_severity_levels):
                    corrupted = self.injector.inject(img, ctype, sev)
                    records.append({
                        "image_id": img_id,
                        "clean": img,
                        "corrupted": corrupted,
                        "corruption_type": ctype,
                        "corruption_name": self.config.corruption_names[ctype],
                        "severity": sev,
                    })
        return records
