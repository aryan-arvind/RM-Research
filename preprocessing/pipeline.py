"""
Baseline Satellite Image Preprocessing Pipeline
================================================
Implements the standard preprocessing chain that satellite imagery goes through
before reaching any ML model. Each step is a separate function so we can
precisely identify which step the corruption injection module is targeting.

Pipeline Steps:
    1. Radiometric Calibration (DN → TOA Reflectance)
    2. Atmospheric Correction (TOA → BOA approximation)
    3. Geometric Correction (alignment/orthorectification)
    4. Normalization (min-max or z-score)
    5. Tiling & Resizing (crop to fixed patches)
"""

import cv2
import numpy as np


class SatellitePreprocessor:
    """
    Standard preprocessing pipeline for satellite imagery.
    Each method is a separate, testable step.
    """

    def __init__(self, target_size: int = 512, normalize_method: str = "minmax"):
        self.target_size = target_size
        self.normalize_method = normalize_method

    def full_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline end-to-end."""
        img = self.radiometric_calibration(image)
        img = self.atmospheric_correction(img)
        img = self.geometric_correction(img)
        img = self.normalize(img)
        img = self.tile_and_resize(img)
        return img

    def radiometric_calibration(self, image: np.ndarray) -> np.ndarray:
        """
        Convert Digital Numbers (DN) to Top-of-Atmosphere (TOA) reflectance.

        For Sentinel-2, this involves:
            reflectance = (DN + offset) / quantification_value
        where quantification_value = 10000 for Level-1C data.

        For generic uint8 images (our working format), we model this as
        a float conversion with gain/offset normalization.
        """
        img_float = image.astype(np.float32)

        # Model gain and offset calibration per channel
        # This approximates the DN → reflectance conversion
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        return img_float

    def atmospheric_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Approximate atmospheric correction (TOA → BOA).

        Full atmospheric correction requires atmospheric models (e.g., 6S, MODTRAN).
        We approximate with Dark Object Subtraction (DOS) — a common method
        that assumes the darkest pixel in each band represents atmospheric
        path radiance.

            BOA = TOA - path_radiance
            path_radiance ≈ percentile_1(band)
        """
        corrected = image.copy()
        if corrected.ndim == 3:
            for c in range(corrected.shape[2]):
                # DOS: subtract 1st percentile as path radiance estimate
                path_radiance = np.percentile(corrected[:, :, c], 1)
                corrected[:, :, c] = np.maximum(
                    corrected[:, :, c] - path_radiance, 0
                )
        else:
            path_radiance = np.percentile(corrected, 1)
            corrected = np.maximum(corrected - path_radiance, 0)

        return corrected

    def geometric_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Geometric alignment. In production, this uses GCPs and orthorectification.
        Here we ensure consistent orientation and correct any rotation.
        """
        # For pre-processed datasets (DOTA, xView), geometric correction
        # is already applied. We model this as an identity + optional
        # perspective correction.
        return image

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalization to standard range.
        - minmax: scale to [0, 1]
        - zscore: per-channel zero mean, unit variance
        """
        if self.normalize_method == "minmax":
            imin = image.min()
            imax = image.max()
            if imax - imin > 1e-8:
                return (image - imin) / (imax - imin)
            return image
        elif self.normalize_method == "zscore":
            if image.ndim == 3:
                for c in range(image.shape[2]):
                    mean = image[:, :, c].mean()
                    std = image[:, :, c].std() + 1e-8
                    image[:, :, c] = (image[:, :, c] - mean) / std
            else:
                mean, std = image.mean(), image.std() + 1e-8
                image = (image - mean) / std
            return image
        else:
            raise ValueError(f"Unknown normalize method: {self.normalize_method}")

    def tile_and_resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to target size. For tiling large images, use extract_tiles()."""
        h, w = image.shape[:2]
        if h == self.target_size and w == self.target_size:
            return image
        return cv2.resize(
            image, (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

    def extract_tiles(
        self, image: np.ndarray, tile_size: int = 512, overlap: int = 64
    ) -> list[tuple[np.ndarray, int, int]]:
        """
        Extract overlapping tiles from a large satellite image.
        Returns list of (tile, row_start, col_start).
        """
        h, w = image.shape[:2]
        stride = tile_size - overlap
        tiles = []
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y : y + tile_size, x : x + tile_size]
                tiles.append((tile, y, x))
        return tiles

    def to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convert float image back to uint8 for display/saving."""
        if image.max() <= 1.0:
            return (image * 255).clip(0, 255).astype(np.uint8)
        return image.clip(0, 255).astype(np.uint8)
