"""
Hardening Autoencoder with Kick (Shift-N-Overlap) Decoder
==========================================================
A U-Net style encoder-decoder that restores corrupted satellite images
to their clean form. The key novelty is in the decoder:

**Kick Decoder (from Shift-N-Overlap paper):**
Standard transposed convolutions produce checkerboard artifacts because
of uneven overlap in the deconvolution kernel. The Kick approach fixes
this by using MULTIPLE shifted transposed convolutions and AVERAGING
their outputs. This eliminates the periodic overlap pattern.

Architecture:
    Encoder: Conv-BN-ReLU blocks with progressive downsampling
    Bottleneck: Dense representation of corruption-invariant features
    Decoder: Kick transposed conv cascades + skip connections
    Output: Restored image in [0, 1]

Loss Function:
    L = w_l1 * L1(restored, clean)
      + w_perceptual * Perceptual(restored, clean)
      + w_ssim * (1 - SSIM(restored, clean))

Inspired by:
    - Paper 2 (Kick): Shift-N-Overlap cascade for artifact-free upsampling
    - DnCNN: Residual learning for denoising
    - U-Net: Skip connections for preserving spatial detail
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==================================================================
# Kick Module: Shift-N-Overlap Transposed Convolution
# ==================================================================


class KickTransposedConv(nn.Module):
    """
    Shift-N-Overlap Cascade of Transposed Convolutional Layers.

    Instead of a single TransposedConv2d (which creates checkerboard artifacts),
    we use N branches, each with a different spatial shift before deconvolution.
    The outputs are then averaged, canceling the periodic overlap pattern.

    From the paper:
        1. Apply N different sub-pixel shifts to the input
        2. Run each through its own TransposedConv2d
        3. Reverse-shift to realign
        4. Average all branches → artifact-free upsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_branches: int = 3,
        shift_pixels: int = 1,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.shift_pixels = shift_pixels

        # Each branch has its own transposed conv with independent weights
        self.branches = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
            )
            for _ in range(num_branches)
        ])

        # Generate shift offsets for each branch
        # Branch 0: no shift. Others: shift in different directions
        self.shifts = [(0, 0)]
        for i in range(1, num_branches):
            angle = 2 * 3.14159 * i / num_branches
            dx = round(shift_pixels * torch.tensor(angle).cos().item())
            dy = round(shift_pixels * torch.tensor(angle).sin().item())
            self.shifts.append((dy, dx))

    def forward(self, x):
        outputs = []
        for i, branch in enumerate(self.branches):
            dy, dx = self.shifts[i]

            # Shift input
            if dy != 0 or dx != 0:
                shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
            else:
                shifted = x

            # Transposed convolution
            upsampled = branch(shifted)

            # Reverse shift to realign
            if dy != 0 or dx != 0:
                upsampled = torch.roll(upsampled, shifts=(-dy, -dx), dims=(2, 3))

            outputs.append(upsampled)

        # Average all branches → cancels checkerboard artifacts
        return torch.stack(outputs, dim=0).mean(dim=0)


# ==================================================================
# Encoder Block
# ==================================================================


class EncoderBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN-ReLU + MaxPool downsampling."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features  # pooled for next layer, features for skip


# ==================================================================
# Decoder Block (with Kick upsampling)
# ==================================================================


class KickDecoderBlock(nn.Module):
    """Kick upsampling + skip connection + Conv-BN-ReLU."""

    def __init__(self, in_ch, skip_ch, out_ch, num_branches=3, shift_pixels=1):
        super().__init__()
        self.kick_up = KickTransposedConv(
            in_ch, out_ch,
            num_branches=num_branches,
            shift_pixels=shift_pixels,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        up = self.kick_up(x)
        # Handle size mismatch between upsampled and skip
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="bilinear",
                               align_corners=False)
        merged = torch.cat([up, skip], dim=1)
        return self.conv(merged)


# ==================================================================
# Full Hardening Autoencoder
# ==================================================================


class HardeningAutoencoder(nn.Module):
    """
    U-Net architecture with Kick decoder for corruption-aware image restoration.

    Input:  corrupted satellite image (B, C, H, W)
    Output: restored image (B, C, H, W) + residual map
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        in_ch = config.in_channels
        base = config.base_filters
        n_branches = config.kick_num_branches
        shift_px = config.kick_shift_pixels

        # ---- Encoder ----
        self.enc1 = EncoderBlock(in_ch, base)         # → base
        self.enc2 = EncoderBlock(base, base * 2)       # → base*2
        self.enc3 = EncoderBlock(base * 2, base * 4)   # → base*4
        self.enc4 = EncoderBlock(base * 4, base * 8)   # → base*8

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 16, base * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
        )

        # ---- Kick Decoder ----
        self.dec4 = KickDecoderBlock(base * 16, base * 8, base * 8,
                                     n_branches, shift_px)
        self.dec3 = KickDecoderBlock(base * 8, base * 4, base * 4,
                                     n_branches, shift_px)
        self.dec2 = KickDecoderBlock(base * 4, base * 2, base * 2,
                                     n_branches, shift_px)
        self.dec1 = KickDecoderBlock(base * 2, base, base,
                                     n_branches, shift_px)

        # ---- Output head ----
        self.output_conv = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, in_ch, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # ---- Residual head (for visualization) ----
        self.residual_conv = nn.Sequential(
            nn.Conv2d(base, in_ch, 1),
            nn.Tanh(),  # Residual can be positive or negative
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) corrupted image in [0, 1]

        Returns:
            restored: (B, C, H, W) restored image in [0, 1]
            residual: (B, C, H, W) corruption residual map
        """
        # Encode
        e1, skip1 = self.enc1(x)      # (B, base, H/2, W/2)
        e2, skip2 = self.enc2(e1)     # (B, base*2, H/4, W/4)
        e3, skip3 = self.enc3(e2)     # (B, base*4, H/8, W/8)
        e4, skip4 = self.enc4(e3)     # (B, base*8, H/16, W/16)

        # Bottleneck
        bn = self.bottleneck(e4)      # (B, base*16, H/16, W/16)

        # Decode with skip connections
        d4 = self.dec4(bn, skip4)     # (B, base*8, H/8, W/8)
        d3 = self.dec3(d4, skip3)     # (B, base*4, H/4, W/4)
        d2 = self.dec2(d3, skip2)     # (B, base*2, H/2, W/2)
        d1 = self.dec1(d2, skip1)     # (B, base, H, W)

        # Output
        restored = self.output_conv(d1)
        residual = self.residual_conv(d1)

        return restored, residual


# ==================================================================
# Loss Functions
# ==================================================================


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss. Compares feature representations at
    multiple scales rather than pixel-by-pixel. This preserves
    structural content and textures that pixel losses miss.
    """

    def __init__(self):
        super().__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        except Exception as exc:
            print(
                "Warning: failed to load pretrained VGG16 weights for perceptual "
                f"loss ({exc}). Falling back to uninitialized VGG16."
            )
            vgg = models.vgg16(weights=None).features
        # Use features at relu1_2, relu2_2, relu3_3
        self.slice1 = nn.Sequential(*list(vgg[:4]))
        self.slice2 = nn.Sequential(*list(vgg[4:9]))
        self.slice3 = nn.Sequential(*list(vgg[9:16]))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        f1_pred = self.slice1(pred)
        f1_target = self.slice1(target)
        f2_pred = self.slice2(f1_pred)
        f2_target = self.slice2(f1_target)
        f3_pred = self.slice3(f2_pred)
        f3_target = self.slice3(f2_target)

        loss = (
            F.l1_loss(f1_pred, f1_target)
            + F.l1_loss(f2_pred, f2_target)
            + F.l1_loss(f3_pred, f3_target)
        )
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss.
    SSIM captures luminance, contrast, and structure similarity.
    Loss = 1 - SSIM (so minimizing loss maximizes SSIM).
    """

    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    @staticmethod
    def _create_window(window_size, channel):
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window = window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)
        return window.contiguous()

    def forward(self, pred, target):
        window = self.window.to(pred.device, dtype=pred.dtype)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.conv2d(pred, window, padding=self.window_size // 2,
                           groups=self.channel)
        mu_target = F.conv2d(target, window, padding=self.window_size // 2,
                             groups=self.channel)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(
            pred ** 2, window, padding=self.window_size // 2,
            groups=self.channel
        ) - mu_pred_sq
        sigma_target_sq = F.conv2d(
            target ** 2, window, padding=self.window_size // 2,
            groups=self.channel
        ) - mu_target_sq
        sigma_cross = F.conv2d(
            pred * target, window, padding=self.window_size // 2,
            groups=self.channel
        ) - mu_cross

        ssim_map = (
            (2 * mu_cross + C1) * (2 * sigma_cross + C2)
        ) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )

        return 1.0 - ssim_map.mean()


class HardeningLoss(nn.Module):
    """
    Combined loss for the hardening autoencoder.
    L = w_l1 * L1 + w_perceptual * Perceptual + w_ssim * SSIM_Loss
    """

    def __init__(self, config):
        super().__init__()
        self.l1_weight = config.l1_weight
        self.perceptual_weight = config.perceptual_weight
        self.ssim_weight = config.ssim_weight

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss(channel=config.in_channels)

    def forward(self, restored, clean):
        l1 = self.l1_loss(restored, clean)
        perceptual = self.perceptual_loss(restored, clean)
        ssim = self.ssim_loss(restored, clean)

        total = (
            self.l1_weight * l1
            + self.perceptual_weight * perceptual
            + self.ssim_weight * ssim
        )
        return total, l1, perceptual, ssim
