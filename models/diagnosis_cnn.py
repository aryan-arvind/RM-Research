"""
Diagnosis CNN — Multi-Task Corruption Classifier
=================================================
A ResNet-based multi-task network that simultaneously predicts:
  1. Corruption TYPE (7 classes: clean + 6 corruption types)
  2. Corruption SEVERITY (3 levels: low / medium / high)

Architecture:
  - Backbone: ResNet-18 (pretrained on ImageNet, fine-tuned)
  - Shared feature extractor → two classification heads
  - Grad-CAM support for spatial localization of corruption signatures

This is the "fault localization" component — it doesn't just detect
that something is wrong, it identifies WHICH preprocessing step failed
and HOW BADLY.

Inspired by:
  - Paper 1 (damage detection pipeline): multi-class defect classification
  - BIQA (Blind Image Quality Assessment) literature
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DiagnosisCNN(nn.Module):
    """
    Multi-task CNN for corruption diagnosis.

    Outputs:
        type_logits:     (B, num_corruption_classes)  — which corruption
        severity_logits: (B, num_severity_classes)     — how severe
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- Backbone (shared feature extractor) ----
        if config.backbone == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if config.pretrained else None
            )
        elif config.backbone == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if config.pretrained else None
            )
        elif config.backbone == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if config.pretrained else None
            )
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        # Adapt first conv if input channels != 3
        if config.in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                config.in_channels, 64, kernel_size=7, stride=2,
                padding=3, bias=False,
            )
            # Initialize new conv by averaging pretrained weights across channels
            if config.pretrained:
                with torch.no_grad():
                    backbone.conv1.weight[:] = old_conv.weight.mean(
                        dim=1, keepdim=True
                    ).repeat(1, config.in_channels, 1, 1)

        # Remove the original FC layer
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = feature_dim

        # Hook for Grad-CAM (we need the last conv layer's output)
        self.gradients = None
        self.activations = None

        # ---- Task-specific heads ----

        # Corruption type classification head
        self.type_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(256, config.num_corruption_classes),
        )

        # Severity classification head
        self.severity_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.num_severity_classes),
        )

        # Register hook on the last convolutional layer for Grad-CAM
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks on the last conv layer for Grad-CAM."""
        # For ResNet, layer4 is the last convolutional block
        target_layer = self.backbone.layer4[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image tensor

        Returns:
            type_logits: (B, num_corruption_classes)
            severity_logits: (B, num_severity_classes)
        """
        features = self.backbone(x)  # (B, feature_dim)

        type_logits = self.type_head(features)
        severity_logits = self.severity_head(features)

        return type_logits, severity_logits

    def get_grad_cam(self, x, target_class: int = None) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for corruption localization.

        Args:
            x: (1, C, H, W) single image tensor
            target_class: which corruption class to visualize (None = predicted)

        Returns:
            heatmap: (H, W) normalized attention map
        """
        self.eval()
        x.requires_grad_(True)

        type_logits, _ = self.forward(x)

        if target_class is None:
            target_class = type_logits.argmax(dim=1).item()

        # Backprop from target class score
        self.zero_grad()
        type_logits[0, target_class].backward(retain_graph=True)

        # Grad-CAM: weight activations by gradient
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # GAP of gradients
        cam = (weights * activations).sum(dim=1, keepdim=True)  # weighted sum
        cam = torch.relu(cam)  # ReLU to keep positive contributions

        # Upsample to input size
        cam = nn.functional.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.detach()


class DiagnosisLoss(nn.Module):
    """
    Combined loss for multi-task corruption diagnosis.
    L_total = w_type * CE(type) + w_severity * CE(severity)

    Uses label smoothing for better calibration and class weights
    to handle the clean-vs-corrupted imbalance.
    """

    def __init__(self, config, type_weights=None, severity_weights=None):
        super().__init__()
        self.type_weight = config.type_loss_weight
        self.severity_weight = config.severity_loss_weight

        self.type_criterion = nn.CrossEntropyLoss(
            weight=type_weights, label_smoothing=0.1
        )
        self.severity_criterion = nn.CrossEntropyLoss(
            weight=severity_weights, label_smoothing=0.05
        )

    def forward(self, type_logits, severity_logits, type_targets, severity_targets):
        type_loss = self.type_criterion(type_logits, type_targets)
        severity_loss = self.severity_criterion(severity_logits, severity_targets)

        total = self.type_weight * type_loss + self.severity_weight * severity_loss
        return total, type_loss, severity_loss
