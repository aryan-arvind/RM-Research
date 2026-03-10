"""
YOLOv8 Downstream Evaluator
============================
Wraps Ultralytics YOLOv8 for measuring downstream detection performance
under different corruption conditions. This is the "stress test" —
we measure mAP degradation as a function of corruption type and severity.

The key output is the "stress curve": mAP vs corruption severity for
each corruption type. This shows the cost of preprocessing failures.

After hardening, we re-evaluate to measure recovery = mAP_hardened - mAP_corrupted.

Inspired by:
    - Paper 3 & 4 (YOLOv8 military satellite detection)
    - Paper 5 (ship detection with DOTA/TGRS-HRRSD)
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLOv8Evaluator:
    """
    YOLOv8 wrapper for downstream evaluation of image quality.
    Measures how much corruption degrades detection performance
    and how much the hardener recovers.
    """

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self, weights_path: Optional[str] = None):
        """Load YOLOv8 model (pretrained or fine-tuned)."""
        path = weights_path or self.config.model_variant
        self.model = YOLO(path)
        return self

    def fine_tune(self, dataset_yaml: str, epochs: Optional[int] = None):
        """
        Fine-tune YOLOv8 on a specific satellite dataset.
        The dataset_yaml should point to the clean training data.
        """
        if self.model is None:
            self.load_model()

        self.model.train(
            data=dataset_yaml,
            epochs=epochs or self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            lr0=self.config.learning_rate,
            device=self.device,
            project="outputs/yolo_training",
            name="finetune",
            exist_ok=True,
        )
        return self

    def evaluate(
        self,
        image_dir: str,
        label_dir: Optional[str] = None,
    ) -> dict:
        """
        Run evaluation on a directory of images.
        Returns mAP@50, mAP@50-95, precision, recall.
        """
        if self.model is None:
            self.load_model()

        results = self.model.val(
            data=self._create_temp_yaml(image_dir, label_dir),
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.device,
        )

        return {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

    def predict_single(self, image: np.ndarray) -> list[dict]:
        """
        Run detection on a single image.
        Returns list of detections: [{bbox, confidence, class_id, class_name}]
        """
        if self.model is None:
            self.load_model()

        results = self.model.predict(
            source=image,
            imgsz=self.config.image_size,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "confidence": float(boxes.conf[i]),
                    "class_id": int(boxes.cls[i]),
                }
                if r.names:
                    det["class_name"] = r.names[int(boxes.cls[i])]
                detections.append(det)
        return detections

    def predict_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        """Run detection on a batch of images."""
        return [self.predict_single(img) for img in images]

    def stress_test(
        self,
        clean_images: list[np.ndarray],
        corruption_injector,
        hardener=None,
        hardener_device: str = "cuda",
    ) -> dict:
        """
        Full stress test: measure mAP for clean, each corruption×severity,
        and optionally hardened versions.

        Returns:
            {
                "clean": {"detections": [...], "count": int},
                "corrupted": {
                    "haze_0": {"detections": [...], "count": int},
                    ...
                },
                "hardened": {  # if hardener provided
                    "haze_0": {"detections": [...], "count": int},
                    ...
                }
            }
        """
        corruption_names = [
            "clean", "haze", "blur", "jpeg", "noise",
            "radiometric_drift", "band_misalignment"
        ]

        results = {"clean": [], "corrupted": {}, "hardened": {}}

        # Evaluate on clean images
        for img in clean_images:
            dets = self.predict_single(img)
            results["clean"].append({"detections": dets, "count": len(dets)})

        # Evaluate on each corruption × severity
        for ctype in range(1, 7):
            for sev in range(3):
                key = f"{corruption_names[ctype]}_{sev}"
                results["corrupted"][key] = []

                for img in clean_images:
                    corrupted = corruption_injector.inject(img, ctype, sev)
                    dets = self.predict_single(corrupted)
                    results["corrupted"][key].append({
                        "detections": dets,
                        "count": len(dets),
                    })

                # Evaluate hardened versions
                if hardener is not None:
                    results["hardened"][key] = []
                    for img in clean_images:
                        corrupted = corruption_injector.inject(img, ctype, sev)
                        hardened = self._harden_image(
                            hardener, corrupted, hardener_device
                        )
                        dets = self.predict_single(hardened)
                        results["hardened"][key].append({
                            "detections": dets,
                            "count": len(dets),
                        })

        return results

    @staticmethod
    def _harden_image(
        hardener, image: np.ndarray, device: str = "cuda"
    ) -> np.ndarray:
        """Pass image through the hardening autoencoder."""
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            restored, _ = hardener(tensor)

        restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (restored_np * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _create_temp_yaml(image_dir: str, label_dir: Optional[str]) -> str:
        """Create a temporary YOLO dataset YAML for evaluation."""
        yaml_content = f"""
path: {Path(image_dir).parent}
val: {Path(image_dir).name}
"""
        if label_dir:
            yaml_content += f"labels: {Path(label_dir).name}\n"

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        tmp.write(yaml_content)
        tmp.close()
        return tmp.name


def prepare_yolo_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    class_names: list[str],
    train_split: float = 0.8,
):
    """
    Prepare a YOLO-format dataset from images + labels directories.
    Creates the required directory structure and data.yaml.
    """
    output = Path(output_dir)
    for split in ["train", "val"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    # Gather and split image files
    img_dir = Path(image_dir)
    images = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif")
    )

    n_train = int(len(images) * train_split)
    train_images = images[:n_train]
    val_images = images[n_train:]

    # Copy files to YOLO structure
    for split, split_images in [("train", train_images), ("val", val_images)]:
        for img_path in split_images:
            shutil.copy2(img_path, output / split / "images" / img_path.name)
            lbl_path = Path(label_dir) / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, output / split / "labels" / lbl_path.name)

    # Write data.yaml
    yaml_path = output / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    return str(yaml_path)
