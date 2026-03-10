# Adaptive Preprocessing Hardening: Implementation Guide for Research Presentation

## 1. Research Problem and Goal

### Research Domain
Effect of Data Preprocessing on Machine Learning Models (satellite imagery context).

### Core Question
How much do realistic preprocessing failures degrade downstream ML performance, and can a learned restoration model recover that loss?

### What this project implements
A full experimental pipeline with 3 linked stages:
1. Corruption simulation of preprocessing failures (physically motivated).
2. Corruption diagnosis (multi-task CNN predicts corruption type and severity).
3. Corruption hardening (autoencoder restores corrupted images), followed by downstream detection evaluation.

### Why this is useful for your paper
Most work reports model accuracy under ideal data. This code operationalizes the missing part: robustness under preprocessing failures and quantitative recovery analysis.

---

## 2. Dataset and Input Data Assumptions

### Dataset used in your current runs
EuroSAT image folder:
`./datasets/eurosat/eurosat/2750`

### Actual layout expected by code
The loader now recursively scans subfolders for image files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`), so class-folder datasets are supported.

### Important note for presentation
EuroSAT is primarily a land-use classification dataset, while downstream module uses YOLO detection counts. That means detection metrics are valid as stress signals for this implementation, but not the strongest benchmark choice for object detection claims.

---

## 3. End-to-End Pipeline Architecture

### Flow overview
1. Load clean images.
2. Inject corruption (6 types x 3 severities).
3. Train diagnosis CNN (type + severity prediction).
4. Train hardening autoencoder (corrupted -> restored).
5. Evaluate with YOLO detector on clean, corrupted, and hardened images.
6. Compute quality and robustness metrics.
7. Save JSON + plots for analysis.

### Main orchestrator
`main.py`

Modes:
1. `train_diagnosis`
2. `train_autoencoder`
3. `train_yolo`
4. `evaluate`
5. `visualize`
6. `full_pipeline`

---

## 4. Module-by-Module Code Understanding

## 4.1 Configuration
File: `config/config.py`

Defines dataclass-based config groups:
1. `PathConfig`: data/output/checkpoint/log folders.
2. `CorruptionConfig`: severity parameters for all corruption generators.
3. `DiagnosisCNNConfig`: model/loss/hyperparameters for diagnosis.
4. `AutoencoderConfig`: architecture and composite loss weights.
5. `YOLOConfig`: detector settings.
6. `ExperimentConfig`: master config object.

Why this design:
Centralized, typed, reproducible experiment control.

## 4.2 Corruption Injector
File: `corruption/injector.py`

Class: `CorruptionInjector`

Implements 7 classes:
1. Clean
2. Haze
3. Blur
4. JPEG artifacts
5. Sensor noise
6. Radiometric drift
7. Band misalignment

Key method:
`inject(image, corruption_type, severity)`

### Corruption logic and formulas

Atmospheric haze (ASM model):
`I(x) = J(x) * t(x) + A * (1 - t(x))`
`t(x) = exp(-beta * d(x))`

Gaussian blur:
Convolution with Gaussian PSF where sigma increases with severity.

JPEG artifacts:
Encode/decode through JPEG with lower quality at higher severity.

Sensor noise:
Gaussian component + signal-dependent shot-like component.

Radiometric drift:
Per-channel gain/offset perturbation.

Band misalignment:
Sub-pixel shift per channel using interpolation (`scipy.ndimage.shift`).

Why this matters:
These are not arbitrary perturbations; they emulate concrete preprocessing failure modes.

## 4.3 Data Layer
File: `data/dataset.py`

Main responsibilities:
1. Generic recursive image loading (`load_images_from_directory`).
2. Dataset wrappers for DOTA/xView/EuroSAT.
3. Training datasets:
- `CorruptionDiagnosisDataset`
- `AutoencoderPairDataset`
4. DataLoader factory (`create_dataloaders`).

Important logic:
1. Diagnosis dataset creates deterministic corruption variants per clean image.
2. Autoencoder dataset creates random corruption pairs online.
3. Train/val split is seeded for reproducibility.

## 4.4 Baseline Preprocessing Module
File: `preprocessing/pipeline.py`

Class: `SatellitePreprocessor`

Implements conceptual baseline preprocessing stages:
1. Radiometric calibration
2. Atmospheric correction (DOS approximation)
3. Geometric correction placeholder
4. Normalization
5. Resize/tiling

Why included:
Provides conceptual bridge between remote sensing preprocessing literature and corruption-injection assumptions.

## 4.5 Diagnosis Network
File: `models/diagnosis_cnn.py`

Class: `DiagnosisCNN`

Architecture:
1. Shared ResNet backbone.
2. Type head (7 classes).
3. Severity head (3 classes).

Loss:
`L_total = w_type * CE(type) + w_severity * CE(severity)`

Extra:
Grad-CAM hooks are registered for explainability/localization.

## 4.6 Hardening Network
File: `models/autoencoder.py`

Class: `HardeningAutoencoder`

Architecture:
1. U-Net-like encoder-decoder.
2. Decoder uses Kick Transposed Convolution (multi-branch shifted upsampling).
3. Skip connections preserve spatial detail.
4. Output head gives restored image.

Loss class:
`HardeningLoss`

Composite objective:
`L = w1 * L1 + w2 * Perceptual + w3 * SSIM_loss`

Where:
1. `L1` preserves pixel fidelity.
2. Perceptual loss uses VGG features.
3. SSIM loss preserves structure.

Robustness patch already added:
If pretrained VGG weights fail to load, fallback to uninitialized VGG so training still runs.

## 4.7 Downstream Detector Wrapper
File: `models/yolo_evaluator.py`

Class: `YOLOv8Evaluator`

Key use here:
Counts detections as downstream performance signal under corruption/hardening.

Functions:
1. Load YOLO weights.
2. Predict on single image.
3. Stress test clean vs corrupted vs hardened.

## 4.8 Training Loops
Files:
1. `training/train_diagnosis.py`
2. `training/train_autoencoder.py`

Both include:
1. Optimizer: AdamW
2. LR scheduler
3. Gradient clipping
4. Validation and best-model checkpointing
5. Early stopping style patience

Recent practical fix:
`max_images` is now wired into both training paths from CLI, so quick experiments are truly quick.

## 4.9 Evaluation Engine
File: `training/evaluate.py`

Function: `run_full_evaluation`

What it does:
1. Loads trained diagnosis and hardening checkpoints.
2. Loads YOLO model.
3. For each corruption x severity:
- Diagnoses type
- Restores image
- Computes PSNR, SSIM
- Runs detector on corrupted and restored image
4. Builds `results` dictionary and summary.

Summary function:
`compute_summary`

Produces:
1. Clean baseline detections.
2. Per-corruption diagnosis accuracy.
3. Per-severity quality and detection drop/recovery.

## 4.10 Metrics and Visualization
Files:
1. `utils/metrics.py`
2. `utils/visualization.py`

Metrics:
1. PSNR
2. SSIM
3. Detection recovery rate
4. Stress curves

Detection recovery formula:
`Recovery(%) = ((hardened - corrupted) / (clean - corrupted)) * 100`

Interpretation:
1. 100% => full recovery to clean baseline.
2. 0% => no improvement.
3. <0% => hardening harmed detection.
4. >100% => hardened outperformed clean baseline.

Visuals saved:
1. Corruption grid
2. Stress curves
3. Dashboard summary
4. Restoration comparison plots

---

## 5. Pseudocode (Core)

### 5.1 Main pipeline pseudocode
```text
parse_args()
config = get_config()
apply_cli_overrides(config)
set_seed(config.seed)

if mode == train_diagnosis:
    train_diagnosis_cnn(config, image_dir, max_images)
elif mode == train_autoencoder:
    train_autoencoder(config, image_dir, max_images)
elif mode == evaluate:
    results = run_full_evaluation(config, image_dir, checkpoints, yolo)
    save_json(results)
    save_plots(results)
elif mode == visualize:
    generate_corruption_grids()
elif mode == full_pipeline:
    train_diagnosis -> train_autoencoder -> evaluate -> visualize
```

### 5.2 Corruption injection pseudocode
```text
function inject(image, corruption_type, severity):
    validate inputs
    fn = corruption_function_registry[corruption_type]
    out = fn(image, severity)
    return clip_to_uint8(out)
```

### 5.3 Evaluation pseudocode
```text
load diagnosis model
load hardening model
load yolo model
load clean images

for each clean image:
    store clean detection count

for corruption in [1..6]:
    for severity in [0..2]:
        for each clean image:
            corrupted = inject(clean, corruption, severity)
            pred_type = diagnosis(corrupted)
            restored = hardener(corrupted)
            psnr, ssim = compare(restored, clean)
            det_corr = yolo(corrupted)
            det_hard = yolo(restored)
        aggregate metrics

compute summary + recovery
return results
```

---

## 6. How to Run for Paper Demo

From project root:
`D:\ARYAN ACADEMIC\SEM 6\RM\adaptive_preprocessing_hardening`

### 6.1 Fast presentation run
```powershell
python main.py --mode full_pipeline --image_dir ./datasets/eurosat/eurosat/2750 --image_size 64 --epochs 1 --batch_size 4 --max_images 10 --num_workers 0
```

### 6.2 Visualization only
```powershell
python main.py --mode visualize --image_dir ./datasets/eurosat/eurosat/2750 --image_size 64
```

### 6.3 Dashboard UI
```powershell
streamlit run demo_ui.py --server.port 8503
```

---

## 7. Where outputs are saved

1. Evaluation JSON: `outputs/evaluation/results.json`
2. Evaluation dashboard plot: `outputs/evaluation/dashboard.png`
3. Stress curves plot: `outputs/evaluation/stress_curves.png`
4. Corruption grids: `outputs/visualizations/*.png`
5. Diagnosis checkpoint: `checkpoints/diagnosis_cnn/best_model.pt`
6. Autoencoder checkpoint: `checkpoints/autoencoder/best_model.pt`

---

## 8. How to interpret current outcomes

From your recent results:
1. Pipeline executes end-to-end successfully.
2. Diagnosis performance is uneven by corruption type.
3. Restoration quality (PSNR/SSIM) is relatively stable but modest.
4. Detection recovery is mixed, including negative recovery for some cases.

Inference for paper discussion:
1. Preprocessing failures materially alter downstream model behavior.
2. A single hardening model does not uniformly recover all corruption families.
3. Recovery depends strongly on corruption type and severity.
4. Robustness should be studied as a curve, not a single scalar metric.

---

## 9. Known limitations and honest reporting points

1. EuroSAT is not the ideal dataset for object detection stress claims.
2. Current quick runs use small data and low epochs (demonstration-focused).
3. Detection count is a coarse proxy compared to full mAP with labeled detection data.
4. Some recovery values >100% or negative are possible by definition and should be explained.

---

## 10. Suggested presentation narrative (implementation proof)

1. Problem: preprocessing failures are under-evaluated in ML robustness studies.
2. Method: simulate realistic failures, diagnose them, harden them, and measure downstream effect.
3. Implementation proof: show code modules + run command + generated artifacts.
4. Result: complete executable framework for robustness research, with quantifiable stress/recovery curves.
5. Future extension: stronger detection datasets, longer training, ablations by corruption family.

---

## 11. Quick viva-style Q&A prep

Q: Why diagnosis before hardening?
A: Diagnosis provides interpretable failure attribution (type/severity), enabling targeted analysis and future conditional hardening.

Q: Why composite loss in hardener?
A: Pixel-level loss alone can over-smooth; perceptual + SSIM preserve structure and texture better.

Q: Why corruption severity levels?
A: To generate stress curves and study robustness under progressive degradation.

Q: Why recovery metric?
A: It normalizes gain relative to observed corruption damage, making comparisons more meaningful than raw counts.

---

This file can serve as your team’s implementation handbook while writing and presenting the paper.
