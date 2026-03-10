# Effect of Data Preprocessing on Machine Learning Models

A research-focused implementation for studying how preprocessing failures affect downstream ML performance on satellite imagery, and how learned hardening can recover that loss.

## Problem Statement
Most ML pipelines assume preprocessing is correct. In real deployments, preprocessing can fail silently (haze residuals, blur, compression artifacts, noise, calibration drift, band misalignment). This project quantifies:
1. How these failures degrade model behavior.
2. How accurately failures can be diagnosed.
3. How much restoration can recover downstream performance.

## Core Contributions in This Codebase
1. Physically motivated corruption injection (`corruption/injector.py`).
2. Multi-task corruption diagnosis network (`models/diagnosis_cnn.py`).
3. Hardening autoencoder with Kick decoder (`models/autoencoder.py`).
4. Downstream stress testing with YOLOv8 (`models/yolo_evaluator.py`).
5. End-to-end evaluation and summary metrics (`training/evaluate.py`).
6. Research dashboard UI for demonstration and analysis (`demo_ui.py`).

## Project Structure
```text
config/          Experiment and model configs
corruption/      Corruption generation logic
data/            Dataset loaders and dataloaders
models/          Diagnosis CNN, Hardener, YOLO wrapper
preprocessing/   Baseline preprocessing pipeline model
training/        Training and full evaluation logic
utils/           Metrics and plotting
main.py          CLI orchestrator
```

## Pipeline Architecture
1. Load clean images.
2. Generate corruption variants (6 types x 3 severity levels).
3. Train diagnosis CNN (corruption type + severity).
4. Train hardening autoencoder (corrupted -> restored).
5. Evaluate clean vs corrupted vs hardened with YOLO.
6. Compute PSNR/SSIM/recovery metrics and export visual reports.

## Corruption Types
1. Atmospheric haze
2. Gaussian blur
3. JPEG compression artifacts
4. Sensor noise
5. Radiometric drift
6. Band misalignment

## Key Metrics
1. `PSNR`: reconstruction fidelity.
2. `SSIM`: structural preservation.
3. `Diagnosis Accuracy`: corruption classification quality.
4. `Detection Drop %`: downstream degradation vs clean baseline.
5. `Recovery Rate %`: hardening gain relative to corruption-induced drop.

### Recovery Formula
```text
Recovery(%) = ((Hardened - Corrupted) / (Clean - Corrupted)) * 100
```

## Quick Start
## 1) Install dependencies
```powershell
pip install -r requirements.txt
```

## 2) Fast end-to-end run (research demo)
```powershell
python main.py --mode full_pipeline --image_dir ./datasets/eurosat/eurosat/2750 --image_size 64 --epochs 1 --batch_size 4 --max_images 10 --num_workers 0
```

## 3) Launch dashboard UI
```powershell
streamlit run demo_ui.py --server.port 8503
```

## Output Artifacts
1. `outputs/evaluation/results.json`
2. `outputs/evaluation/dashboard.png`
3. `outputs/evaluation/stress_curves.png`
4. `outputs/visualizations/*.png`
5. `checkpoints/diagnosis_cnn/best_model.pt`
6. `checkpoints/autoencoder/best_model.pt`

## Datasets
Current implementation has been run using EuroSAT folder structure:
`datasets/eurosat/eurosat/2750`

Note: EuroSAT is primarily classification-oriented. Detection stress results should be interpreted as implementation evidence and robustness trend analysis rather than final detection benchmark claims.

## Detailed Documentation
See:
- `RESEARCH_IMPLEMENTATION_GUIDE.md`

This file includes:
1. Module-by-module code explanation.
2. Mathematical formulations.
3. Core pseudocode.
4. Interpretation guidelines and presentation framing.

## Reproducibility Notes
1. Seed is set in `main.py` via `set_seed`.
2. Windows-safe worker fallback is enabled (`num_workers=0` by default if unset).
3. Use `--max_images` for controlled quick experiments.

## Suggested Citation Context
When reporting results, frame this repository as an implementation study for:
- robustness under preprocessing failures,
- diagnosis-restoration downstream coupling,
- stress-curve-based analysis rather than a single aggregate score.
