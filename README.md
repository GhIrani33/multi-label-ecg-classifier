# PTB-XL / PTB-XL+ ECG CDS (Reproducible)

A multi-label ECG pipeline for **71 PTB-XL statements** + **5 super-classes**, using **raw 12-lead waveforms** and **PTB-XL+ engineered features + median beats**.  
Two reproducible profiles:
- **Baseline (General)** — calibrated, stable across all 71 labels.
- **Hybrid (Minority-Boosted)** — per-label switch to improve recall for 15 rare labels (no retraining at inference).

> This repo ships code only. You download PTB-XL and PTB-XL+ from PhysioNet, then run the pipeline end-to-end on Windows/PowerShell.

---

## Data (download first)

- PTB-XL 1.0.3: https://physionet.org/content/ptb-xl/1.0.3/  
- PTB-XL+ 1.0.1: https://physionet.org/content/ptb-xl-plus/1.0.1/

**Expected directory (absolute Windows paths)**

D:\Project\ECG\old
├─ ptbxl\ # unpack PTB-XL here
├─ ptbxl+\ # unpack PTB-XL+ here
├─ processed\ # all processed artifacts will be created here
└─ scr
├─ preprocessing.py
├─ ptbxl_plus_fusion.py
└─ train
├─ train_fusion_v2.py # Baseline training
└─ train_fusion_v2_hybrid.py # Sampler training

r
Copy code

---

## Environment

- Windows 10/11, Python ≥3.9, PyTorch (CUDA optional).
- Create a venv and install requirements (your pinned file, e.g. `requirements.txt` or `ENVIRONMENT.txt`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r D:\Project\ECG\old\processed\ENVIRONMENT.txt
Quick Start (end-to-end)
All commands assume PowerShell and BASE=D:\Project\ECG\old.

1) Build processed datasets
powershell
Copy code
python D:\Project\ECG\old\scr\preprocessing.py `
  --base D:\Project\ECG\old `
  --ptbxl_root D:\Project\ECG\old\ptbxl

python D:\Project\ECG\old\scr\ptbxl_plus_fusion.py `
  --base D:\Project\ECG\old `
  --ptbxl_plus_root D:\Project\ECG\old\ptbxl+
Creates (under processed\): ptbxl_core.parquet, ptbxl_plus_features_imputed_scaled.parquet, ptbxl_train_ready_plus.parquet, label matrices, splits (train/val/test_ids.csv), and reports.

2) Train models (optional if you already have runs)
Baseline (General)

powershell
Copy code
python D:\Project\ECG\old\scr\train\train_fusion_v2.py `
  --base D:\Project\ECG\old --use_waveform 1 --use_median 1 `
  --epochs 60 --bs 12 --sched cosine
Hybrid (Sampler)

powershell
Copy code
python D:\Project\ECG\old\scr\train\train_fusion_v2_hybrid.py `
  --base D:\Project\ECG\old --use_waveform 1 --use_median 1 `
  --epochs 60 --bs 12 --sched cosine
Outputs: run folders under processed\runs\ (each with best_model.pt, TRAIN_PLAN.txt, train_log.txt, FINAL_REPORT.txt).

3) Evaluate & calibrate
Baseline

powershell
Copy code
python D:\Project\ECG\old\processed\runs\eval_calibrate_v2.py `
  --base D:\Project\ECG\old `
  --run_dir D:\Project\ECG\old\processed\runs\fusion_v2
Produces: class_metrics_calibrated.csv, calib_tau71.json, thresholds_val.json, CDS_REPORT.txt.

Hybrid (no retraining)

powershell
Copy code
# Build per-class VAL thresholds for hybrid (once)
python D:\Project\ECG\old\processed\runs\hybrid_thresholds_val.py `
  --base D:\Project\ECG\old `
  --run_base D:\Project\ECG\old\processed\runs\fusion_v2 `
  --run_samp D:\Project\ECG\old\processed\runs\fusion_v2_hybrid

# Evaluate hybrid with per-label switching
python D:\Project\ECG\old\processed\runs\eval_calibrate_v2_hybrid.py `
  --base D:\Project\ECG\old `
  --run_base D:\Project\ECG\old\processed\runs\fusion_v2 `
  --run_samp D:\Project\ECG\old\processed\runs\fusion_v2_hybrid
Produces: class_metrics_hybrid.csv, CDS_REPORT_HYBRID.txt, thresholds_val_hybrid.json.

4) Operating points on TEST (for CDS)
powershell
Copy code
# Hybrid policies: F1 / TPR>=90% / FPR<=5%
python D:\Project\ECG\old\processed\runs\ops_metrics_hybrid.py `
  --base D:\Project\ECG\old `
  --run_dir D:\Project\ECG\old\processed\runs\fusion_v2_hybrid `
  --policy f1

python D:\Project\ECG\old\processed\runs\ops_metrics_hybrid.py `
  --base D:\Project\ECG\old `
  --run_dir D:\Project\ECG\old\processed\runs\fusion_v2_hybrid `
  --policy tpr90

python D:\Project\ECG\old\processed\runs\ops_metrics_hybrid.py `
  --base D:\Project\ECG\old `
  --run_dir D:\Project\ECG\old\processed\runs\fusion_v2_hybrid `
  --policy fpr05
Outputs: operating_metrics_test_hybrid_{f1,tpr90,fpr05}.csv + OPS_F1_SUMMARY.csv.

5) Compare Baseline vs Hybrid (ΔAUC per class)
powershell
Copy code
python D:\Project\ECG\old\processed\runs\compare_runs_v2.py `
  --base D:\Project\ECG\old `
  --run_base D:\Project\ECG\old\processed\runs\fusion_v2 `
  --run_samp D:\Project\ECG\old\processed\runs\fusion_v2_hybrid
What we did (high-level)
Stage-1: PTB-XL core build (labels 71 + 5, splits, winsorized age, no raw edits).

Stage-2: PTB-XL+ fusion (feature QC, impute/scale, median beats, alignment).

Stage-3: Training (Baseline & Sampler), Calibration (τ), VAL thresholds (quantile grid), Hybrid per-label switch for 15 rare labels, and OPS on TEST for clinical policies.

See TECHNICAL.md for design, losses, calibration, file semantics, and reproducibility notes.

yaml
Copy code

---

# `TECHNICAL.md` (focused, technical)

```markdown
# TECHNICAL.md — PTB-XL / PTB-XL+ ECG CDS

## 1) Scope

- **Labels:** 71 PTB-XL statements (+ 5 super-classes as guidance).
- **Inputs:** raw 12-lead 10-sec waveforms (500 Hz), PTB-XL+ engineered features (12SL/ECGdeli/UniG), and median-beat morphology.
- **Outputs:** calibrated probabilities for 71 labels; per-class thresholds from VAL to realize **F1**, **TPR≥90%**, or **FPR≤5%** on TEST.

Two profiles:
- **Baseline** — generalist; no class-aware sampler.
- **Hybrid** — at inference, switch logits to **Sampler model** for a curated `rare_plus` list (15 labels) to lift minority recall; other labels use Baseline logits.

---

## 2) Data flow (stages)

### Stage-1 — Core (PTB-XL)
Script: `scr\preprocessing.py`  
- Builds `processed\ptbxl_core.parquet` and label matrices:
  - `labels_71_ge50.csv` (71 columns + `ecg_id`)
  - `labels_5_super.csv` (NORM, MI, STTC, CD, HYP)
- Official folds respected; splits written to `train_ids.csv`, `val_ids.csv`, `test_ids.csv`.
- Age winsorization at 90 (per PhysioNet practice).
- Report: `PREP_REPORT.txt`.

### Stage-2 — PTB-XL+ fusion
Script: `scr\ptbxl_plus_fusion.py`  
- Validates & merges engineered features and **median beats**.
- Imputation/scaling → `ptbxl_plus_features_imputed_scaled.parquet` (+ metadata JSON).
- Final training table: `ptbxl_train_ready_plus.parquet`.
- Report: `PTBXL_PLUS_FUSION_REPORT.txt`.

### Stage-3 — Training & Evaluation
- **Models:** `train_fusion_v2.py` (Baseline), `train_fusion_v2_hybrid.py` (Sampler variant).
- **FusionNetV2:** MLP(features) ⊕ CNN(waveform) ⊕ CNN(median) → heads (71/5).
- **Loss/regularization:** BCE + label smoothing (71), logit-L2, confidence penalty; class weights if needed.
- **Calibration:** temperature scaling τ (VAL ECE minimization) → `calib_tau71.json`.
- **Thresholds:** per-class quantile search on VAL → `thresholds_val.json` (Baseline); Hybrid thresholds via `hybrid_thresholds_val.py` → `thresholds_val_hybrid.json`.
- **Hybrid selection:** `rare_plus.json` decides which labels use Sampler logits; rest use Baseline.

---

## 3) Reproducible artifacts (per run)

Each run folder contains:

- `best_model.pt` — state dict plus metadata: `labels71`, `feat_cols`, `use_wave`, `use_median`.
- `TRAIN_PLAN.txt`, `train_log.txt`, `FINAL_REPORT.txt`.
- After eval:
  - `class_metrics_raw.csv` / `class_metrics_calibrated.csv` (per-class ROC-AUC / PR-AUC; raw vs calibrated).
  - `calib_tau71.json` (scalar τ).
  - `thresholds_val.json` (Baseline) or `thresholds_val_hybrid.json` (Hybrid).
  - `CDS_REPORT.txt` and/or `CDS_REPORT_HYBRID.txt`.

**OPS outputs (Hybrid)**  
`operating_metrics_test_hybrid_{f1,tpr90,fpr05}.csv` with columns:  
`label, th, tp, fp, tn, fn, precision, recall, specificity, f1` (+ `prev` if available).

---

## 4) Minimal commands (PowerShell)

**Data preparation**
```powershell
python D:\Project\ECG\old\scr\preprocessing.py --base D:\Project\ECG\old --ptbxl_root D:\Project\ECG\old\ptbxl
python D:\Project\ECG\old\scr\ptbxl_plus_fusion.py --base D:\Project\ECG\old --ptbxl_plus_root D:\Project\ECG\old\ptbxl+
Train (optional)

powershell
Copy code
# Baseline
python D:\Project\ECG\old\scr\train\train_fusion_v2.py --base D:\Project\ECG\old --use_waveform 1 --use_median 1 --epochs 60 --bs 12 --sched cosine
# Sampler
python D:\Project\ECG\old\scr\train\train_fusion_v2_hybrid.py --base D:\Project\ECG\old --use_waveform 1 --use_median 1 --epochs 60 --bs 12 --sched cosine
Evaluate & calibrate

powershell
Copy code
# Baseline
python D:\Project\ECG\old\processed\runs\eval_calibrate_v2.py --base D:\Project\ECG\old --run_dir D:\Project\ECG\old\processed\runs\fusion_v2

# Hybrid thresholds on VAL, then eval Hybrid on TEST
python D:\Project\ECG\old\processed\runs\hybrid_thresholds_val.py --base D:\Project\ECG\old --run_base D:\Project\ECG\old\processed\runs\fusion_v2 --run_samp D:\Project\ECG\old\processed\runs\fusion_v2_hybrid
python D:\Project\ECG\old\processed\runs\eval_calibrate_v2_hybrid.py --base D:\Project\ECG\old --run_base D:\Project\ECG\old\processed\runs\fusion_v2 --run_samp D:\Project\ECG\old\processed\runs\fusion_v2_hybrid
Operating points (Hybrid policies)

powershell
Copy code
python D:\Project\ECG\old\processed\runs\ops_metrics_hybrid.py --base D:\Project\ECG\old --run_dir D:\Project\ECG\old\processed\runs\fusion_v2_hybrid --policy f1
python D:\Project\ECG\old\processed\runs\ops_metrics_hybrid.py --base D:\Project\ECG\old --run_dir D:\Project\ECG\old\processed\runs\fusion_v2_hybrid --policy tpr90
python D:\Project\ECG\old\processed\runs\ops_metrics_hybrid.py --base D:\Project\ECG\old --run_dir D:\Project\ECG\old\processed\runs\fusion_v2_hybrid --policy fpr05
Compare runs

powershell
Copy code
python D:\Project\ECG\old\processed\runs\compare_runs_v2.py --base D:\Project\ECG\old --run_base D:\Project\ECG\old\processed\runs\fusion_v2 --run_samp D:\Project\ECG\old\processed\runs\fusion_v2_hybrid
5) Notes & troubleshooting
Paths: The code expects absolute paths as shown; if you change the root, pass your --base accordingly everywhere.

Threshold files:

Baseline OPS expects thresholds_val.json in its run directory.
Hybrid OPS expects thresholds_val_hybrid.json. Rebuild with hybrid_thresholds_val.py if missing.
Calibration: calib_tau71.json holds the temperature τ; metrics tables include both raw and calibrated AUCs.
GPU memory: Reduce --bs if you hit OOM; you can temporarily run with --use_waveform 0 to validate the tabular+median stack.

6) What changed and why (succinct)
We kept Stage-1/2 fully deterministic (no mutation of raw PTB-XL/XL+).
We introduced an inference-time hybrid (per-label switch to the Sampler model for 15 rare classes) to raise recall without retraining.
Calibration and VAL-derived thresholds ensure decision-level control under F1 / TPR≥90% / FPR≤5% policies on TEST, enabling CDS-grade risk management.
