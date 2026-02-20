# SimCortexPP (SCPP) — Nipoppy-ready MRI Pipeline

SimCortexPP (SCPP) is a **CLI-first** Python package that provides two practical stages commonly needed in neuroimaging workflows and easy to integrate into **Nipoppy**:

1. **Preprocessing (FreeSurfer → MNI152)**  
   Export FreeSurfer volumes/surfaces, register to MNI152, and write outputs in a **BIDS-derivatives-style** layout.

2. **Segmentation (3D U-Net in MNI space)**  
   Train and apply a 3D U-Net to predict a **9-class** segmentation in **MNI152 space**, with inference and evaluation utilities.

This README focuses on **how to run the pipeline correctly** (inputs, outputs, expected folder/file naming, and commands).

---

## Table of Contents

- [Installation](#installation)
- [Data and Folder Conventions](#data-and-folder-conventions)
- [Session-less BIDS Datasets (Optional)](#session-less-bids-datasets-optional)
- [Nipoppy Project Setup (Optional)](#nipoppy-project-setup-optional)
- [Stage 1 — Preprocessing: FreeSurfer → MNI152](#stage-1--preprocessing-freesurfer--mni152)
- [Stage 2 — Segmentation: 3D U-Net (MNI space)](#stage-2--segmentation-3d-u-net-mni-space)
  - [Split File Format](#split-file-format)
  - [Train](#train)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Outputs Summary](#outputs-summary)
- [Troubleshooting](#troubleshooting)

---

## Installation

From the repository root:

```bash
pip install -e .
scpp --help
scpp seg --help
```

### Recommended environment
- Python 3.10+
- PyTorch + MONAI
- `nibabel`, `numpy`, `pandas`, `openpyxl` (for Excel report)

---

## Data and Folder Conventions

You will typically work with **three roots**:

1) **Code repo (this repository)**  
Contains code + configs + scripts (no data).

2) **Study data root (raw + derivatives)**  
Example (conceptually):  
- `rawdata/` or BIDS-like dataset
- `derivatives/freesurfer-7.4.1/` (FreeSurfer outputs)
- additional derivatives

3) **Nipoppy dataset root (workspace for the study)** *(optional but recommended)*  
Contains Nipoppy project structure:
- `global_config.json`, `manifest.tsv`, `pipelines/`, etc.
- symlinks to large derivatives to avoid copying

---

## Session-less BIDS Datasets (Optional)

Some datasets do not have a `ses-*` level (session-less). If you want a consistent `session_id` in Nipoppy manifests, you can create a **non-destructive sessioned view** via symlinks.

Script:
```bash
scripts/create_ses_view.sh
```

Edit variables at the top of the script:
```bash
SRC="/path/to/rawdata"
DST="/path/to/rawdata_ses"
SES="01"
```

Run:
```bash
bash scripts/create_ses_view.sh
```

This creates:
- `sub-XXXX/ses-01/anat/`
- symlinked files
- filenames updated to include `_ses-01_` where needed

To undo:
```bash
rm -rf /path/to/rawdata_ses
```

---

## Nipoppy Project Setup (Optional)

Initialize a Nipoppy dataset root using the sessioned view:

```bash
NIPROOT=/path/to/nipoppy-projects/<study-name>
nipoppy init --dataset "$NIPROOT" --bids-source "/path/to/rawdata_ses"
```

Link existing FreeSurfer derivatives (recommended to avoid copying):
```bash
ln -sfn "/path/to/study/derivatives/freesurfer-7.4.1" "$NIPROOT/derivatives/freesurfer-7.4.1"
```

---

## Stage 1 — Preprocessing: FreeSurfer → MNI152

This stage converts key FreeSurfer outputs into a **BIDS-derivatives-style** directory in MNI space.

### Inputs
- FreeSurfer derivatives root (contains subject folders)
- MNI template (e.g., `MNI152_T1_1mm.nii.gz`)

### Dependencies (system tools)
- **NiftyReg**: `reg_aladin`, `reg_resample` must be in `PATH`
- FreeSurfer installation is recommended (for conversions and consistency)

### Run (all subjects discovered automatically)
```bash
python scripts/preprocess_fs_to_mni_bidsderiv.py \
  --freesurfer-root /path/to/derivatives/freesurfer-7.4.1 \
  --out-deriv-root  /path/to/derivatives/scpp-preproc-0.1 \
  --mni-template    /path/to/MNI152_T1_1mm.nii.gz \
  --decimate 0.3 \
  -v
```

### Run (selected subjects)
```bash
python scripts/preprocess_fs_to_mni_bidsderiv.py \
  --freesurfer-root /path/to/derivatives/freesurfer-7.4.1 \
  --out-deriv-root  /path/to/derivatives/scpp-preproc-0.1 \
  --mni-template    /path/to/MNI152_T1_1mm.nii.gz \
  -p sub-100307 -p sub-101915 \
  -v
```

### Output layout (example)
```text
scpp-preproc-0.1/
  dataset_description.json
  sub-XXXX/
    ses-01/
      anat/
        sub-XXXX_ses-01_space-MNI152_desc-preproc_T1w.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-aparc+aseg_dseg.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-filled_T1w.nii.gz
        sub-XXXX_ses-01_from-T1w_to-MNI152_mode-image_xfm.txt
      surfaces/
        sub-XXXX_ses-01_space-MNI152_hemi-L_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-L_pial.surf.ply
        ...
```

---

## Stage 2 — Segmentation: 3D U-Net (MNI space)

This stage trains and applies a 3D U-Net to predict a **9-class segmentation** in MNI space using the preprocessed outputs.

### Expected inputs (from preprocessing)
Under `dataset.path` (or per-dataset roots), for each subject:

- MNI T1
  - `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-preproc_T1w.nii.gz`
- MNI aparc+aseg (used to create the 9-class GT)
  - `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-aparc+aseg_dseg.nii.gz`
- MNI filled (used for ambiguity fix in label mapping)
  - `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-filled_T1w.nii.gz`

### Output naming (predictions)
Predictions are written in BIDS-derivatives style under the configured prediction root:

- `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-seg9_pred.nii.gz`

---

## Split File Format

A split CSV is required. For **single-dataset** use, the minimal columns are:

- `subject` (e.g., `sub-100307`)
- `split` in `{train, val, test}`

For **multi-dataset** workflows (recommended when evaluating across datasets), include:

- `dataset` (string key that matches config keys, e.g., `HCP_YA`, `OASIS1`)

Example:
```csv
subject,split,dataset
sub-100307,test,HCP_YA
sub-101915,test,HCP_YA
sub-0001,test,OASIS1
```

---

## Train

### Single-GPU
```bash
scpp seg train \
  dataset.path=/path/to/scpp-preproc-0.1 \
  dataset.split_file=/path/to/dataset_split.csv \
  outputs.root=/path/to/scpp-runs/seg/exp01 \
  trainer.use_ddp=false
```

### Multi-GPU DDP (torchrun)
```bash
scpp seg train --torchrun --nproc-per-node 2 \
  dataset.path=/path/to/scpp-preproc-0.1 \
  dataset.split_file=/path/to/dataset_split.csv \
  outputs.root=/path/to/scpp-runs/seg/exp01 \
  trainer.use_ddp=true
```

Training outputs typically include:
- checkpoints (best Dice, last, etc.)
- logs (TensorBoard / text logs depending on your setup)

---

## Inference

### Single dataset
```bash
scpp seg infer \
  dataset.path=/path/to/scpp-preproc-0.1 \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.pred_root=/path/to/scpp-seg-0.1
```

### Multi-dataset inference (two datasets example)
Use per-dataset input roots and per-dataset output roots (keys must match your split CSV `dataset` values):

```bash
scpp seg infer \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/nipoppy-hcpya-u100-scpp/derivatives/scpp-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/nipoppy-oasis-1-scpp/derivatives/scpp-preproc-0.1 \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_roots.HCP_YA=/path/to/nipoppy-hcpya-u100-scpp/derivatives/scpp-seg-0.1 \
  outputs.out_roots.OASIS1=/path/to/nipoppy-oasis-1-scpp/derivatives/scpp-seg-0.1
```

---

## Evaluation

Evaluation computes (per-subject):
- **Dice** (mean over classes, optionally excluding background)
- **Accuracy**
- **NSD / Surface Dice** (MONAI `compute_surface_dice` with configurable tolerance)

### Multi-dataset evaluation (recommended)
```bash
scpp seg eval \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/nipoppy-hcpya-u100-scpp/derivatives/scpp-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/nipoppy-oasis-1-scpp/derivatives/scpp-preproc-0.1 \
  outputs.pred_roots.HCP_YA=/path/to/nipoppy-hcpya-u100-scpp/derivatives/scpp-seg-0.1 \
  outputs.pred_roots.OASIS1=/path/to/nipoppy-oasis-1-scpp/derivatives/scpp-seg-0.1 \
  outputs.eval_csv=/path/to/scpp-runs/seg/exp01/evals/seg_eval_test.csv \
  outputs.eval_xlsx=/path/to/scpp-runs/seg/exp01/evals/seg_eval_test.xlsx
```

### Reports
- CSV: `outputs.eval_csv` (per-subject table)
- Excel (optional): `outputs.eval_xlsx` with sheets:
  - `per_subject`
  - `summary_by_dataset`
  - `summary_overall`

---

## Outputs Summary

### Preprocessing output (`scpp-preproc-0.1`)
- MNI T1, MNI aparc+aseg, MNI filled
- transforms (`*_xfm.txt`)
- MNI-space surfaces (PLY)

### Segmentation output (`scpp-seg-0.1`)
- predictions per subject:
  - `sub-XXXX_ses-01_space-MNI152_desc-seg9_pred.nii.gz`
- evaluation reports:
  - `seg_eval_test.csv`
  - `seg_eval_test.xlsx` (optional)

---

## Troubleshooting

### 1) `FileNotFoundError` for split CSV
Ensure you passed the correct absolute path:
```bash
scpp seg infer dataset.split_file=/absolute/path/to/dataset_split.csv
```

### 2) Multi-dataset keys mismatch
If your split file has `dataset=HCP_YA`, your config/overrides must use the same key:
- `dataset.roots.HCP_YA=...`
- `outputs.pred_roots.HCP_YA=...`

### 3) Session label mismatch
If your data is under `ses-01`, your config should use:
- `dataset.session_label="01"` (or `"ses-01"` depending on implementation)

### 4) Missing expected preproc files
Verify the preprocessing outputs exist for a subject:
- `*_desc-preproc_T1w.nii.gz`
- `*_desc-aparc+aseg_dseg.nii.gz`
- `*_desc-filled_T1w.nii.gz`

---

## License
Add your license and citation details here if needed.

