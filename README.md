# SimCortexPP (SCPP)

SimCortexPP (SCPP) is a **CLI-first** Python package that provides three practical stages commonly needed in neuroimaging workflows:

1. **Preprocessing (FreeSurfer → MNI152)**  
   Export FreeSurfer volumes and surfaces, register them to MNI152, and write outputs in a **BIDS-derivatives-style** layout.

2. **Segmentation (3D U-Net in MNI space)**  
   Train and apply a 3D U-Net to predict a **9-class** segmentation in **MNI152 space**, with inference and evaluation utilities.

3. **Initial Surfaces (InitSurf)**  
   Generate collision-free initial White Matter and Pial cortical surfaces from the segmentation predictions.

This README focuses on **how to run the pipeline correctly** (inputs, outputs, expected folder/file naming, and commands).

---


## Table of Contents

- [Installation](#installation)
- [Data and Folder Conventions](#data-and-folder-conventions)
- [Session-less BIDS Datasets (Optional)](#session-less-bids-datasets-optional)
- [Stage 1 — Preprocessing: FreeSurfer → MNI152](#stage-1--preprocessing-freesurfer--mni152)
- [Stage 2 — Segmentation: 3D U-Net (MNI space)](#stage-2--segmentation-3d-u-net-mni-space)
  - [Split File Format](#split-file-format)
  - [Train](#train)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Stage 3 — Initial Surfaces (InitSurf)](#stage-3--initial-surfaces-initsurf)
- [Outputs Summary](#outputs-summary)
- [Troubleshooting](#troubleshooting)

---

## Installation

From the repository root:

```bash
pip install -e .
scpp --help
scpp seg --help
scpp initsurf --help
```

### Recommended environment
- Python 3.10+
- PyTorch + MONAI
- `nibabel`, `numpy`, `pandas`, `openpyxl` (for Excel report)
- `trimesh`, `scipy (for surface generation)
---

## Data and Folder Conventions

You will typically work with **two roots**:

1) **Code repository (this repository)**  
Contains code, configs, and scripts (no data).

2) **Dataset root (BIDS + derivatives)**  
Each dataset has its own root directory. Recommended structure:

---

## Session-less BIDS Datasets (Optional)

Some datasets are **session-less** (no `ses-*` folder). If you want a consistent `ses-01` layout, you can create a **non-destructive sessioned view** using symlinks.

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

To remove the sessioned view:
```bash
rm -rf /path/to/rawdata_ses
```
---


## Stage 1 — Preprocessing: FreeSurfer → MNI152

This stage exports key FreeSurfer outputs (volumes + surfaces), registers them to **MNI152**, and writes results to a **BIDS-derivatives-style** folder.

### Inputs
- FreeSurfer derivatives root (contains subject folders)
- MNI template (e.g., `MNI152_T1_1mm.nii.gz`)

### Dependencies (system tools)
- **NiftyReg**: `reg_aladin`, `reg_resample` must be in `PATH`
- FreeSurfer installation is recommended (for conversions and consistency)

### Run (all subjects discovered automatically)
```bash
scpp fs-to-mni \
  --freesurfer-root /path/to/derivatives/freesurfer-7.4.1 \
  --out-deriv-root  /path/to/derivatives/scpp-preproc-0.1 \
  --mni-template    /path/to/MNI152_T1_1mm.nii.gz \
  --decimate 0.3 \
  -v
```

### Run (selected subjects)
```bash
scpp fs-to-mni \
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

### Expected inputs (from Stage 1)
Under `dataset.path` (single dataset) or per-dataset `dataset.roots` (multi-dataset), for each subject:

- MNI T1  
  `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-preproc_T1w.nii.gz`

- MNI aparc+aseg (used to create the 9-class ground truth)  
  `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-aparc+aseg_dseg.nii.gz`

- MNI filled (used for ambiguity fix in label mapping)  
  `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-filled_T1w.nii.gz`

### Output naming (predictions)
Predictions are written in BIDS-derivatives style under the configured output root:

- `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg.nii.gz`


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
  dataset.path=/path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1 \
  dataset.split_file=/path/to/<dataset>_split.csv \
  outputs.root=/path/to/scpp-runs/seg/exp01 \
  trainer.use_ddp=false
```

### Multi-GPU DDP (torchrun)
```bash
scpp seg train --torchrun --nproc-per-node 2 \
  dataset.path=/path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1 \
  dataset.split_file=/path/to/<dataset>_split.csv \
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
  dataset.path=/path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1 \
  dataset.split_file=/path/to/<dataset>_split.csv \
  dataset.split_name=test \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_root=/path/to/datasets/<dataset>/derivatives/scpp-seg-0.1
```

### Multi-dataset inference (two datasets example)
Use per-dataset input roots and per-dataset output roots (keys must match your split CSV `dataset` values):

```bash
scpp seg infer \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.split_name=test \
  dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/scpp-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/scpp-preproc-0.1 \
  model.ckpt_path=/path/to/seg_best_dice.pt \
  outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/scpp-seg-0.1 \
  outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/scpp-seg-0.1
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
  dataset.roots.HCP_YA=/path/to/Datasets/hcpya-u100/derivatives/scpp-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/Datasets/oasis-1/derivatives/scpp-preproc-0.1 \
  outputs.pred_roots.HCP_YA=/path/to/Datasets/hcpya-u100/derivatives/scpp-seg-0.1 \
  outputs.pred_roots.OASIS1=/path/to/Datasets/oasis-1/derivatives/scpp-seg-0.1 \
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

## Stage 3 — Initial surfaces (InitSurf)

Generates initial cortical surfaces from saved segmentation predictions.

What it does per subject:
- reads MNI T1 from `scpp-preproc-*`
- reads 9-class segmentation from `scpp-seg-*`
- builds hemisphere WM masks
- computes SDFs and applies topology correction
- finds **collision-free** WM and Pial surfaces (LH/RH)
- writes WM/Pial meshes and SDF volumes
- builds a ribbon mask and saves ribbon SDF + ribbon probability

### Inputs
- Preproc roots (`scpp-preproc-*`) for MNI T1
- Seg roots (`scpp-seg-*`) for `..._desc-seg9_dseg.nii.gz`
- Split CSV (same as segmentation)

### Outputs
BIDS-derivatives-style outputs under `scpp-initsurf-*`:

```text
scpp-initsurf-0.1/
  dataset_description.json
  sub-XXXX/
    ses-01/
      anat/
        sub-XXXX_ses-01_space-MNI152_desc-lh_white_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-rh_white_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-lh_pial_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-rh_pial_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-ribbon_sdf.nii.gz
        sub-XXXX_ses-01_space-MNI152_desc-ribbon_prob.nii.gz
      surfaces/
        sub-XXXX_ses-01_space-MNI152_hemi-L_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-L_pial.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_pial.surf.ply
```

## Run (multi-dataset example)
You can override Hydra config parameters directly from the CLI. Break down the paths for readability:
```bash
scpp initsurf generate \
  dataset.split_file=/path/to/dataset_split.csv \
  dataset.split_name=all \
  dataset.roots.HCP_YA=/path/to/Datasets/hcpya-u100/derivatives/scpp-preproc-0.1 \
  dataset.roots.OASIS1=/path/to/Datasets/oasis-1/derivatives/scpp-preproc-0.1 \
  dataset.seg_roots.HCP_YA=/path/to/Datasets/hcpya-u100/derivatives/scpp-seg-0.1 \
  dataset.seg_roots.OASIS1=/path/to/Datasets/oasis-1/derivatives/scpp-seg-0.1 \
  outputs.out_roots.HCP_YA=/path/to/Datasets/hcpya-u100/derivatives/scpp-initsurf-0.1 \
  outputs.out_roots.OASIS1=/path/to/Datasets/oasis-1/derivatives/scpp-initsurf-0.1 \
  outputs.log_dir=/path/to/scpp-runs/initsurf/exp01/logs_generate
```

⏱️ Typical runtime: ~31 s/subject (based on a 515-subject run; hardware-dependent).

---

## Recommended workflow order

1) Run preprocessing for each dataset → `scpp-preproc-0.1`  
2) Train segmentation (once) → checkpoint `seg_best_dice.pt`  
3) Run segmentation inference for all subjects → `scpp-seg-0.1`  
4) Run InitSurf generation from saved predictions → `scpp-initsurf-0.1`

---



## License
Add your license and citation details here if needed.

