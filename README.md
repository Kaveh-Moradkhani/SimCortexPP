# SimCortexPP (SCPP)

SimCortexPP (SCPP) is a modular pipeline for cortical surface reconstruction in MNI space. It provides four stages:

1. **Preprocessing (FreeSurfer → MNI152)**  
   Export key FreeSurfer volumes/surfaces, register them to MNI152, and write outputs in a **BIDS-derivatives-style** layout.

2. **Segmentation (3D U-Net in MNI space)**  
   Train and apply a 3D U-Net to predict a **9-class segmentation** in **MNI152 space**, with inference and evaluation utilities.

3. **Initial Surfaces (InitSurf)**  
   Generate initial White Matter and Pial surfaces from segmentation predictions (plus ribbon SDF/probability outputs).

4. **Deformation (Deform)**  
   Deform the initial surfaces toward MNI-aligned FreeSurfer surfaces using geometric losses and optional collision metrics, and write **deformed surfaces** as BIDS-derivatives.

This README focuses on **how to run the pipeline correctly** (inputs, outputs, folder/file naming, and commands).

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Data and Folder Conventions](#data-and-folder-conventions)
- [Split File Format](#split-file-format)
- [Stage 1 — Preprocessing: FreeSurfer → MNI152](#stage-1--preprocessing-freesurfer--mni152)
- [Stage 2 — Segmentation: 3D U-Net (MNI space)](#stage-2--segmentation-3d-u-net-mni-space)
- [Stage 3 — Initial Surfaces (InitSurf)](#stage-3--initial-surfaces-initsurf)
- [Stage 4 — Deformation (Deform)](#stage-4--deformation-deform)
- [License](#license)

---

## Installation

From the repository root:

```bash
pip install -e .
scpp --help
scpp seg --help
scpp initsurf --help
scpp deform --help
```

### Recommended environment
- Python 3.10+
- PyTorch + MONAI
- `nibabel`, `numpy`, `pandas`, `openpyxl`
- `trimesh`, `scipy`, `tqdm`
- External tools for Stage 1: **NiftyReg** (`reg_aladin`, `reg_resample`)
- Optional (Deform metrics):
  - `python-fcl` for collision metrics
  - `pymeshlab` for SIF (self-intersection fraction) in Deform evaluation

---

## Configuration

All stages use Hydra YAML configs shipped with the package (see `src/simcortexpp/configs/<stage>/*.yaml`).

You have **two** ways to configure a run:

1) **Edit the stage YAML** (recommended for longer runs / stable experiments), then run commands with no extra arguments, e.g.:
   - `scpp deform eval`

2) **Use Hydra overrides on the CLI** (recommended for quick tests), e.g.:
   - `scpp deform eval dataset.split_name=test outputs.out_dir=/tmp/deform_eval`

---

## Data and Folder Conventions

You will typically work with **two roots**:

1) **Code repository (this repository)**  
Contains code, configs, and scripts (no data).

2) **Dataset root (BIDS + derivatives)**  
Each dataset has its own root directory. Recommended structure:

```text
datasets/<dataset-name>/
  bids/                 # raw BIDS dataset
  derivatives/          # processed outputs (BIDS derivatives)
    freesurfer-7.4.1/
    scpp-preproc-0.1/
    scpp-seg-0.1/
    scpp-initsurf-0.1/
    scpp-deform-0.1/
  splits/
    <dataset>_split.csv
```

SCPP reads inputs from `derivatives/` and writes outputs back to `derivatives/` using BIDS-derivatives-style naming.

> Important: keep the dataset root naming consistent (e.g., use `datasets/...` everywhere).

---

## Split File Format

A split CSV is required for Segmentation, InitSurf, and Deform.

### Single-dataset split
Minimal columns:
- `subject` (e.g., `sub-0001`)
- `split` in `{train, val, test}`

### Multi-dataset split
Include an additional column:
- `dataset` (string key that matches config keys, e.g., `HCP_YA`, `OASIS1`)

Example:
```csv
subject,split,dataset
sub-100307,test,HCP_YA
sub-101915,test,HCP_YA
sub-0001,test,OASIS1
```

---

## Stage 1 — Preprocessing: FreeSurfer → MNI152

This stage exports key FreeSurfer outputs (volumes + surfaces), registers them to **MNI152**, and writes results to a **BIDS-derivatives-style** folder.

### Inputs
- FreeSurfer derivatives root (contains `sub-*` folders)
- MNI template (e.g., `src/MNI152_T1_1mm.nii.gz`)

### Dependencies (system tools)
- **NiftyReg**: `reg_aladin`, `reg_resample` must be in `PATH`
- **FreeSurfer** tools are recommended (e.g., `mri_convert`, `mris_convert`) for consistent conversions

### Run (all subjects discovered automatically)
```bash
scpp fs-to-mni   --freesurfer-root /path/to/datasets/<dataset>/derivatives/freesurfer-7.4.1   --out-deriv-root  /path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1   --mni-template    /path/to/SimCortexPP/src/MNI152_T1_1mm.nii.gz   --decimate 0.3   -v
```

### Run (selected subjects)
```bash
scpp fs-to-mni   --freesurfer-root /path/to/datasets/<dataset>/derivatives/freesurfer-7.4.1   --out-deriv-root  /path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1   --mni-template    /path/to/SimCortexPP/src/MNI152_T1_1mm.nii.gz   -p sub-0001 -p sub-0019   -v
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
        sub-XXXX_ses-01_space-MNI152_hemi-R_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_hemi-R_pial.surf.ply
```

---

## Stage 2 — Segmentation: 3D U-Net (MNI space)

This stage trains and applies a 3D U-Net to predict a **9-class segmentation** in **MNI152 space** using Stage 1 outputs.

### Expected inputs (from Stage 1)
Under `scpp-preproc-*`, for each subject:
- `..._desc-preproc_T1w.nii.gz`
- `..._desc-aparc+aseg_dseg.nii.gz`
- `..._desc-filled_T1w.nii.gz`

### Output naming (predictions)
Predictions are written under `scpp-seg-*`:
- `sub-XXXX/ses-01/anat/sub-XXXX_ses-01_space-MNI152_desc-seg9_dseg.nii.gz`

### Train (single GPU)
```bash
scpp seg train   dataset.path=/path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1   dataset.split_file=/path/to/datasets/<dataset>/splits/<dataset>_split.csv   outputs.root=/path/to/scpp-runs/seg/exp01   trainer.use_ddp=false
```

### Train (multi-GPU, torchrun)
```bash
scpp seg train --torchrun --nproc-per-node 2   dataset.path=/path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1   dataset.split_file=/path/to/datasets/<dataset>/splits/<dataset>_split.csv   outputs.root=/path/to/scpp-runs/seg/exp01   trainer.use_ddp=true
```

### Inference
```bash
scpp seg infer   dataset.path=/path/to/datasets/<dataset>/derivatives/scpp-preproc-0.1   dataset.split_file=/path/to/datasets/<dataset>/splits/<dataset>_split.csv   dataset.split_name=test   model.ckpt_path=/path/to/seg_best_dice.pt   outputs.out_root=/path/to/datasets/<dataset>/derivatives/scpp-seg-0.1
```

### Evaluation (multi-dataset example)
```bash
scpp seg eval   dataset.split_file=/path/to/datasets/splits/dataset_split.csv   dataset.split_name=test   dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/scpp-preproc-0.1   dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/scpp-preproc-0.1   outputs.pred_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/scpp-seg-0.1   outputs.pred_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/scpp-seg-0.1   outputs.eval_csv=/path/to/scpp-runs/seg/exp01/evals/seg_eval_test.csv   outputs.eval_xlsx=/path/to/scpp-runs/seg/exp01/evals/seg_eval_test.xlsx
```

---

## Stage 3 — Initial Surfaces (InitSurf)

This stage generates initial cortical surfaces from saved segmentation predictions (not end-to-end).

### Inputs
- Preproc roots (`scpp-preproc-*`) for MNI T1
- Seg roots (`scpp-seg-*`) for `..._desc-seg9_dseg.nii.gz`
- Split CSV (same format as Stage 2)

### Outputs
BIDS-derivatives-style outputs under `scpp-initsurf-*` (meshes + SDF volumes + ribbon prob):
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

### Run (multi-dataset example)
```bash
scpp initsurf generate   dataset.split_file=/path/to/datasets/splits/dataset_split.csv   dataset.split_name=all   dataset.roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/scpp-preproc-0.1   dataset.roots.OASIS1=/path/to/datasets/oasis-1/derivatives/scpp-preproc-0.1   dataset.seg_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/scpp-seg-0.1   dataset.seg_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/scpp-seg-0.1   outputs.out_roots.HCP_YA=/path/to/datasets/hcpya-u100/derivatives/scpp-initsurf-0.1   outputs.out_roots.OASIS1=/path/to/datasets/oasis-1/derivatives/scpp-initsurf-0.1   outputs.log_dir=/path/to/scpp-runs/initsurf/exp01/logs_generate
```

Typical runtime: ~31 s/subject (hardware-dependent).

---

## Stage 4 — Deformation (Deform)

This stage deforms InitSurf meshes using input volumes and geometric losses, and writes **deformed** surfaces to a BIDS-derivatives folder.

### Inputs
- Preproc root (`scpp-preproc-*`): MNI T1 + GT FreeSurfer surfaces in MNI space
- InitSurf root (`scpp-initsurf-*`): ribbon probability + initial surfaces
- Split CSV (same format as Stage 2)

### Outputs
Deformed surfaces under `scpp-deform-*`:
```text
scpp-deform-0.1/
  dataset_description.json
  sub-XXXX/
    ses-01/
      surfaces/
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-L_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-L_pial.surf.ply
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-R_white.surf.ply
        sub-XXXX_ses-01_space-MNI152_desc-deform_hemi-R_pial.surf.ply
```

### Train (multi-GPU example)
```bash
scpp deform train --torchrun --nproc-per-node 2   outputs.root=/path/to/scpp-runs/deform/exp01
```

### Inference
```bash
scpp deform infer
```

### Evaluation
```bash
scpp deform eval
```

Evaluation writes four Excel files:
- `surface_metrics.xlsx`
- `collision_metrics.xlsx`
- `collision_metrics_enhanced.xlsx`
- `collision_summary.xlsx`

---

## Outputs Summary

For each dataset root:

- `derivatives/scpp-preproc-0.1/`  
  MNI T1, aparc+aseg, filled, transforms, and MNI-aligned FreeSurfer surfaces.

- `derivatives/scpp-seg-0.1/`  
  9-class segmentation predictions (`*_desc-seg9_dseg.nii.gz`).

- `derivatives/scpp-initsurf-0.1/`  
  Initial surfaces (`*.surf.ply`) + SDF volumes + ribbon SDF/probability.

- `derivatives/scpp-deform-0.1/`  
  Deformed surfaces (`*_desc-deform_*.surf.ply`).

---

