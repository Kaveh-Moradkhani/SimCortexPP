# Nipoppy Custom Pipeline Integration Example

This repository is a **hands-on template** for integrating a custom MRI processing pipeline into the **Nipoppy** framework.  
So far, we focused on creating a clean project skeleton and preparing **session-less BIDS** datasets for Nipoppy **without modifying the original data**.

---

## What we have done so far

### 1) Created a CLI-first Python project skeleton
We set up a standard Python package structure and a minimal CLI so this repo can evolve into a real pipeline package.

- Python package: `src/nipoppy_custom_pipeline/`
- CLI entrypoint: `ncp` (Typer)
  - `ncp hello` — sanity check
  - `ncp check-env` — prints Python version and verifies `nipoppy` imports
- Packaging: `pyproject.toml` (editable install via `pip install -e .`)
- Standard folders: `docs/`, `tests/`, `scripts/`

### 2) Identified a common dataset issue: session-less BIDS
Some BIDS datasets are organized at the **subject level** only (no `ses-*` folders).  
When initializing Nipoppy directly from such a dataset, Nipoppy may create a manifest with a dummy session label (e.g., `unnamed`), which is not ideal for consistent tracking.

### 3) Created a **non-destructive sessioned BIDS view** (symlinks)
We implemented a small helper script that builds a session-aware view of an existing session-less BIDS dataset:

- Original dataset remains unchanged (source stays intact)
- New dataset view is created using **symlinks**
- Adds a `ses-01/` level and places data under `ses-01/anat/`
- Renames filenames to include `_ses-01_` where needed (BIDS-style)

Script: `scripts/create_ses_view.sh`

### 4) Initialized a Nipoppy project pointing to the sessioned view
We initialized a Nipoppy dataset root using `nipoppy init --bids-source <rawdata_ses>`, so the generated `manifest.tsv` contains a real `session_id` (e.g., `01`) instead of `unnamed`.

### 5) Linked existing FreeSurfer derivatives into the Nipoppy project
If FreeSurfer has already been run, we link its output directory into the Nipoppy dataset root to avoid copying large files.

---

## Terminology and folder roles (important)

You will usually have **three separate roots**:

1) **Code repo (this repository)**  
   Example: `/project/.../workspace/nipoppy-custom-pipeline`  
   Contains code + scripts (no data).

2) **Study dataset (your raw + derivatives)**  
   Example: `/project/.../dataset/HCP_YA_U100/`  
   Contains:
   - `rawdata/` (original BIDS-ish data)
   - `rawdata_ses/` (sessioned symlink view we create)
   - `derivatives/` (e.g., FreeSurfer outputs)

3) **Nipoppy dataset root (workspace for a study)**  
   Example: `/project/.../workspace/nipoppy-projects/<study-name>/`  
   Contains Nipoppy structure (`global_config.json`, `manifest.tsv`, `pipelines/`, etc.) and points to data via symlinks.

> This repo is meant to be reusable across many studies.  
> The Nipoppy dataset root name can be study-specific (e.g., `nipoppy-hcpya-u100`, `nipoppy-oasis1`, etc.).

---

## Repository layout

```text
nipoppy-custom-pipeline/
├── src/
│   └── nipoppy_custom_pipeline/
│       ├── cli/
│       │   └── main.py            # 'ncp' CLI entrypoint (Typer)
│       └── utils/
├── scripts/
│   └── create_ses_view.sh         # Build a sessioned BIDS view using symlinks
├── pyproject.toml
├── README.md
└── tests/
```

---

## Install and run the CLI (smoke test)

From the repository root:

```bash
pip install -e .
ncp hello
ncp check-env
```

---

## Create a sessioned BIDS view (rawdata_ses)

### Why this step?
If your dataset has `sub-*/anat/...` but no `ses-*` level, Nipoppy may assign `session_id=unnamed`.  
This script creates a **session-aware view** so Nipoppy produces a clean manifest with `session_id=01`.

### How it works
- Input: a session-less BIDS dataset (subject-level)
- Output: a new dataset folder with:
  - `sub-XXXX/ses-01/anat/`
  - symlinked files
  - `_ses-01_` injected into filenames when needed

### Run it
Open the script and edit the variables at the top:

```bash
nano scripts/create_ses_view.sh
```

Set these three variables:

```bash
SRC="/path/to/rawdata"
DST="/path/to/rawdata_ses"
SES="01"
```

Then run:

```bash
bash scripts/create_ses_view.sh
```

✅ Result: `rawdata_ses/` is a **safe, disposable symlink view**.  
To undo it, just delete the `rawdata_ses/` directory.

---

## Initialize a Nipoppy dataset from the sessioned view

Example:

```bash
NIPROOT=/path/to/nipoppy-projects/<study-name>
nipoppy init --dataset "$NIPROOT" --bids-source "/path/to/rawdata_ses"
```

This creates the Nipoppy folder structure and a `manifest.tsv` with proper session IDs.

---

## Link existing FreeSurfer derivatives into Nipoppy

FreeSurfer outputs commonly live under a BIDS derivatives folder and do **not** need to be restructured.
BIDS explicitly allows FreeSurfer-style derivatives under `derivatives/freesurfer/.../sub-XX/...`.

Example:

```bash
ln -sfn "/path/to/study/derivatives/freesurfer-7.4.1"        "$NIPROOT/derivatives/freesurfer-7.4.1"
```

---

## Next steps (coming next)

- Decide the I/O contract for the custom pipeline (inputs from BIDS, outputs to derivatives)
- Containerize the pipeline (Apptainer/Singularity on Narval)
- Add a Boutiques descriptor + invocation template
- Register the pipeline in `global_config.json` under `PROC_PIPELINES`
- Run the pipeline via Nipoppy and validate outputs
