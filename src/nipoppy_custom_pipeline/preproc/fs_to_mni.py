#!/usr/bin/env python3
"""
SimCortexPP Preprocessing: FreeSurfer -> MNI (BIDS-Derivatives-inspired outputs)

Outputs (example):
  <OUT_DERIV_ROOT>/
    dataset_description.json
    sub-100307/
      ses-01/
        anat/
          sub-100307_ses-01_desc-preproc_T1w.nii.gz
          sub-100307_ses-01_space-MNI152_desc-preproc_T1w.nii.gz
          sub-100307_ses-01_desc-aseg_dseg.nii.gz
          sub-100307_ses-01_space-MNI152_desc-aseg_dseg.nii.gz
          sub-100307_ses-01_desc-aparc+aseg_dseg.nii.gz
          sub-100307_ses-01_space-MNI152_desc-aparc+aseg_dseg.nii.gz
          sub-100307_ses-01_desc-filled_T1w.nii.gz
          sub-100307_ses-01_space-MNI152_desc-filled_T1w.nii.gz
          sub-100307_ses-01_from-T1w_to-MNI152_mode-image_xfm.txt
        surfaces/
          sub-100307_ses-01_hemi-L_white.surf.ply
          sub-100307_ses-01_space-MNI152_hemi-L_white.surf.ply
          sub-100307_ses-01_hemi-L_pial.surf.ply
          sub-100307_ses-01_space-MNI152_hemi-L_pial.surf.ply
          ...

Requirements:
- Python: numpy, nibabel, typer, trimesh (only for decimation)
- External commands:
    - reg_aladin, reg_resample (NiftyReg)

Notes:
- Surface reading is done via nibabel (no mris_convert needed).
- If a canonical FreeSurfer surface is missing (e.g., lh.pial), we fall back to lh.pial.T1
  and convert coordinates from scanner RAS -> tkr RAS using T1.mgz header matrices.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import nibabel as nib
from nibabel.freesurfer.io import read_geometry
import typer

# trimesh is optional unless --decimate is used
try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None


APP_NAME = "SimCortexPP-Preproc"
__version__ = "0.1"
PIPELINE_NAME = f"scpp-preproc-{__version__}"

app = typer.Typer(add_completion=False, invoke_without_command=True, help="FreeSurfer -> MNI preprocessing.")


# -------------------------
# Logging
# -------------------------
def setup_logger(verbosity: int = 0, log_file: Optional[Path] = None) -> logging.Logger:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logger = logging.getLogger("scpp-preproc")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -------------------------
# Small helpers
# -------------------------
def strip_prefix(x: str, prefix: str) -> str:
    return x[len(prefix):] if x.startswith(prefix) else x


def bids_sub_id(label: str) -> str:
    return f"sub-{strip_prefix(label, 'sub-')}"


def bids_ses_id(label: str) -> str:
    return f"ses-{strip_prefix(label, 'ses-')}"


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable not found in PATH: {name}")


def run_cmd(cmd: list[str], logger: logging.Logger) -> None:
    logger.debug("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    if p.stdout.strip():
        logger.debug("STDOUT: %s", p.stdout.strip())
    if p.stderr.strip():
        logger.debug("STDERR: %s", p.stderr.strip())


def apply_affine(aff: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    h = np.concatenate([pts, ones], axis=1)
    out = (h @ aff.T)[:, :3]
    return out


def inside_fraction(vertices_world: np.ndarray, ref_img: nib.Nifti1Image) -> float:
    vox = nib.affines.apply_affine(np.linalg.inv(ref_img.affine), vertices_world)
    shape = np.array(ref_img.shape[:3], dtype=np.float64)
    inside = np.all((vox >= 0) & (vox < shape), axis=1)
    return float(np.mean(inside))


def read_affine_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path))
    if arr.shape != (4, 4):
        raise ValueError(f"Unexpected affine shape in {path}: {arr.shape}")
    return arr.astype(np.float64)


def write_dataset_description(out_root: Path) -> None:
    dd = {
        "Name": PIPELINE_NAME,
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": APP_NAME,
                "Version": __version__,
                "Description": "FreeSurfer-derived preprocessing: export volumes/surfaces and affine normalize to MNI.",
            }
        ],
    }
    f = out_root / "dataset_description.json"
    if not f.exists():
        f.write_text(json.dumps(dd, indent=2) + "\n")


def save_like_ply_ascii(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Minimal, robust ASCII PLY writer (triangles).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Faces must be (M, 3) triangles")

    with path.open("w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {v.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write(f"element face {f.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for i in range(v.shape[0]):
            fp.write(f"{v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f}\n")
        for j in range(f.shape[0]):
            fp.write(f"3 {f[j,0]} {f[j,1]} {f[j,2]}\n")


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Decimate mesh using fast_simplification directly.
    ratio: Target fraction of faces to KEEP (e.g. 0.3).
    """
    try:
        import fast_simplification
    except ImportError:
        raise RuntimeError("fast_simplification is required. Run: pip install fast-simplification")

    # Calculate reduction amount (amount to REMOVE)
    # If ratio is 0.3 (keep 30%), we reduce by 0.7 (remove 70%)
    target_reduction = 1.0 - float(ratio)
    
    # Safety clamp to ensure it's always between 0 and 1
    target_reduction = max(0.0, min(0.999, target_reduction))

    # Run simplification directly
    v_new, f_new = fast_simplification.simplify(
        vertices, 
        faces, 
        target_reduction=target_reduction
    )
    
    return v_new, f_new


# -------------------------
# FreeSurfer subject discovery
# -------------------------
def is_fs_subject_dir(d: Path) -> bool:
    return d.is_dir() and (d / "surf").is_dir() and (d / "mri").is_dir()


def discover_subjects(fs_root: Path) -> list[str]:
    """
    Prefer sub-* naming; otherwise include all directories that look like FS subjects.
    """
    subs = [d.name for d in fs_root.iterdir() if d.is_dir() and d.name.startswith("sub-") and is_fs_subject_dir(d)]
    subs.sort()
    if subs:
        return subs

    # fallback: classic SUBJECTS_DIR style (e.g., 100307)
    subs = [d.name for d in fs_root.iterdir() if is_fs_subject_dir(d)]
    subs.sort()
    return subs


def find_fs_subject_dir(fs_root: Path, sub_label: str) -> Path:
    """
    Accept either sub-XXXX or XXXX.
    """
    sub = bids_sub_id(sub_label)
    cand1 = fs_root / sub
    if is_fs_subject_dir(cand1):
        return cand1

    cand2 = fs_root / strip_prefix(sub_label, "sub-")
    if is_fs_subject_dir(cand2):
        return cand2

    # if user passed a directory name already existing
    cand3 = fs_root / sub_label
    if is_fs_subject_dir(cand3):
        return cand3

    raise FileNotFoundError(f"Could not find FreeSurfer subject directory for {sub_label} under {fs_root}")


# -------------------------
# Surface reading with fallback
# -------------------------
@dataclass(frozen=True)
class SurfaceCandidate:
    path: Path
    is_scanner_ras: bool  # True for *.T1 surfaces


def surface_candidates(fs_surf_dir: Path, fs_hemi: str, name: str) -> list[SurfaceCandidate]:
    """
    Canonical FS surfaces are typically:
      lh.white, lh.pial, rh.white, rh.pial

    Some datasets may instead contain:
      lh.pial.T1, rh.pial.T1
    """
    cands: list[SurfaceCandidate] = []
    base = fs_surf_dir / f"{fs_hemi}.{name}"
    if base.exists():
        cands.append(SurfaceCandidate(base, is_scanner_ras=False))

    # Fallback for missing pial surfaces (and sometimes others): *.T1
    t1 = fs_surf_dir / f"{fs_hemi}.{name}.T1"
    if t1.exists():
        cands.append(SurfaceCandidate(t1, is_scanner_ras=True))

    return cands


def scanner_to_tkr_affine(t1_mgz: nib.spatialimages.SpatialImage) -> np.ndarray:
    """
    Build affine that maps scanner RAS -> tkr RAS using T1.mgz header:
      v_tkr = vox2ras_tkr @ inv(vox2ras) @ v_scanner
    """
    hdr = t1_mgz.header
    # these are MGHHeader methods
    vox2ras = hdr.get_vox2ras()
    vox2ras_tkr = hdr.get_vox2ras_tkr()
    return vox2ras_tkr @ np.linalg.inv(vox2ras)


def read_fs_surface_vertices_faces(
    fs_surf_dir: Path,
    fs_hemi: str,
    surf_name: str,
    t1_mgz: nib.spatialimages.SpatialImage,
    logger: logging.Logger,
    stem: str,
) -> Optional[tuple[np.ndarray, np.ndarray, Path]]:
    
    cands = surface_candidates(fs_surf_dir, fs_hemi, surf_name)
    if not cands:
        return None

    cand = cands[0]
    
    temp_scanner = cand.path.with_suffix(".tmp_scanner")
    try:
        subprocess.run(["mris_convert", "--to-scanner", str(cand.path), str(temp_scanner)], 
                       check=True, capture_output=True)
        
        v, f = read_geometry(str(temp_scanner))
        
        if temp_scanner.exists():
            temp_scanner.unlink()
            
    except Exception as e:
        logger.error("[%s] Error aligning surface %s: %s", stem, cand.path.name, e)
        v, f = read_geometry(str(cand.path))
        hdr = t1_mgz.header
        M = hdr.get_vox2ras_tkr()
        M_inv = np.linalg.inv(M)
        v2w = hdr.get_vox2ras()
        full_m = v2w @ M_inv
        v = apply_affine(full_m, v)

    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), cand.path

# -------------------------
# Core per-subject processing
# -------------------------
def process_one(
    *,
    fs_root: Path,
    out_root: Path,
    participant_label: str,
    session_label: str,
    mni_template: Path,
    space: str,
    surface_names: Sequence[str],
    hemis: Sequence[str],
    write_aparc_aseg: bool,
    write_filled: bool,
    decimate_ratio: Optional[float],
    skip_existing: bool,
    affine_for_surfaces: Optional[str],  # forward|inverse|None(auto)
    strict_surfaces: bool,
    logger: logging.Logger,
) -> None:
    sub = bids_sub_id(participant_label)
    ses = bids_ses_id(session_label)
    stem = f"{sub}_{ses}"

    fs_sub_dir = find_fs_subject_dir(fs_root, sub)
    fs_mri = fs_sub_dir / "mri"
    fs_surf = fs_sub_dir / "surf"

    out_sub = out_root / sub / ses
    out_anat = out_sub / "anat"
    out_surfaces = out_sub / "surfaces"
    safe_mkdir(out_anat)
    safe_mkdir(out_surfaces)

    # Output paths
    f_t1_native = out_anat / f"{stem}_desc-preproc_T1w.nii.gz"
    f_t1_mni = out_anat / f"{stem}_space-{space}_desc-preproc_T1w.nii.gz"
    f_aff = out_anat / f"{stem}_from-T1w_to-{space}_mode-image_xfm.txt"

    f_aseg_native = out_anat / f"{stem}_desc-aseg_dseg.nii.gz"
    f_aseg_mni = out_anat / f"{stem}_space-{space}_desc-aseg_dseg.nii.gz"

    f_aparc_native = out_anat / f"{stem}_desc-aparc+aseg_dseg.nii.gz"
    f_aparc_mni = out_anat / f"{stem}_space-{space}_desc-aparc+aseg_dseg.nii.gz"

    f_filled_native = out_anat / f"{stem}_desc-filled_T1w.nii.gz"
    f_filled_mni = out_anat / f"{stem}_space-{space}_desc-filled_T1w.nii.gz"

    # Load T1.mgz once (needed for surface fallback conversion too)
    t1_mgz_path = fs_mri / "T1.mgz"
    if not t1_mgz_path.exists():
        raise FileNotFoundError(f"Missing {t1_mgz_path}")
    t1_mgz = nib.load(str(t1_mgz_path))

    # 1) Export volumes (mgz -> nii.gz)
    if (not f_t1_native.exists()) or (not skip_existing):
        logger.info("[%s] Export T1.mgz -> %s", stem, f_t1_native.name)
        nib.save(t1_mgz, str(f_t1_native))

    aseg_mgz = fs_mri / "aseg.mgz"
    if (not f_aseg_native.exists()) or (not skip_existing):
        if not aseg_mgz.exists():
            raise FileNotFoundError(f"Missing {aseg_mgz}")
        logger.info("[%s] Export aseg.mgz -> %s", stem, f_aseg_native.name)
        nib.save(nib.load(str(aseg_mgz)), str(f_aseg_native))

    if write_aparc_aseg:
        aparc_mgz = fs_mri / "aparc+aseg.mgz"
        if aparc_mgz.exists() and ((not f_aparc_native.exists()) or (not skip_existing)):
            logger.info("[%s] Export aparc+aseg.mgz -> %s", stem, f_aparc_native.name)
            nib.save(nib.load(str(aparc_mgz)), str(f_aparc_native))

    if write_filled:
        filled_mgz = fs_mri / "filled.mgz"
        if filled_mgz.exists() and ((not f_filled_native.exists()) or (not skip_existing)):
            logger.info("[%s] Export filled.mgz -> %s", stem, f_filled_native.name)
            nib.save(nib.load(str(filled_mgz)), str(f_filled_native))

    # 2) Register T1 to template (affine)
    if (not f_t1_mni.exists()) or (not f_aff.exists()) or (not skip_existing):
        logger.info("[%s] reg_aladin: T1w -> space-%s (affine)", stem, space)
        run_cmd(
            [
                "reg_aladin",
                "-ref",
                str(mni_template),
                "-flo",
                str(f_t1_native),
                "-res",
                str(f_t1_mni),
                "-aff",
                str(f_aff),
            ],
            logger=logger,
        )

    # 3) Resample labels / masks
    if (not f_aseg_mni.exists()) or (not skip_existing):
        logger.info("[%s] reg_resample: aseg -> space-%s (NN)", stem, space)
        run_cmd(
            [
                "reg_resample",
                "-ref",
                str(mni_template),
                "-flo",
                str(f_aseg_native),
                "-res",
                str(f_aseg_mni),
                "-aff",
                str(f_aff),
                "-inter",
                "0",
            ],
            logger=logger,
        )

    if write_aparc_aseg and f_aparc_native.exists():
        if (not f_aparc_mni.exists()) or (not skip_existing):
            logger.info("[%s] reg_resample: aparc+aseg -> space-%s (NN)", stem, space)
            run_cmd(
                [
                    "reg_resample",
                    "-ref",
                    str(mni_template),
                    "-flo",
                    str(f_aparc_native),
                    "-res",
                    str(f_aparc_mni),
                    "-aff",
                    str(f_aff),
                    "-inter",
                    "0",
                ],
                logger=logger,
            )

    if write_filled and f_filled_native.exists():
        if (not f_filled_mni.exists()) or (not skip_existing):
            logger.info("[%s] reg_resample: filled -> space-%s (linear)", stem, space)
            run_cmd(
                [
                    "reg_resample",
                    "-ref",
                    str(mni_template),
                    "-flo",
                    str(f_filled_native),
                    "-res",
                    str(f_filled_mni),
                    "-aff",
                    str(f_aff),
                    "-inter",
                    "1",
                ],
                logger=logger,
            )

    # 4) Prepare surface transform (choose ONCE per subject)
    aff = read_affine_txt(f_aff)
    inv_aff = np.linalg.inv(aff)
    ref_img = nib.load(str(f_t1_mni))

    # Pick a representative surface (prefer lh.white, then rh.white, then anything available)
    rep_vertices = None
    for hemi in ("L", "R"):
        fs_hemi = "lh" if hemi == "L" else "rh"
        rep = read_fs_surface_vertices_faces(fs_surf, fs_hemi, "white", t1_mgz, logger, stem)
        if rep is not None:
            rep_vertices = rep[0]
            break

    def choose_subject_surface_affine() -> np.ndarray:
        if affine_for_surfaces == "forward":
            return aff
        if affine_for_surfaces == "inverse":
            return inv_aff
        if rep_vertices is None:
            # no representative surface -> default to forward
            logger.warning("[%s] No representative surface found for auto affine; defaulting to forward.", stem)
            return aff

        fwd_inside = inside_fraction(apply_affine(aff, rep_vertices), ref_img)
        inv_inside = inside_fraction(apply_affine(inv_aff, rep_vertices), ref_img)
        logger.debug("[%s] Subject surface inside fraction: forward=%.4f inverse=%.4f", stem, fwd_inside, inv_inside)
        return aff if fwd_inside >= inv_inside else inv_aff

    surf_aff = choose_subject_surface_affine()

    # 5) Export surfaces (native + template)
    for hemi in hemis:
        hemi_u = hemi.upper()
        if hemi_u not in ("L", "R"):
            raise ValueError("hemi must be L or R")
        fs_hemi = "lh" if hemi_u == "L" else "rh"

        for sname in surface_names:
            sname = sname.strip()
            rep = read_fs_surface_vertices_faces(fs_surf, fs_hemi, sname, t1_mgz, logger, stem)

            if rep is None:
                msg = f"[{stem}] Missing surface: {fs_hemi}.{sname} (and no .T1 fallback)"
                if strict_surfaces:
                    raise FileNotFoundError(msg)
                logger.warning(msg)
                continue

            v_native, f_native, used_path = rep

            out_native = out_surfaces / f"{stem}_hemi-{hemi_u}_{sname}.surf.ply"
            out_mni = out_surfaces / f"{stem}_space-{space}_hemi-{hemi_u}_{sname}.surf.ply"

            if (not out_native.exists()) or (not skip_existing):
                logger.info("[%s] Export surface (native): %s", stem, out_native.name)
                save_like_ply_ascii(out_native, v_native, f_native)

            if (not out_mni.exists()) or (not skip_existing):
                logger.info("[%s] Write surface (space-%s): %s", stem, space, out_mni.name)
                v_mni = apply_affine(surf_aff, v_native)
                save_like_ply_ascii(out_mni, v_mni, f_native)

            if decimate_ratio is not None:
                tag = str(decimate_ratio).replace(".", "p")
                out_native_dec = out_surfaces / f"{stem}_desc-decim{tag}_hemi-{hemi_u}_{sname}.surf.ply"
                out_mni_dec = out_surfaces / f"{stem}_space-{space}_desc-decim{tag}_hemi-{hemi_u}_{sname}.surf.ply"

                if (not out_native_dec.exists()) or (not skip_existing):
                    logger.info("[%s] Decimate native (%.3f): %s", stem, decimate_ratio, out_native_dec.name)
                    v_dec, f_dec = decimate_mesh(v_native, f_native, decimate_ratio)
                    save_like_ply_ascii(out_native_dec, v_dec, f_dec)
                else:
                    # If already exists, still need f_dec for MNI decimation; reload via simple parse is annoying.
                    # We'll just recompute decimation (fast enough compared to registration).
                    v_dec, f_dec = decimate_mesh(v_native, f_native, decimate_ratio)

                if (not out_mni_dec.exists()) or (not skip_existing):
                    logger.info("[%s] Decimate MNI (%.3f): %s", stem, decimate_ratio, out_mni_dec.name)
                    v_dec_mni = apply_affine(surf_aff, v_dec)
                    save_like_ply_ascii(out_mni_dec, v_dec_mni, f_dec)


# -------------------------
# CLI (single entry)
# -------------------------
@app.callback()
def main(
    freesurfer_root: Path = typer.Option(..., "--freesurfer-root", exists=True, file_okay=False, dir_okay=True),
    out_deriv_root: Path = typer.Option(..., "--out-deriv-root"),
    mni_template: Path = typer.Option(..., "--mni-template", exists=True, file_okay=True, dir_okay=False),
    participant_label: Optional[list[str]] = typer.Option(
        None,
        "--participant-label",
        "-p",
        help="If omitted, all FS subjects found under --freesurfer-root will be processed.",
    ),
    session_label: str = typer.Option("01", "--session-label", "-s"),
    space: str = typer.Option("MNI152", "--space"),
    hemi: list[str] = typer.Option(["L", "R"], "--hemi"),
    surface: list[str] = typer.Option(["white", "pial"], "--surface"),
    with_aparc_aseg: bool = typer.Option(True, "--with-aparc-aseg/--no-aparc-aseg"),
    with_filled: bool = typer.Option(True, "--with-filled/--no-filled"),
    decimate: Optional[float] = typer.Option(None, "--decimate", min=0.01, max=1.0),
    overwrite: bool = typer.Option(False, "--overwrite", help="Recompute outputs even if files exist."),
    affine_for_surfaces: Optional[str] = typer.Option(
        None,
        "--affine-for-surfaces",
        help="Force surface affine direction: forward|inverse. Default: auto (choose once per subject).",
    ),
    strict_surfaces: bool = typer.Option(
        False,
        "--strict-surfaces",
        help="If set, fail the subject if any requested surface is missing (including .T1 fallback missing).",
    ),
    start: Optional[int] = typer.Option(None, "--start", help="Process subjects from this index (0-based) after sorting."),
    stop: Optional[int] = typer.Option(None, "--stop", help="Stop before this index (0-based) after sorting."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Optional path to write logs (e.g., pipeline.log)."),
    verbosity: int = typer.Option(0, "-v", count=True),
):
    """
    Example (all subjects):
      module load freesurfer
      module load niftyreg
      python scripts/preprocess_fs_to_mni_bidsderiv.py \\
        --freesurfer-root /path/to/freesurfer-7.4.1 \\
        --out-deriv-root  /path/to/derivatives/scpp-preproc-0.1 \\
        --mni-template    /path/to/MNI152_T1_1mm.nii.gz \\
        --decimate 0.3 -v \\
        --log-file pipeline.log
    """
    logger = setup_logger(verbosity, log_file=log_file)

    require_bin("reg_aladin")
    require_bin("reg_resample")

    if affine_for_surfaces not in (None, "forward", "inverse"):
        raise typer.BadParameter("--affine-for-surfaces must be one of: forward, inverse")

    safe_mkdir(out_deriv_root)
    write_dataset_description(out_deriv_root)

    # subject list
    if participant_label is None or len(participant_label) == 0:
        logger.info("No --participant-label provided. Discovering subjects in %s", freesurfer_root)
        subjects = discover_subjects(freesurfer_root)
        logger.info("Discovered %d subject(s).", len(subjects))
    else:
        subjects = [bids_sub_id(x) for x in participant_label]

    subjects = sorted(subjects)
    if start is not None or stop is not None:
        subjects = subjects[start:stop]
        logger.info("After slicing (--start/--stop), processing %d subject(s).", len(subjects))

    if not subjects:
        logger.error("No subjects to process.")
        raise typer.Exit(code=1)

    skip_existing = not overwrite

    failed: list[str] = []
    for s in subjects:
        try:
            process_one(
                fs_root=freesurfer_root,
                out_root=out_deriv_root,
                participant_label=s,
                session_label=session_label,
                mni_template=mni_template,
                space=space,
                surface_names=surface,
                hemis=hemi,
                write_aparc_aseg=with_aparc_aseg,
                write_filled=with_filled,
                decimate_ratio=decimate,
                skip_existing=skip_existing,
                affine_for_surfaces=affine_for_surfaces,
                strict_surfaces=strict_surfaces,
                logger=logger,
            )
        except Exception as e:
            failed.append(s)
            logger.error("[%s] FAILED: %s", s, e)

    if failed:
        logger.error("Done with failures (%d): %s", len(failed), ", ".join(failed))
        raise typer.Exit(code=1)

    logger.info("Done. Outputs written under: %s", out_deriv_root)


if __name__ == "__main__":
    app()
