from __future__ import annotations

import os
import logging
from typing import List

import numpy as np
import nibabel as nib
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from simcortexpp.deform.utils.coords import (
    world_to_voxel,
    make_center_crop_pad_slices,
)

logger = logging.getLogger(__name__)


# ----------------------------
# BIDS-derivatives path helpers
# ----------------------------
def _ses(session_label: str) -> str:
    return f"ses-{session_label}"


def mni_t1_path(preproc_root: str, subj: str, session_label: str, space: str) -> str:
    ses = _ses(session_label)
    return os.path.join(
        preproc_root, subj, ses, "anat",
        f"{subj}_{ses}_space-{space}_desc-preproc_T1w.nii.gz",
    )


def ribbon_prob_path(initsurf_root: str, subj: str, session_label: str, space: str) -> str:
    ses = _ses(session_label)
    return os.path.join(
        initsurf_root, subj, ses, "anat",
        f"{subj}_{ses}_space-{space}_desc-ribbon_prob.nii.gz",
    )


_SURF_MAP = {
    "lh_pial":  ("L", "pial"),
    "lh_white": ("L", "white"),
    "rh_pial":  ("R", "pial"),
    "rh_white": ("R", "white"),
}


def surf_path(root: str, subj: str, session_label: str, space: str, surf_name: str) -> str:
    ses = _ses(session_label)
    hemi, surf = _SURF_MAP[surf_name]
    return os.path.join(
        root, subj, ses, "surfaces",
        f"{subj}_{ses}_space-{space}_hemi-{hemi}_{surf}.surf.ply",
    )


# ----------------------------
# IO helpers
# ----------------------------
def read_nii(path: str):
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    aff = nii.affine.astype(np.float32)
    return vol, aff


def read_mesh(path: str):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = next(iter(m.geometry.values()))
    v = np.asarray(m.vertices, dtype=np.float32)
    f = np.asarray(m.faces, dtype=np.int64)
    return v, f


def normalize_mri_mean_std(mri: np.ndarray) -> np.ndarray:
    mask = (mri != 0)
    if mask.sum() < 100:
        m = float(mri.mean())
        s = float(mri.std())
    else:
        m = float(mri[mask].mean())
        s = float(mri[mask].std())
    s = max(s, 1e-6)
    return ((mri - m) / s).astype(np.float32)


def _norm01_by_p99(x: np.ndarray, eps=1e-6) -> np.ndarray:
    p = np.percentile(x, 99)
    p = max(float(p), eps)
    y = x / p
    return np.clip(y, 0.0, 1.0).astype(np.float32)


class CSRDeformDataset(Dataset):
    """
    Returns per subject:
      vol: (C,D,H,W) float32  [MRI, RIBBON_PROB, (optional) PROB_GRAD]
      affine: (4,4) float32 (vox->world)
      shift_ijk: (3,) float32
      init_verts_vox[surf], init_faces[surf]
      gt_verts_vox[surf], gt_faces[surf]
    """

    def __init__(
        self,
        preproc_root: str,
        initsurf_root: str,
        subjects: List[str],
        session_label: str,
        space: str,
        surface_names,
        inshape_dhw,
        prob_clip_min: float = 0.0,
        prob_clip_max: float = 1.0,
        prob_gamma: float = 1.0,
        add_prob_grad: bool = False,
        aug: bool = False,
    ):
        self.preproc_root = str(preproc_root)
        self.initsurf_root = str(initsurf_root)
        self.subjects = [str(s) for s in subjects]
        self.session_label = str(session_label)
        self.space = str(space)

        self.surface_names = list(surface_names)
        self.inshape = tuple(int(x) for x in inshape_dhw)

        self.prob_clip_min = float(prob_clip_min)
        self.prob_clip_max = float(prob_clip_max)
        self.prob_gamma = float(prob_gamma)
        self.add_prob_grad = bool(add_prob_grad)

        self.aug = bool(aug)  # (reserved) augmentation is applied in train.py

        self.samples = []
        dropped = 0

        for subj in self.subjects:
            mri_path = mni_t1_path(self.preproc_root, subj, self.session_label, self.space)
            prob_path = ribbon_prob_path(self.initsurf_root, subj, self.session_label, self.space)

            gt_paths = {s: surf_path(self.preproc_root, subj, self.session_label, self.space, s) for s in self.surface_names}
            ini_paths = {s: surf_path(self.initsurf_root, subj, self.session_label, self.space, s) for s in self.surface_names}

            missing = []
            if not os.path.isfile(mri_path): missing.append(mri_path)
            if not os.path.isfile(prob_path): missing.append(prob_path)
            for s in self.surface_names:
                if not os.path.isfile(gt_paths[s]):  missing.append(gt_paths[s])
                if not os.path.isfile(ini_paths[s]): missing.append(ini_paths[s])

            if missing:
                dropped += 1
                continue

            self.samples.append((subj, mri_path, prob_path, gt_paths, ini_paths))

        if dropped > 0:
            logger.warning(f"[CSRDeformDataset] Dropped {dropped} subjects due to missing files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj, mri_path, prob_path, gt_paths, ini_paths = self.samples[idx]
        mri, affine = read_nii(mri_path)
        prob, _ = read_nii(prob_path)

        if prob.shape != mri.shape:
            raise ValueError(f"PROB/MRI shape mismatch for {subj}: {prob.shape} vs {mri.shape}")

        mri = normalize_mri_mean_std(mri)

        prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        if self.prob_clip_min > 0:
            prob[prob < self.prob_clip_min] = 0.0
        prob = np.clip(prob, 0.0, self.prob_clip_max).astype(np.float32)
        if abs(self.prob_gamma - 1.0) > 1e-6:
            prob = np.power(prob, self.prob_gamma).astype(np.float32)

        D0, H0, W0 = mri.shape
        D1, H1, W1 = self.inshape

        crop_slices, pad_before, pad_after, crop_before = make_center_crop_pad_slices(
            (D0, H0, W0), (D1, H1, W1)
        )

        mri_c = mri[crop_slices[0], crop_slices[1], crop_slices[2]]
        prob_c = prob[crop_slices[0], crop_slices[1], crop_slices[2]]

        pbD, pbH, pbW = pad_before
        paD, paH, paW = pad_after

        mri_t = torch.from_numpy(mri_c)[None, None]
        prob_t = torch.from_numpy(prob_c)[None, None]

        mri_t = F.pad(mri_t, (pbW, paW, pbH, paH, pbD, paD), mode="replicate")
        prob_t = F.pad(prob_t, (pbW, paW, pbH, paH, pbD, paD), mode="constant", value=0.0)

        mri_out = mri_t[0, 0].numpy()
        prob_out = prob_t[0, 0].numpy()
        assert mri_out.shape == self.inshape, (mri_out.shape, self.inshape)

        prob_grad_out = None
        if self.add_prob_grad:
            gx, gy, gz = np.gradient(prob_out.astype(np.float32))
            gmag = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)
            prob_grad_out = _norm01_by_p99(gmag)

        shift_ijk = np.array(pad_before, dtype=np.float32) - np.array(crop_before, dtype=np.float32)

        A = torch.from_numpy(affine).float()
        init_verts_vox, init_faces = {}, {}
        gt_verts_vox, gt_faces = {}, {}

        for s in self.surface_names:
            v_ini_mm, f_ini = read_mesh(ini_paths[s])
            v_gt_mm,  f_gt  = read_mesh(gt_paths[s])

            v_ini = world_to_voxel(torch.from_numpy(v_ini_mm).float(), A).numpy()
            v_gt  = world_to_voxel(torch.from_numpy(v_gt_mm).float(),  A).numpy()

            v_ini = (v_ini + shift_ijk).astype(np.float32)
            v_gt  = (v_gt  + shift_ijk).astype(np.float32)

            init_verts_vox[s] = torch.from_numpy(v_ini).float()
            init_faces[s] = torch.from_numpy(f_ini).long()
            gt_verts_vox[s] = torch.from_numpy(v_gt).float()
            gt_faces[s] = torch.from_numpy(f_gt).long()

        chans = [
            torch.from_numpy(mri_out).float(),
            torch.from_numpy(prob_out).float(),
        ]
        if prob_grad_out is not None:
            chans.append(torch.from_numpy(prob_grad_out).float())

        vol = torch.stack(chans, dim=0)  # (C,D,H,W)

        return {
            "subject": subj,
            "vol": vol,
            "affine": torch.from_numpy(affine).float(),
            "shift_ijk": torch.from_numpy(shift_ijk).float(),
            "init_verts_vox": init_verts_vox,
            "init_faces": init_faces,
            "gt_verts_vox": gt_verts_vox,
            "gt_faces": gt_faces,
        }


def collate_csr_deform(batch_list):
    return {
        "subject": [b["subject"] for b in batch_list],
        "vol": torch.stack([b["vol"] for b in batch_list], dim=0),
        "affine": torch.stack([b["affine"] for b in batch_list], dim=0),
        "shift_ijk": torch.stack([b["shift_ijk"] for b in batch_list], dim=0),
        "init_verts_vox": [b["init_verts_vox"] for b in batch_list],
        "init_faces": [b["init_faces"] for b in batch_list],
        "gt_verts_vox": [b["gt_verts_vox"] for b in batch_list],
        "gt_faces": [b["gt_faces"] for b in batch_list],
    }