# nipoppy_custom_pipeline/seg/data/dataloader.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib

from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Set, Tuple

from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandBiasFieldd,
)

# ---------------- Label mapping ----------------
LABEL_GROUPS: Dict[int, Set[int]] = {
    1: {2, 5, 10, 11, 12, 13, 26, 28, 30, 31},                      # lh white matter
    2: {41, 44, 49, 50, 51, 52, 58, 60, 62, 63},                    # rh white matter
    3: set(range(1000, 1004)) | set(range(1005, 1036)),             # lh cortex (pial)
    4: set(range(2000, 2004)) | set(range(2005, 2036)),             # rh cortex (pial)
    5: {17, 18},                                                    # lh amyg/hip
    6: {53, 54},                                                    # rh amyg/hip
    7: {4},                                                         # lh ventricle
    8: {43},                                                        # rh ventricle
}


def map_labels(seg_arr: np.ndarray, filled_arr: np.ndarray) -> np.ndarray:
    """
    Map FreeSurfer aparc+aseg labels into 8 groups, with the same ambiguity fix
    using 'filled' as your original code intended.
    """
    seg_mapped = np.zeros_like(seg_arr, dtype=np.int32)
    for cls, labels in LABEL_GROUPS.items():
        seg_mapped[np.isin(seg_arr, list(labels))] = cls

    # Make filled robust to interpolation artifacts (if any)
    filled_i = np.rint(filled_arr).astype(np.int32)

    ambiguous = np.isin(seg_arr, [77, 80])
    seg_mapped[ambiguous & (filled_i == 255)] = 1  # lh WM
    seg_mapped[ambiguous & (filled_i == 127)] = 2  # rh WM
    return seg_mapped


def robust_normalize(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    positive = vol[vol > 0]
    if positive.size == 0:
        return vol
    p99 = np.percentile(positive, 99)
    if p99 <= 0:
        return vol
    vol = np.clip(vol, 0, p99)
    return vol / p99


def get_augmentations() -> Compose:
    return Compose([
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandBiasFieldd(keys=["image"], prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.1, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    ])


def pad_vol_to_multiple(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    _, D, H, W = x.shape
    pads = (
        0, (mult - W % mult) % mult,
        0, (mult - H % mult) % mult,
        0, (mult - D % mult) % mult,
    )
    return F.pad(x, pads, mode="replicate")


def pad_seg_to_multiple(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    _, D, H, W = x.shape
    pads = (
        0, (mult - W % mult) % mult,
        0, (mult - H % mult) % mult,
        0, (mult - D % mult) % mult,
    )
    return F.pad(x, pads, mode="constant", value=0)


# ---------------- Path helpers (NEW derivative layout) ----------------
def _ses_id(session_label: str) -> str:
    return session_label if session_label.startswith("ses-") else f"ses-{session_label}"


def _stem(sub: str, ses: str) -> str:
    return f"{sub}_{ses}"


def _anat_dir(deriv_root: Path, sub: str, ses: str) -> Path:
    return deriv_root / sub / ses / "anat"


def _t1_mni_path(deriv_root: Path, sub: str, ses: str, space: str) -> Path:
    st = _stem(sub, ses)
    return _anat_dir(deriv_root, sub, ses) / f"{st}_space-{space}_desc-preproc_T1w.nii.gz"


def _aparc_aseg_mni_path(deriv_root: Path, sub: str, ses: str, space: str) -> Path:
    st = _stem(sub, ses)
    return _anat_dir(deriv_root, sub, ses) / f"{st}_space-{space}_desc-aparc+aseg_dseg.nii.gz"


def _filled_mni_path(deriv_root: Path, sub: str, ses: str, space: str) -> Path:
    st = _stem(sub, ses)
    return _anat_dir(deriv_root, sub, ses) / f"{st}_space-{space}_desc-filled_T1w.nii.gz"


def _pred_seg9_path(pred_root: Path, sub: str, ses: str, space: str) -> Path:
    st = _stem(sub, ses)
    return pred_root / sub / ses / "anat" / f"{st}_space-{space}_desc-seg9_pred.nii.gz"


def _read_split_subjects(split_csv: Path, split_name: str) -> list[str]:
    df = pd.read_csv(split_csv)
    if "subject" not in df.columns or "split" not in df.columns:
        raise ValueError(f"split_csv must have columns ['subject','split'], got: {list(df.columns)}")
    subs = df[df["split"] == split_name]["subject"].astype(str).tolist()
    subs = sorted(subs)
    if not subs:
        raise ValueError(f"No subjects found for split='{split_name}' in {split_csv}")
    return subs


class SegDataset(Dataset):
    def __init__(
        self,
        deriv_root: str,
        split_csv: str,
        split: str = "train",
        session_label: str = "01",
        space: str = "MNI152",
        pad_mult: int = 16,
        augment: bool = False,
    ):
        super().__init__()
        self.deriv_root = Path(deriv_root)
        self.split_csv = Path(split_csv)
        self.split = split
        self.ses = _ses_id(session_label)
        self.space = space
        self.pad_mult = pad_mult

        self.subjects = _read_split_subjects(self.split_csv, split)
        self.transforms = get_augmentations() if (split == "train" and augment) else None

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sub = self.subjects[idx]

        t1_path = _t1_mni_path(self.deriv_root, sub, self.ses, self.space)
        seg_path = _aparc_aseg_mni_path(self.deriv_root, sub, self.ses, self.space)
        fill_path = _filled_mni_path(self.deriv_root, sub, self.ses, self.space)

        if not t1_path.exists():
            raise FileNotFoundError(f"Missing T1 (MNI): {t1_path}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing aparc+aseg (MNI): {seg_path}")
        if not fill_path.exists():
            raise FileNotFoundError(f"Missing filled (MNI): {fill_path}")

        img = nib.load(str(t1_path))
        vol = img.get_fdata().astype(np.float32)

        seg_arr = nib.load(str(seg_path)).get_fdata().astype(np.int32)
        fill_arr = nib.load(str(fill_path)).get_fdata().astype(np.float32)

        vol = robust_normalize(vol)
        seg9 = map_labels(seg_arr, fill_arr)

        data = {"image": vol[None], "label": seg9[None]}  # [1,D,H,W]
        if self.transforms is not None:
            data = self.transforms(data)

        vol_t = torch.as_tensor(data["image"], dtype=torch.float32)
        seg_t = torch.as_tensor(data["label"], dtype=torch.long)

        vol_t = pad_vol_to_multiple(vol_t, self.pad_mult)
        seg_t = pad_seg_to_multiple(seg_t, self.pad_mult)

        return vol_t, seg_t.squeeze(0)  # label [D,H,W]


class PredictSegDataset(Dataset):
    def __init__(
        self,
        deriv_root: str,
        split_csv: str,
        split_name: str = "test",
        session_label: str = "01",
        space: str = "MNI152",
        pad_mult: int = 16,
    ):
        super().__init__()
        self.deriv_root = Path(deriv_root)
        self.split_csv = Path(split_csv)
        self.ses = _ses_id(session_label)
        self.space = space
        self.pad_mult = pad_mult

        self.subjects = _read_split_subjects(self.split_csv, split_name)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sub = self.subjects[idx]
        t1_path = _t1_mni_path(self.deriv_root, sub, self.ses, self.space)
        if not t1_path.exists():
            raise FileNotFoundError(f"Missing T1 (MNI): {t1_path}")

        img = nib.load(str(t1_path))
        vol = img.get_fdata().astype(np.float32)
        affine = img.affine
        orig_shape = np.array(vol.shape[:3], dtype=np.int16)

        vol = robust_normalize(vol)
        vol_t = torch.from_numpy(vol[None]).float()
        vol_t = pad_vol_to_multiple(vol_t, mult=self.pad_mult)

        return vol_t, sub, self.ses, affine, orig_shape


class EvalSegDataset(Dataset):
    def __init__(
        self,
        deriv_root: str,
        split_csv: str,
        pred_root: str,
        split_name: str = "test",
        session_label: str = "01",
        space: str = "MNI152",
    ):
        super().__init__()
        self.deriv_root = Path(deriv_root)
        self.split_csv = Path(split_csv)
        self.pred_root = Path(pred_root)
        self.ses = _ses_id(session_label)
        self.space = space

        self.subjects = _read_split_subjects(self.split_csv, split_name)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sub = self.subjects[idx]

        gt_path = _aparc_aseg_mni_path(self.deriv_root, sub, self.ses, self.space)
        fill_path = _filled_mni_path(self.deriv_root, sub, self.ses, self.space)
        pred_path = _pred_seg9_path(self.pred_root, sub, self.ses, self.space)

        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT aparc+aseg (MNI): {gt_path}")
        if not fill_path.exists():
            raise FileNotFoundError(f"Missing filled (MNI): {fill_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        gt_arr = nib.load(str(gt_path)).get_fdata().astype(np.int32)
        fill_arr = nib.load(str(fill_path)).get_fdata().astype(np.float32)
        pred_arr = nib.load(str(pred_path)).get_fdata().astype(np.int32)

        gt9 = map_labels(gt_arr, fill_arr)

        D, H, W = gt9.shape
        pred_arr = pred_arr[:D, :H, :W]  # crop if padded

        return gt9, pred_arr, sub, self.ses
