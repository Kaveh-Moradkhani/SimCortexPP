from __future__ import annotations

import os
import time
import json
import logging
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import trimesh

from simcortexpp.deform.data.dataloader import CSRDeformDataset, collate_csr_deform
from simcortexpp.deform.utils.coords import voxel_to_world
from simcortexpp.deform.models.surfdeform import SurfDeform

log = logging.getLogger(__name__)

_SURF_MAP = {
    "lh_pial": ("L", "pial"),
    "lh_white": ("L", "white"),
    "rh_pial": ("R", "pial"),
    "rh_white": ("R", "white"),
}


def _ses(session_label: str) -> str:
    return f"ses-{session_label}"


def ensure_derivative_description(out_root: str, name: str = "scpp-deform"):
    p = os.path.join(out_root, "dataset_description.json")
    if os.path.isfile(p):
        return
    os.makedirs(out_root, exist_ok=True)
    desc = {
        "Name": name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "SimCortexPP", "Description": "Surface deformation stage"}],
    }
    with open(p, "w") as f:
        json.dump(desc, f, indent=2)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, strict: bool = True):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
        sd = sd.get("state_dict", sd.get("model", sd))

    # strip DDP prefix if any
    sd = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }

    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(sd, strict=strict)
    log.info("Loaded checkpoint: %s (strict=%s)", ckpt_path, strict)


def build_unified_init(batch: Dict, device: torch.device, surface_names: List[str]):
    B = len(batch["subject"])

    unified_list = []
    per_counts = []
    faces_per_subj = []

    affines = batch["affine"].to(device)     # [B,4,4]
    shifts  = batch["shift_ijk"].to(device)  # [B,3]

    for i in range(B):
        verts_cat = []
        counts_i = []
        faces_i = []

        for s in surface_names:
            v = batch["init_verts_vox"][i][s].to(device)      # [Ni,3] voxel in cropped/padded space
            f = batch["init_faces"][i][s].to(device).long()   # [Fi,3]
            verts_cat.append(v)
            counts_i.append(int(v.shape[0]))
            faces_i.append(f.detach().cpu().numpy().astype(np.int64))

        merged = torch.cat(verts_cat, dim=0)
        unified_list.append(merged)
        per_counts.append(counts_i)
        faces_per_subj.append(faces_i)

    lengths = torch.tensor([v.shape[0] for v in unified_list], device=device, dtype=torch.long)
    padded = pad_sequence(unified_list, batch_first=True).to(device)  # [B,Nmax,3]
    return padded, lengths, per_counts, faces_per_subj, affines, shifts


def out_surface_path(out_root: str, subj: str, session_label: str, space: str, surf_name: str) -> str:
    ses = _ses(session_label)
    hemi, surf = _SURF_MAP[surf_name]
    return os.path.join(
        out_root, subj, ses, "surfaces",
        f"{subj}_{ses}_space-{space}_desc-deform_hemi-{hemi}_{surf}.surf.ply"
    )


@hydra.main(version_base=None, config_path="pkg://simcortexpp.configs.deform", config_name="inference")
def main(cfg: DictConfig):
    level = getattr(logging, str(getattr(cfg.inference, "log_level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if cfg.user_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.user_config))

    surface_names = list(cfg.dataset.surface_name)

    # add_prob_grad forced if c_in==3
    add_prob_grad = bool(getattr(cfg.dataset, "add_prob_grad", False))
    if int(cfg.model.c_in) == 3:
        add_prob_grad = True

    device_str = str(getattr(cfg.inference, "device", "cuda:0"))
    device = torch.device(device_str if (("cuda" not in device_str) or torch.cuda.is_available()) else "cpu")

    split_file = str(cfg.dataset.split_file)
    split_name = str(cfg.dataset.split_name)
    session_label = str(getattr(cfg.dataset, "session_label", "01"))
    space = str(getattr(cfg.dataset, "space", "MNI152"))

    df = pd.read_csv(split_file)
    df = df[df["split"] == split_name]
    if len(df) == 0:
        raise RuntimeError(f"No subjects found for split_name='{split_name}' in {split_file}")

    # model
    model = SurfDeform(
        C_hid=cfg.model.c_hid,
        C_in=int(cfg.model.c_in),
        inshape=list(cfg.model.inshape),
        sigma=float(cfg.model.sigma),
        device=device,
        geom_ratio=float(getattr(cfg.model, "geom_ratio", 0.5)),
        geom_depth=int(getattr(cfg.model, "geom_depth", 6)),
        gn_groups=int(getattr(cfg.model, "gn_groups", 8)),
        gate_init=float(getattr(cfg.model, "gate_init", -3.0)),
    ).to(device)

    load_checkpoint(model, str(cfg.model.ckpt_path), strict=bool(getattr(cfg.model, "strict_load", True)))
    model.eval()

    overwrite = bool(getattr(cfg.inference, "overwrite", False))
    bs = int(getattr(cfg.inference, "batch_size", 1))
    nw = int(getattr(cfg.inference, "num_workers", 2))

    # per-dataset inference
    times = []
    for ds_key, ds_df in df.groupby("dataset"):
        if ds_key not in cfg.dataset.roots or ds_key not in cfg.dataset.initsurf_roots:
            raise KeyError(f"Missing dataset key in config roots: {ds_key}")

        preproc_root = str(cfg.dataset.roots[ds_key])
        initsurf_root = str(cfg.dataset.initsurf_roots[ds_key])
        out_root = str(cfg.outputs.out_roots[ds_key])

        ensure_derivative_description(out_root)
        subjects = ds_df["subject"].astype(str).tolist()

        ds = CSRDeformDataset(
            preproc_root=preproc_root,
            initsurf_root=initsurf_root,
            subjects=subjects,
            session_label=session_label,
            space=space,
            surface_names=surface_names,
            inshape_dhw=list(cfg.model.inshape),
            prob_clip_min=float(cfg.dataset.prob_clip_min),
            prob_clip_max=float(cfg.dataset.prob_clip_max),
            prob_gamma=float(cfg.dataset.prob_gamma),
            add_prob_grad=add_prob_grad,
            aug=False,
        )

        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            collate_fn=collate_csr_deform,
        )

        log.info("[%s] subjects=%d | out_root=%s", ds_key, len(ds), out_root)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Infer {ds_key}", leave=False):
                vol = batch["vol"].to(device)  # [B,C,D,H,W]
                B = vol.shape[0]

                padded_init, lengths, per_counts, faces_per_subj, affines, shifts = build_unified_init(
                    batch, device, surface_names
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                pred_all = model(padded_init, vol, int(cfg.model.n_steps))  # [B,Nmax,3]

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                times.extend([(t1 - t0) / max(B, 1)] * B)

                for i in range(B):
                    subj = str(batch["subject"][i])
                    A = affines[i]
                    sh = shifts[i]  # [3]

                    pred_unified = pred_all[i, : int(lengths[i].item())]
                    splits = torch.split(pred_unified, per_counts[i], dim=0)

                    for j, surf in enumerate(surface_names):
                        out_path = out_surface_path(out_root, subj, session_label, space, surf)
                        if (not overwrite) and os.path.isfile(out_path):
                            continue

                        os.makedirs(os.path.dirname(out_path), exist_ok=True)

                        v_vox_cp = splits[j]       # cropped/padded voxel space
                        v_vox_orig = v_vox_cp - sh # undo shift (back to original voxel space)
                        v_mm = voxel_to_world(v_vox_orig, A).detach().cpu().numpy().astype(np.float32)

                        f = faces_per_subj[i][j]
                        trimesh.Trimesh(vertices=v_mm, faces=f, process=False).export(out_path)

    if times:
        log.info("Avg inference time/subject: %.4fs", float(sum(times) / len(times)))
    log.info("Done.")


if __name__ == "__main__":
    main()