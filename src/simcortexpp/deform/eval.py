#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import json
import logging
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance

log = logging.getLogger(__name__)

SURFACE_NAMES = ["lh_pial", "lh_white", "rh_pial", "rh_white"]
_SURF_MAP = {
    "lh_pial":  ("L", "pial"),
    "lh_white": ("L", "white"),
    "rh_pial":  ("R", "pial"),
    "rh_white": ("R", "white"),
}

# Optional deps
try:
    import pymeshlab as pyml
    HAS_PYMESHLAB = True
except Exception:
    HAS_PYMESHLAB = False

try:
    from trimesh.collision import CollisionManager
    _ = CollisionManager()
    HAS_FCL = True
except Exception:
    HAS_FCL = False


def _ses(session_label: str) -> str:
    return f"ses-{session_label}"


def gt_surface_path(preproc_root: str, subj: str, session_label: str, space: str, surf_name: str) -> str:
    ses = _ses(session_label)
    hemi, tissue = _SURF_MAP[surf_name]
    return os.path.join(
        preproc_root, subj, ses, "surfaces",
        f"{subj}_{ses}_space-{space}_hemi-{hemi}_{tissue}.surf.ply"
    )


def pred_surface_path(pred_root: str, subj: str, session_label: str, space: str, pred_desc: str, surf_name: str) -> str:
    ses = _ses(session_label)
    hemi, tissue = _SURF_MAP[surf_name]
    return os.path.join(
        pred_root, subj, ses, "surfaces",
        f"{subj}_{ses}_space-{space}_desc-{pred_desc}_hemi-{hemi}_{tissue}.surf.ply"
    )


def _safe_load_trimesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        geoms = [g for g in m.geometry.values()]
        if len(geoms) == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        m = trimesh.util.concatenate(geoms)
    return m


def load_mesh_to_p3d(path: str, device: torch.device) -> Tuple[Meshes, trimesh.Trimesh]:
    m = _safe_load_trimesh(path)
    verts = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces]), m


def compute_chamfer_mse(mesh_p: Meshes, mesh_g: Meshes, n_pts: int) -> float:
    p_pts = sample_points_from_meshes(mesh_p, num_samples=n_pts)
    g_pts = sample_points_from_meshes(mesh_g, num_samples=n_pts)
    loss, _ = chamfer_distance(p_pts, g_pts)  # mean squared distance
    return float(loss.item())


_PointFaceDistanceOP = _PointFaceDistance.apply


def point_to_mesh_dist(pcls: Pointclouds, mesh: Meshes) -> torch.Tensor:
    pts = pcls.points_packed()
    first_idx = pcls.cloud_to_packed_first_idx()
    max_pts = pcls.num_points_per_cloud().max().item()

    tris = mesh.verts_packed()[mesh.faces_packed()]  # (F,3,3)
    tri_first = mesh.mesh_to_faces_packed_first_idx()

    d2 = _PointFaceDistanceOP(pts, first_idx, tris, tri_first, max_pts)
    return d2.sqrt()


def compute_assd_hd_exact(mesh_p: Meshes, mesh_g: Meshes, n_pts: int) -> Tuple[float, float]:
    p_pts = sample_points_from_meshes(mesh_p, num_samples=n_pts)
    g_pts = sample_points_from_meshes(mesh_g, num_samples=n_pts)

    pcl_p = Pointclouds(p_pts)
    pcl_g = Pointclouds(g_pts)

    d_p2g = point_to_mesh_dist(pcl_p, mesh_g)
    d_g2p = point_to_mesh_dist(pcl_g, mesh_p)

    assd = float((d_p2g.mean().item() + d_g2p.mean().item()) / 2.0)
    hd = float(max(d_p2g.max().item(), d_g2p.max().item()))
    return assd, hd


def compute_sif(tri_mesh: trimesh.Trimesh) -> float:
    if not HAS_PYMESHLAB or tri_mesh.faces is None or len(tri_mesh.faces) == 0:
        return float("nan")

    v = np.asarray(tri_mesh.vertices, dtype=np.float64)
    f = np.asarray(tri_mesh.faces, dtype=np.int32)

    ms = pyml.MeshSet()
    ms.add_mesh(pyml.Mesh(vertex_matrix=v, face_matrix=f), "m")
    orig = ms.current_mesh().face_number()
    if orig == 0:
        return float("nan")

    ms.apply_filter("compute_selection_by_self_intersections_per_face")
    ms.apply_filter("meshing_remove_selected_faces")
    new = ms.current_mesh().face_number()
    return float((orig - new) / orig * 100.0)


def compute_collisions(path_a: str, path_b: str) -> Dict[str, Any]:
    m1 = _safe_load_trimesh(path_a)
    m2 = _safe_load_trimesh(path_b)
    tot_a, tot_b = len(m1.faces), len(m2.faces)

    if not HAS_FCL:
        return {"total_faces": (tot_a, tot_b), "intersecting_faces": (np.nan, np.nan), "num_intersections": np.nan}

    cm = CollisionManager()
    cm.add_object("A", m1)
    cm.add_object("B", m2)

    is_col, contacts = cm.in_collision_internal(return_names=False, return_data=True)
    if (not is_col) or (contacts is None) or (len(contacts) == 0):
        return {"total_faces": (tot_a, tot_b), "intersecting_faces": (0, 0), "num_intersections": 0}

    faces_a = set([c.index("A") for c in contacts])
    faces_b = set([c.index("B") for c in contacts])

    return {
        "total_faces": (tot_a, tot_b),
        "intersecting_faces": (len(faces_a), len(faces_b)),
        "num_intersections": int(len(contacts)),
    }


def _parse_pair(x) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    if x is None:
        return (0, 0)
    s = str(x).strip()
    if s.lower() == "nan":
        return (0, 0)
    s = s.strip("()")
    a, b = s.split(",")
    return int(a), int(b)


def enhance_collision_metrics(collision_xlsx_path: str, out_dir: str) -> None:
    if not os.path.exists(collision_xlsx_path):
        return

    df = pd.read_excel(collision_xlsx_path)
    if df.empty or ("subject" not in df.columns):
        return

    collision_keys = ["pial_lr", "white_lr", "white_pial_left", "white_pial_right"]
    out_cols = {
        "pial_lr":          ("pial_LR",          "LH", "RH"),
        "white_lr":         ("white_LR",         "LH", "RH"),
        "white_pial_left":  ("white-pial_LH",    "white_LH", "pial_LH"),
        "white_pial_right": ("white-pial_RH",    "white_RH", "pial_RH"),
    }

    enhanced = {"subject": df["subject"], "dataset": df.get("dataset", pd.Series([""] * len(df)))}

    for key in collision_keys:
        if f"{key}_total_faces" not in df.columns:
            continue
        if f"{key}_intersecting_faces" not in df.columns:
            continue
        if f"{key}_num_intersections" not in df.columns:
            continue

        base, Aname, Bname = out_cols[key]

        totals = df[f"{key}_total_faces"].map(_parse_pair)
        inters = df[f"{key}_intersecting_faces"].map(_parse_pair)
        interN = df[f"{key}_num_intersections"]

        totA = totals.map(lambda t: t[0]).replace(0, np.nan)
        totB = totals.map(lambda t: t[1]).replace(0, np.nan)
        intA = inters.map(lambda t: t[0])
        intB = inters.map(lambda t: t[1])

        enhanced[f"{base}__pct_faces_{Aname}"] = (intA / totA * 100.0)
        enhanced[f"{base}__pct_faces_{Bname}"] = (intB / totB * 100.0)
        enhanced[f"{base}__density_{Aname}"] = (interN / totA)
        enhanced[f"{base}__density_{Bname}"] = (interN / totB)

    df_enh = pd.DataFrame(enhanced)
    enh_path = os.path.join(out_dir, "collision_metrics_enhanced.xlsx")
    df_enh.to_excel(enh_path, index=False)

    summary = df_enh.drop(columns=[c for c in ["subject", "dataset"] if c in df_enh.columns]).agg(["mean", "std"]).T.round(6)
    sum_path = os.path.join(out_dir, "collision_summary.xlsx")
    summary.to_excel(sum_path)

    log.info("Wrote: %s", enh_path)
    log.info("Wrote: %s", sum_path)


@hydra.main(version_base=None, config_path="pkg://simcortexpp.configs.deform", config_name="eval")
def main(cfg: DictConfig) -> None:
    level = getattr(logging, str(getattr(cfg.eval, "log_level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if cfg.user_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.user_config))

    device = torch.device(str(cfg.eval.device) if torch.cuda.is_available() else "cpu")
    out_dir = str(cfg.outputs.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pred_desc = str(getattr(cfg.eval, "pred_desc", "deform"))

    split_file = str(cfg.dataset.split_file)
    split_name = str(cfg.dataset.split_name)
    session_label = str(getattr(cfg.dataset, "session_label", "01"))
    space = str(getattr(cfg.dataset, "space", "MNI152"))

    df = pd.read_csv(split_file)
    df = df[df["split"] == split_name]
    if len(df) == 0:
        raise RuntimeError(f"No subjects found for split='{split_name}' in {split_file}")

    metrics_list = []
    collisions_list = []

    collision_pairs = [
        ("pial_lr",         "lh_pial",  "rh_pial"),
        ("white_lr",        "lh_white", "rh_white"),
        ("white_pial_left", "lh_white", "lh_pial"),
        ("white_pial_right","rh_white", "rh_pial"),
    ]

    for ds_key, ds_df in df.groupby("dataset"):
        if ds_key not in cfg.dataset.roots:
            raise KeyError(f"Missing dataset key in dataset.roots: {ds_key}")
        if ds_key not in cfg.outputs.pred_roots:
            raise KeyError(f"Missing dataset key in outputs.pred_roots: {ds_key}")

        preproc_root = str(cfg.dataset.roots[ds_key])
        pred_root = str(cfg.outputs.pred_roots[ds_key])

        subjects = ds_df["subject"].astype(str).tolist()
        log.info("[%s] evaluating subjects=%d", ds_key, len(subjects))

        for subj in tqdm(subjects, desc=f"Eval {ds_key}", leave=False):
            row_m = {"dataset": ds_key, "subject": subj}
            row_c = {"dataset": ds_key, "subject": subj}

            # surface metrics
            have_all = True
            pred_paths = {}
            for surf in SURFACE_NAMES:
                p = pred_surface_path(pred_root, subj, session_label, space, pred_desc, surf)
                g = gt_surface_path(preproc_root, subj, session_label, space, surf)
                if not (os.path.exists(p) and os.path.exists(g)):
                    have_all = False
                    break
                pred_paths[surf] = p

                mp_p3d, mp_tri = load_mesh_to_p3d(p, device)
                mg_p3d, _      = load_mesh_to_p3d(g, device)

                ch_mse = compute_chamfer_mse(mp_p3d, mg_p3d, int(cfg.eval.n_chamfer))
                assd, hd = compute_assd_hd_exact(mp_p3d, mg_p3d, int(cfg.eval.n_assd_hd))
                sif = compute_sif(mp_tri)

                row_m[f"{surf}_ChamferMSE_mm2"] = ch_mse
                row_m[f"{surf}_ChamferRMSE_mm"] = math.sqrt(ch_mse)
                row_m[f"{surf}_ASSD_mm"] = assd
                row_m[f"{surf}_HD_mm"] = hd
                row_m[f"{surf}_SIF_pct"] = sif

            if not have_all:
                continue

            metrics_list.append(row_m)

            # collisions (pred vs pred)
            for key, s1, s2 in collision_pairs:
                info = compute_collisions(pred_paths[s1], pred_paths[s2])
                row_c[f"{key}_num_intersections"] = info["num_intersections"]
                row_c[f"{key}_intersecting_faces"] = str(info["intersecting_faces"])
                row_c[f"{key}_total_faces"] = str(info["total_faces"])

            collisions_list.append(row_c)

    if len(metrics_list) == 0:
        log.error("No subjects had complete pred+gt surfaces. Check paths and pred_desc.")
        return

    df_m = pd.DataFrame(metrics_list)
    df_c = pd.DataFrame(collisions_list)

    path_m = os.path.join(out_dir, "surface_metrics.xlsx")
    path_c = os.path.join(out_dir, "collision_metrics.xlsx")

    df_m.to_excel(path_m, index=False)
    df_c.to_excel(path_c, index=False)

    log.info("Wrote: %s", path_m)
    log.info("Wrote: %s", path_c)

    enhance_collision_metrics(path_c, out_dir)

    log.info("Done. HAS_FCL=%s, HAS_PYMESHLAB=%s", HAS_FCL, HAS_PYMESHLAB)


if __name__ == "__main__":
    main()