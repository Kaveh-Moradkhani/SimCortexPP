
import os
import gc
import math
import logging
from datetime import timedelta
from contextlib import nullcontext
from typing import Dict, List, Tuple

import hydra
import torch
import torch.nn.functional as F
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance 

from simcortexpp.deform.data.dataloader import CSRDeformDataset, collate_csr_deform
from simcortexpp.deform.utils.coords import voxel_to_world
from simcortexpp.deform.models.surfdeform import SurfDeform 
 

import trimesh

try:
    from trimesh.collision import CollisionManager
    _ = CollisionManager()
    HAS_FCL = True
except Exception:
    HAS_FCL = False


log = logging.getLogger(__name__)


def count_collisions_inmemory(
    vA_mm: torch.Tensor, fA: torch.Tensor,
    vB_mm: torch.Tensor, fB: torch.Tensor
):
    """
    vA_mm, vB_mm: (V,3) torch float in mm-space (GPU/CPU)
    fA, fB: (F,3) torch long
    Returns: (is_col: bool or None, n_contacts: int or None)
    """
    if not HAS_FCL:
        return None, None

    vA = vA_mm.detach().float().cpu().numpy()
    vB = vB_mm.detach().float().cpu().numpy()
    fA_np = fA.detach().long().cpu().numpy()
    fB_np = fB.detach().long().cpu().numpy()

    if vA.shape[0] == 0 or vB.shape[0] == 0 or fA_np.shape[0] == 0 or fB_np.shape[0] == 0:
        return False, 0

    mA = trimesh.Trimesh(vertices=vA, faces=fA_np, process=False)
    mB = trimesh.Trimesh(vertices=vB, faces=fB_np, process=False)

    cm = CollisionManager()
    cm.add_object("A", mA)
    cm.add_object("B", mB)

    is_col, contacts = cm.in_collision_internal(return_names=False, return_data=True)
    if (not is_col) or (contacts is None):
        return False, 0
    return True, int(len(contacts))


# -----------------------
# DDP helpers
# -----------------------
def setup_ddp() -> Tuple[int, int, int, bool]:
    """Return (rank, world_size, local_rank, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(hours=6),
        )
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True

    return 0, 1, 0, False


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def seed_all(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Geometry helpers
# -----------------------
def mesh_is_valid(verts: torch.Tensor, faces: torch.Tensor) -> bool:
    if verts is None or faces is None:
        return False
    if verts.ndim != 2 or faces.ndim != 2:
        return False
    if verts.shape[1] != 3 or faces.shape[1] != 3:
        return False
    if verts.numel() == 0 or faces.numel() == 0:
        return False
    if torch.isnan(verts).any() or torch.isinf(verts).any():
        return False
    f = faces.long()
    if f.min().item() < 0:
        return False
    if f.max().item() >= verts.shape[0]:
        return False
    return True


# -----------------------
# HD_p separation penalty
# -----------------------
_PointFaceDistanceOP = _PointFaceDistance.apply


def point_to_mesh_dist_p3d(points: torch.Tensor, mesh: Meshes) -> torch.Tensor:
    """
    points: (N,3) float on device
    mesh: Meshes (batch size 1)
    returns: (N,) distances in same units as verts (here mm)
    """
    pts = points
    first_idx = torch.zeros((1,), device=pts.device, dtype=torch.int64)  # batch size 1
    max_pts = int(pts.shape[0])

    tris = mesh.verts_packed()[mesh.faces_packed()]  # (F,3,3)
    tri_first = mesh.mesh_to_faces_packed_first_idx()  # (1,)

    d2 = _PointFaceDistanceOP(pts, first_idx, tris, tri_first, max_pts)  # squared
    return d2.sqrt()


def partial_hd_penalty(mesh_a: Meshes, mesh_b: Meshes, p: float, lam: float, n_pts: int):
    """
    Compute HD_p (LOW quantile) over symmetric point-to-surface distances.
    Penalty: relu(lam - HD_p)

    Returns:
      hd_p_mm: scalar tensor (mm)
      penalty: scalar tensor
    """
    pa = sample_points_from_meshes(mesh_a, num_samples=n_pts).squeeze(0)
    pb = sample_points_from_meshes(mesh_b, num_samples=n_pts).squeeze(0)

    da = point_to_mesh_dist_p3d(pa, mesh_b)
    db = point_to_mesh_dist_p3d(pb, mesh_a)

    d_all = torch.cat([da, db], dim=0)  # (2n,)
    hd_p_mm = torch.quantile(d_all, q=float(p))

    lam_t = hd_p_mm.new_tensor(float(lam))
    penalty = F.relu(lam_t - hd_p_mm)
    return hd_p_mm, penalty


# -----------------------
# Random affine augmentation in NDC (volume + verts)
# -----------------------
def voxel_sizes_xyz_from_affine(A: torch.Tensor) -> torch.Tensor:
    A3 = A[:3, :3]
    vsize_ijk = torch.linalg.norm(A3, dim=0).clamp(min=1e-6)
    return vsize_ijk[[2, 1, 0]]  # xyz


def ijk_to_xyz(v_ijk: torch.Tensor) -> torch.Tensor:
    return torch.stack([v_ijk[..., 2], v_ijk[..., 1], v_ijk[..., 0]], dim=-1)


def xyz_to_ijk(v_xyz: torch.Tensor) -> torch.Tensor:
    return torch.stack([v_xyz[..., 2], v_xyz[..., 1], v_xyz[..., 0]], dim=-1)


def voxel_to_ndc_xyz(v_xyz: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    den = torch.tensor([W - 1, H - 1, D - 1], device=v_xyz.device, dtype=v_xyz.dtype).clamp(min=1.0)
    return 2.0 * (v_xyz / den) - 1.0


def ndc_to_voxel_xyz(u_xyz: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    den = torch.tensor([W - 1, H - 1, D - 1], device=u_xyz.device, dtype=u_xyz.dtype).clamp(min=1.0)
    return 0.5 * (u_xyz + 1.0) * den


def random_affine_ndc_xyz(B: int, rot_deg: float, scale_range: float, trans_ndc_xyz: torch.Tensor, device, dtype):
    ang = (torch.rand(B, 3, device=device, dtype=dtype) * 2 - 1) * (rot_deg * math.pi / 180.0)
    cx, sx = torch.cos(ang[:, 0]), torch.sin(ang[:, 0])
    cy, sy = torch.cos(ang[:, 1]), torch.sin(ang[:, 1])
    cz, sz = torch.cos(ang[:, 2]), torch.sin(ang[:, 2])

    Rx = torch.stack([
        torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx),
        torch.zeros_like(cx), cx, -sx,
        torch.zeros_like(cx), sx, cx
    ], dim=-1).view(-1, 3, 3)

    Ry = torch.stack([
        cy, torch.zeros_like(cy), sy,
        torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy),
        -sy, torch.zeros_like(cy), cy
    ], dim=-1).view(-1, 3, 3)

    Rz = torch.stack([
        cz, -sz, torch.zeros_like(cz),
        sz, cz, torch.zeros_like(cz),
        torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)
    ], dim=-1).view(-1, 3, 3)

    R = Rz @ Ry @ Rx

    ds = (torch.rand(B, 1, device=device, dtype=dtype) * 2 - 1) * scale_range
    s = 1.0 + ds
    A = R * s.view(B, 1, 1)

    t = (torch.rand(B, 3, device=device, dtype=dtype) * 2 - 1) * trans_ndc_xyz
    b = t
    return A, b


def apply_aug(vol, padded_init_ijk, lengths, gt_verts_dict_list, affines, cfg, surface_names):
    prob = float(getattr(cfg.dataset, "aug_prob", 0.0))
    if prob <= 0.0:
        return vol, padded_init_ijk, gt_verts_dict_list

    B, C, D, H, W = vol.shape
    device = vol.device
    dtype = vol.dtype

    mask = (torch.rand(B, device=device) < prob)
    if mask.sum().item() == 0:
        return vol, padded_init_ijk, gt_verts_dict_list

    rot_deg = float(getattr(cfg.dataset, "aug_rot_range_deg", 0.0))
    scale_range = float(getattr(cfg.dataset, "aug_scale_range", 0.0))
    trans_mm = float(getattr(cfg.dataset, "aug_trans_range_mm", 0.0))

    trans_ndc_xyz = torch.zeros((B, 3), device=device, dtype=dtype)
    den_xyz = torch.tensor([W - 1, H - 1, D - 1], device=device, dtype=dtype).clamp(min=1.0)

    for i in range(B):
        vsize_xyz = voxel_sizes_xyz_from_affine(affines[i].to(device=device, dtype=dtype))
        trans_vox_xyz = (trans_mm / vsize_xyz)
        trans_ndc_xyz[i] = 2.0 * (trans_vox_xyz / den_xyz)

    A_fwd, b_fwd = random_affine_ndc_xyz(B, rot_deg, scale_range, trans_ndc_xyz, device, dtype)

    I = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).repeat(B, 1, 1)
    Z = torch.zeros((B, 3), device=device, dtype=dtype)
    A_fwd = torch.where(mask.view(B, 1, 1), A_fwd, I)
    b_fwd = torch.where(mask.view(B, 1), b_fwd, Z)

    A_inv = torch.linalg.inv(A_fwd)
    b_inv = -(A_inv @ b_fwd.unsqueeze(-1)).squeeze(-1)

    theta = torch.zeros((B, 3, 4), device=device, dtype=dtype)
    theta[:, :, :3] = A_inv
    theta[:, :, 3] = b_inv

    grid = F.affine_grid(theta, size=vol.size(), align_corners=True)
    vol = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)

    for i in range(B):
        if not mask[i].item():
            continue

        L = int(lengths[i].item())

        v_ijk = padded_init_ijk[i, :L]
        v_xyz = ijk_to_xyz(v_ijk)
        u = voxel_to_ndc_xyz(v_xyz, D, H, W)
        u2 = (A_fwd[i] @ u.t()).t() + b_fwd[i].view(1, 3)
        v_xyz2 = ndc_to_voxel_xyz(u2, D, H, W)
        padded_init_ijk[i, :L] = xyz_to_ijk(v_xyz2)

        gdict = gt_verts_dict_list[i]
        for s in surface_names:
            gv_ijk = gdict[s]
            gv_xyz = ijk_to_xyz(gv_ijk)
            ug = voxel_to_ndc_xyz(gv_xyz, D, H, W)
            ug2 = (A_fwd[i] @ ug.t()).t() + b_fwd[i].view(1, 3)
            gv_xyz2 = ndc_to_voxel_xyz(ug2, D, H, W)
            gdict[s] = xyz_to_ijk(gv_xyz2)
        gt_verts_dict_list[i] = gdict

    return vol, padded_init_ijk, gt_verts_dict_list


# -----------------------
# Utilities for building padded init verts
# -----------------------
def build_merged_init_and_metadata(batch, device, surface_names):
    B = len(batch["init_verts_vox"])

    per_counts_init: List[List[int]] = []
    merged_init_list: List[torch.Tensor] = []
    init_faces_list: List[Dict[str, torch.Tensor]] = []
    gt_verts_list: List[Dict[str, torch.Tensor]] = []
    gt_faces_list: List[Dict[str, torch.Tensor]] = []

    for i in range(B):
        counts = []
        v_all = []
        f_init_dict = {}
        gv_dict = {}
        gf_dict = {}

        for s in surface_names:
            v = batch["init_verts_vox"][i][s].to(device)
            f = batch["init_faces"][i][s].to(device).long()
            gv = batch["gt_verts_vox"][i][s].to(device)
            gf = batch["gt_faces"][i][s].to(device).long()

            counts.append(int(v.shape[0]))
            v_all.append(v)
            f_init_dict[s] = f
            gv_dict[s] = gv
            gf_dict[s] = gf

        per_counts_init.append(counts)
        merged_init_list.append(torch.cat(v_all, dim=0))
        init_faces_list.append(f_init_dict)
        gt_verts_list.append(gv_dict)
        gt_faces_list.append(gf_dict)

    lengths = torch.tensor([v.shape[0] for v in merged_init_list], device=device, dtype=torch.long)
    Vmax = int(lengths.max().item())

    padded_init = torch.zeros((B, Vmax, 3), device=device, dtype=merged_init_list[0].dtype)
    for i in range(B):
        padded_init[i, :lengths[i]] = merged_init_list[i]

    return lengths, padded_init, per_counts_init, init_faces_list, gt_verts_list, gt_faces_list


# -----------------------
# Main
# -----------------------
@hydra.main(version_base=None, config_path="pkg://simcortexpp.configs.deform", config_name="train")
def main(cfg: DictConfig):
    rank, world_size, local_rank, is_distributed = setup_ddp()

    level = getattr(logging, str(getattr(cfg.trainer, "log_level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if cfg.user_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.user_config))

    if rank == 0:
        log.info("world_size=%d, local_rank=%d", world_size, local_rank)
        print(OmegaConf.to_yaml(cfg))

    seed_all(int(cfg.trainer.seed))
    torch.backends.cudnn.benchmark = True

    # datasets
    surface_names = list(cfg.dataset.surface_name)
    inshape = tuple(int(x) for x in cfg.model.inshape)

    split_file = str(cfg.dataset.split_file)
    train_split = str(getattr(cfg.dataset, "train_split_name", "train"))
    val_split = str(getattr(cfg.dataset, "val_split_name", "val"))

    session_label = str(getattr(cfg.dataset, "session_label", "01"))
    space = str(getattr(cfg.dataset, "space", "MNI152"))

    df = pd.read_csv(split_file)

    # ---- Multi-dataset mode ----
    if hasattr(cfg.dataset, "roots") and hasattr(cfg.dataset, "initsurf_roots"):
        train_sets = []
        val_sets = []

        for ds_key, ds_df in df.groupby("dataset"):
            if ds_key not in cfg.dataset.roots or ds_key not in cfg.dataset.initsurf_roots:
                raise KeyError(f"Missing dataset key in config: {ds_key}")

            preproc_root = cfg.dataset.roots[ds_key]
            initsurf_root = cfg.dataset.initsurf_roots[ds_key]

            tr_subs = ds_df[ds_df["split"] == train_split]["subject"].astype(str).tolist()
            va_subs = ds_df[ds_df["split"] == val_split]["subject"].astype(str).tolist()

            if len(tr_subs) > 0:
                train_sets.append(
                    CSRDeformDataset(
                        preproc_root=preproc_root,
                        initsurf_root=initsurf_root,
                        subjects=tr_subs,
                        session_label=session_label,
                        space=space,
                        surface_names=surface_names,
                        inshape_dhw=inshape,
                        prob_clip_min=cfg.dataset.prob_clip_min,
                        prob_clip_max=cfg.dataset.prob_clip_max,
                        prob_gamma=cfg.dataset.prob_gamma,
                        add_prob_grad=bool(getattr(cfg.dataset, "add_prob_grad", False)),
                        aug=False,
                    )
                )
            if len(va_subs) > 0:
                val_sets.append(
                    CSRDeformDataset(
                        preproc_root=preproc_root,
                        initsurf_root=initsurf_root,
                        subjects=va_subs,
                        session_label=session_label,
                        space=space,
                        surface_names=surface_names,
                        inshape_dhw=inshape,
                        prob_clip_min=cfg.dataset.prob_clip_min,
                        prob_clip_max=cfg.dataset.prob_clip_max,
                        prob_gamma=cfg.dataset.prob_gamma,
                        add_prob_grad=bool(getattr(cfg.dataset, "add_prob_grad", False)),
                        aug=False,
                    )
                )

        if len(train_sets) == 0:
            raise RuntimeError("No training subjects found (multi-dataset). Check split_file and train_split_name.")
        if len(val_sets) == 0:
            raise RuntimeError("No validation subjects found (multi-dataset). Check split_file and val_split_name.")

        train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
        val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]

    # ---- Single-dataset mode ----
    else:
        preproc_root = str(getattr(cfg.dataset, "preproc_root", getattr(cfg.dataset, "path", "")))
        initsurf_root = str(getattr(cfg.dataset, "initsurf_root", getattr(cfg.dataset, "initial_surface_path", "")))

        tr_subs = df[df["split"] == train_split]["subject"].astype(str).tolist()
        va_subs = df[df["split"] == val_split]["subject"].astype(str).tolist()

        train_ds = CSRDeformDataset(
            preproc_root=preproc_root,
            initsurf_root=initsurf_root,
            subjects=tr_subs,
            session_label=session_label,
            space=space,
            surface_names=surface_names,
            inshape_dhw=inshape,
            prob_clip_min=cfg.dataset.prob_clip_min,
            prob_clip_max=cfg.dataset.prob_clip_max,
            prob_gamma=cfg.dataset.prob_gamma,
            add_prob_grad=bool(getattr(cfg.dataset, "add_prob_grad", False)),
            aug=False,
        )
        val_ds = CSRDeformDataset(
            preproc_root=preproc_root,
            initsurf_root=initsurf_root,
            subjects=va_subs,
            session_label=session_label,
            space=space,
            surface_names=surface_names,
            inshape_dhw=inshape,
            prob_clip_min=cfg.dataset.prob_clip_min,
            prob_clip_max=cfg.dataset.prob_clip_max,
            prob_gamma=cfg.dataset.prob_gamma,
            add_prob_grad=bool(getattr(cfg.dataset, "add_prob_grad", False)),
            aug=False,
        )
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.trainer.img_batch_size),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=int(cfg.trainer.num_workers),
        pin_memory=True,
        collate_fn=collate_csr_deform,
    )

    # IMPORTANT: validation loader is NOT distributed to avoid sampler padding (77 -> 78)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(cfg.trainer.img_batch_size),
        shuffle=False,
        num_workers=int(cfg.trainer.num_workers),
        pin_memory=True,
        collate_fn=collate_csr_deform,
    )

    if rank == 0:
        log.info("Loaded %d training subjects", len(train_ds))
        log.info("Loaded %d validation subjects", len(val_ds))

    # model
    model = SurfDeform(
        C_hid=cfg.model.c_hid,
        C_in=int(cfg.model.c_in),
        inshape=inshape,
        sigma=float(cfg.model.sigma),
        device=device,
        geom_ratio=float(getattr(cfg.model, "geom_ratio", 0.5)),
        geom_depth=int(getattr(cfg.model, "geom_depth", 4)),
        gn_groups=int(getattr(cfg.model, "gn_groups", 8)),
        gate_init=float(getattr(cfg.model, "gate_init", -3.0)),
    ).to(device)

    # optional init checkpoint
    init_ckpt = str(getattr(cfg.model, "init_ckpt", "") or "")
    if init_ckpt:
        if rank == 0:
            log.info("Loading init_ckpt: %s", init_ckpt)
        sd = torch.load(init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=bool(getattr(cfg.model, "init_strict", True)))
        if rank == 0:
            log.info("Init load done. missing=%d unexpected=%d", len(missing), len(unexpected))

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.trainer.scheduler_factor),
        patience=int(cfg.trainer.scheduler_patience),
        threshold=float(cfg.trainer.scheduler_threshold_mm),
        threshold_mode=str(cfg.trainer.scheduler_threshold_mode),
        cooldown=int(cfg.trainer.scheduler_cooldown),
        min_lr=float(cfg.trainer.scheduler_min_lr),
        verbose=(rank == 0),
    )

    # Logging & Config Saving

    out_root = str(getattr(cfg.outputs, 'root', getattr(cfg.outputs, 'output_dir', '')))

    tb_writer = None
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)
        tb_dir = os.path.join(out_root, "tb_logs")
        os.makedirs(tb_dir, exist_ok=True)

        log.info("TensorBoard logging to %s", tb_dir)

        resolved_conf_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        config_path = os.path.join(out_root, "config_resolved.yaml")
        with open(config_path, "w") as f:
            f.write(resolved_conf_yaml)
        log.info("Resolved config saved to %s", config_path)

        tb_writer = SummaryWriter(tb_dir)
        formatted_config = resolved_conf_yaml.replace("\n", "  \n")
        tb_writer.add_text(
            "Hyperparameters",
            f"### Training Configuration\n```yaml\n{formatted_config}\n```",
            0
        )

    # weights
    chamfer_w = float(cfg.objective.chamfer_weight)
    chamfer_scale = float(getattr(cfg.objective, "chamfer_scale", 1.0))
    edge_w_base = float(cfg.objective.edge_loss_weight)
    normal_w_base = float(cfg.objective.normal_weight)
    reg_warmup = int(getattr(cfg.objective, "reg_warmup_epochs", 0))

    # HD weights/settings (white vs pial per hemisphere)
    hd_w_base = float(getattr(cfg.objective, "hd_weight", 0.0))
    hd_p = float(getattr(cfg.objective, "hd_p", 0.05))
    hd_lam = float(getattr(cfg.objective, "hd_lambda_mm", 0.5))
    Phd = int(getattr(cfg.objective, "hd_points", 30000))

    # Pial-LR separation (lh_pial vs rh_pial)
    pial_lr_w_base = float(getattr(cfg.objective, "pial_lr_hd_weight", 0.0))
    pial_lr_p = float(getattr(cfg.objective, "pial_lr_hd_p", hd_p))
    pial_lr_lam = float(getattr(cfg.objective, "pial_lr_hd_lambda_mm", hd_lam))
    pial_lr_pts = int(getattr(cfg.objective, "pial_lr_hd_points", Phd))

    # train setup
    num_epochs = int(cfg.trainer.num_epochs)
    accum_steps = max(1, int(cfg.trainer.grad_accum_steps))
    grad_clip = float(cfg.trainer.grad_clip_norm)
    mesh_chunk = max(1, int(cfg.trainer.mesh_chunk))
    Ptrain = int(cfg.trainer.points_per_image)
    Pval = int(cfg.trainer.val_points_per_image)
    val_interval = max(1, int(cfg.trainer.validation_interval))
    col_interval = int(getattr(cfg.trainer, "collision_interval", 50))

    best_val = float("inf")
    no_improve = 0
    early_patience = int(getattr(cfg.trainer, "early_stop_patience", 0))
    early_delta = float(getattr(cfg.trainer, "early_stop_min_delta_mm", 0.0))

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(1, num_epochs + 1):
        clean_gpu()

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            log.info("Epoch %d/%d", epoch, num_epochs)

        # warmup for regularizers (including HD)
        t = 1.0
        if reg_warmup > 0:
            t = min(1.0, epoch / float(reg_warmup))
        edge_w = edge_w_base * t
        normal_w = normal_w_base * t
        hd_w_eff = hd_w_base * t
        pial_lr_w_eff = pial_lr_w_base * t

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # epoch stats (sum over meshes)
        csq_sum = 0.0
        edge_sum = 0.0
        normal_sum = 0.0
        total_sum = 0.0
        mesh_count = 0.0

        # HD stats (sum over pairs)
        hd_pen_sum = 0.0
        hdp_sum = 0.0
        hd_count = 0.0

        # Pial-LR stats (sum over pairs)
        pial_lr_pen_sum = 0.0
        pial_lr_hdp_sum = 0.0
        pial_lr_count = 0.0

        surf_stats = {s: {"csq": 0.0, "count": 0.0} for s in surface_names}

        accum_counter = 0
        did_backward = False

        for batch in tqdm(train_loader, disable=(rank != 0), desc=f"Train {epoch} [r{rank}]"):
            vol = batch["vol"].to(device)
            aff = batch["affine"].to(device)
            shift = batch["shift_ijk"].to(device)

            B, _, D, H, W = vol.shape

            lengths, padded_init, per_counts_init, init_faces_list, gt_verts_list, gt_faces_list = \
                build_merged_init_and_metadata(batch, device, surface_names)

            # augmentation
            vol, padded_init, gt_verts_list = apply_aug(
                vol=vol,
                padded_init_ijk=padded_init,
                lengths=lengths,
                gt_verts_dict_list=gt_verts_list,
                affines=aff,
                cfg=cfg,
                surface_names=surface_names,
            )

            # forward
            pred_vox = model(padded_init, vol, int(cfg.model.n_steps))

            # Build mesh lists in WORLD(mm) for Chamfer/edge/normal
            pred_verts_mm, pred_faces = [], []
            gt_verts_mm, gt_faces = [], []
            surf_of_mesh = []

            # store pred meshes per sample for HD (white/pial and pialLR)
            pred_mesh_mm_per_sample = [dict() for _ in range(B)]

            for i in range(B):
                pred_i = pred_vox[i, :lengths[i]]
                splits = torch.split(pred_i, per_counts_init[i], dim=0)

                A = aff[i]
                sh = shift[i].view(1, 3)

                for j, s in enumerate(surface_names):
                    pv = splits[j]
                    gv = gt_verts_list[i][s]

                    f = init_faces_list[i][s]
                    gf = gt_faces_list[i][s]

                    pv_mm = voxel_to_world(pv - sh, A)
                    gv_mm = voxel_to_world(gv - sh, A)

                    # store pred mesh for separation losses if pred is valid
                    if mesh_is_valid(pv_mm, f):
                        pred_mesh_mm_per_sample[i][s] = (pv_mm, f)

                    # for chamfer/regularizers, need both pred and gt valid
                    if (not mesh_is_valid(pv_mm, f)) or (not mesh_is_valid(gv_mm, gf)):
                        continue

                    pred_verts_mm.append(pv_mm)
                    pred_faces.append(f)
                    gt_verts_mm.append(gv_mm)
                    gt_faces.append(gf)
                    surf_of_mesh.append(s)

            M = len(pred_verts_mm)
            if M == 0:
                zero = sum(p.sum() * 0.0 for p in (model.module.parameters() if hasattr(model, "module") else model.parameters()))
                (zero / accum_steps).backward()
                did_backward = True
                accum_counter += 1
                continue

            # -----------------------
            # HD separation penalty: white vs pial within hemisphere
            # -----------------------
            loss_hd = torch.zeros((), device=device)
            pair_count = 0
            hdp_sum_batch = 0.0

            if hd_w_eff > 0.0:
                for i in range(B):
                    md = pred_mesh_mm_per_sample[i]

                    if ("lh_white" in md) and ("lh_pial" in md):
                        vw, fw = md["lh_white"]
                        vp, fp = md["lh_pial"]
                        mw = Meshes(verts=[vw], faces=[fw])
                        mp = Meshes(verts=[vp], faces=[fp])
                        hdp, pen = partial_hd_penalty(mw, mp, p=hd_p, lam=hd_lam, n_pts=Phd)
                        loss_hd = loss_hd + pen
                        hdp_sum_batch += float(hdp.detach().item())
                        pair_count += 1

                    if ("rh_white" in md) and ("rh_pial" in md):
                        vw, fw = md["rh_white"]
                        vp, fp = md["rh_pial"]
                        mw = Meshes(verts=[vw], faces=[fw])
                        mp = Meshes(verts=[vp], faces=[fp])
                        hdp, pen = partial_hd_penalty(mw, mp, p=hd_p, lam=hd_lam, n_pts=Phd)
                        loss_hd = loss_hd + pen
                        hdp_sum_batch += float(hdp.detach().item())
                        pair_count += 1

                if pair_count > 0:
                    loss_hd = loss_hd / float(pair_count)

            # -----------------------
            # Pial-LR separation: lh_pial vs rh_pial
            # -----------------------
            loss_pial_lr = torch.zeros((), device=device)
            pial_lr_pair_count = 0
            pial_lr_hdp_sum_batch = 0.0

            if pial_lr_w_eff > 0.0:
                for i in range(B):
                    md = pred_mesh_mm_per_sample[i]
                    if ("lh_pial" in md) and ("rh_pial" in md):
                        vl, fl = md["lh_pial"]
                        vr, fr = md["rh_pial"]
                        ml = Meshes(verts=[vl], faces=[fl])
                        mr = Meshes(verts=[vr], faces=[fr])

                        hdp_lr, pen_lr = partial_hd_penalty(
                            ml, mr, p=pial_lr_p, lam=pial_lr_lam, n_pts=pial_lr_pts
                        )
                        loss_pial_lr = loss_pial_lr + pen_lr
                        pial_lr_hdp_sum_batch += float(hdp_lr.detach().item())
                        pial_lr_pair_count += 1

                if pial_lr_pair_count > 0:
                    loss_pial_lr = loss_pial_lr / float(pial_lr_pair_count)

            # -----------------------
            # Chamfer/edge/normal losses (chunked)
            # -----------------------
            loss_csq = torch.zeros((), device=device)
            loss_edge = torch.zeros((), device=device)
            loss_norm = torch.zeros((), device=device)

            csq_det_sum = 0.0

            for start in range(0, M, mesh_chunk):
                end = min(M, start + mesh_chunk)

                mp = Meshes(verts=pred_verts_mm[start:end], faces=pred_faces[start:end])
                mg = Meshes(verts=gt_verts_mm[start:end], faces=gt_faces[start:end])

                pp = sample_points_from_meshes(mp, num_samples=Ptrain)
                pg = sample_points_from_meshes(mg, num_samples=Ptrain)

                csq_per, _ = chamfer_distance(pp, pg, batch_reduction=None)
                e = mesh_edge_loss(mp)
                n = mesh_normal_consistency(mp)

                mchunk = (end - start)

                loss_csq = loss_csq + csq_per.mean() * mchunk
                loss_edge = loss_edge + e * mchunk
                loss_norm = loss_norm + n * mchunk

                csq_det_sum += float(csq_per.detach().sum().item())
                for k in range(mchunk):
                    ss = surf_of_mesh[start + k]
                    surf_stats[ss]["csq"] += float(csq_per[k].detach().item())
                    surf_stats[ss]["count"] += 1.0

            loss_csq = loss_csq / M
            loss_edge = loss_edge / M
            loss_norm = loss_norm / M

            # total loss
            total_loss = (
                chamfer_w * (chamfer_scale * loss_csq)
                + edge_w * loss_edge
                + normal_w * loss_norm
                + hd_w_eff * loss_hd
                + pial_lr_w_eff * loss_pial_lr
            )

            accum_counter += 1
            loss_to_back = total_loss / accum_steps

            sync_ctx = nullcontext()
            if is_distributed and hasattr(model, "no_sync") and (accum_counter % accum_steps) != 0:
                sync_ctx = model.no_sync()

            with sync_ctx:
                loss_to_back.backward()
            did_backward = True

            if (accum_counter % accum_steps) == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # stats
            csq_sum += csq_det_sum
            edge_sum += float((loss_edge.detach() * M).item())
            normal_sum += float((loss_norm.detach() * M).item())
            total_sum += float((total_loss.detach() * M).item())
            mesh_count += float(M)

            if pair_count > 0:
                hd_pen_sum += float((loss_hd.detach() * pair_count).item())
                hdp_sum += float(hdp_sum_batch)
                hd_count += float(pair_count)

            if pial_lr_pair_count > 0:
                pial_lr_pen_sum += float((loss_pial_lr.detach() * pial_lr_pair_count).item())
                pial_lr_hdp_sum += float(pial_lr_hdp_sum_batch)
                pial_lr_count += float(pial_lr_pair_count)

        # last partial step
        if did_backward and (accum_counter % accum_steps) != 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # reduce train stats
        if is_distributed:
            tstat = torch.tensor(
                [csq_sum, edge_sum, normal_sum, total_sum, mesh_count,
                 hd_pen_sum, hdp_sum, hd_count,
                 pial_lr_pen_sum, pial_lr_hdp_sum, pial_lr_count],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(tstat, op=dist.ReduceOp.SUM)
            (csq_sum, edge_sum, normal_sum, total_sum, mesh_count,
             hd_pen_sum, hdp_sum, hd_count,
             pial_lr_pen_sum, pial_lr_hdp_sum, pial_lr_count) = tstat.tolist()

            surf_tensor = torch.zeros((len(surface_names), 2), device=device, dtype=torch.float64)
            for i, s in enumerate(surface_names):
                surf_tensor[i, 0] = surf_stats[s]["csq"]
                surf_tensor[i, 1] = surf_stats[s]["count"]
            dist.all_reduce(surf_tensor, op=dist.ReduceOp.SUM)
            surf_global = {
                s: {"csq": surf_tensor[i, 0].item(), "count": surf_tensor[i, 1].item()}
                for i, s in enumerate(surface_names)
            }
        else:
            surf_global = surf_stats

        # log train
        if rank == 0 and mesh_count > 0:
            csq_mean = csq_sum / mesh_count
            rmse_mm_train = math.sqrt(max(csq_mean, 0.0))
            edge_mean = edge_sum / mesh_count
            norm_mean = normal_sum / mesh_count
            total_mean = total_sum / mesh_count

            if hd_count > 0:
                hd_pen_mean = hd_pen_sum / hd_count
                hdp_mean_mm = hdp_sum / hd_count
            else:
                hd_pen_mean = 0.0
                hdp_mean_mm = 0.0

            if pial_lr_count > 0:
                pial_lr_pen_mean = pial_lr_pen_sum / pial_lr_count
                pial_lr_hdp_mean = pial_lr_hdp_sum / pial_lr_count
            else:
                pial_lr_pen_mean = 0.0
                pial_lr_hdp_mean = 0.0

            surf_str = ", ".join(
                f"{s}={math.sqrt(max(surf_global[s]['csq']/max(surf_global[s]['count'],1.0),0.0)):.4f}mm"
                for s in surface_names
            )

            log.info(
                "Epoch %d [Train] | ChamferRMSE=%.4f mm | Edge=%.6f | Normal=%.6f | "
                "HDpen=%.6f | HDp=%.4f mm | wHD=%.4f | "
                "PialLRpen=%.6f | PialLRp=%.4f mm | wPialLR=%.4f | "
                "Total=%.6f | Surfaces: %s",
                epoch,
                rmse_mm_train, edge_mean, norm_mean,
                hd_pen_mean, hdp_mean_mm, hd_w_eff,
                pial_lr_pen_mean, pial_lr_hdp_mean, pial_lr_w_eff,
                total_mean, surf_str
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/rmse_mm", rmse_mm_train, epoch)
                tb_writer.add_scalar("train/edge", edge_mean, epoch)
                tb_writer.add_scalar("train/normal", norm_mean, epoch)
                tb_writer.add_scalar("train/total", total_mean, epoch)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

                tb_writer.add_scalar("train/hd_penalty", hd_pen_mean, epoch)
                tb_writer.add_scalar("train/hdp_mean_mm", hdp_mean_mm, epoch)
                tb_writer.add_scalar("train/hd_weight_eff", hd_w_eff, epoch)

                tb_writer.add_scalar("train/pial_lr_penalty", pial_lr_pen_mean, epoch)
                tb_writer.add_scalar("train/pial_lr_hdp_mean_mm", pial_lr_hdp_mean, epoch)
                tb_writer.add_scalar("train/pial_lr_weight_eff", pial_lr_w_eff, epoch)

        # -----------------------
        # Validation (rank0 only) + collisions
        # -----------------------
        stop_tensor = torch.tensor(0, device=device, dtype=torch.int64)

        if (epoch % val_interval) == 0:
            # Use underlying module to avoid DDP collectives in forward
            net = model.module if hasattr(model, "module") else model
            net.eval()

            do_collision_check = (epoch % col_interval == 0)

            val_csq_sum = 0.0
            val_count = 0.0
            val_surf = {s: {"csq": 0.0, "count": 0.0} for s in surface_names}

            lh_total = lh_hit = lh_contacts_sum = 0.0
            rh_total = rh_hit = rh_contacts_sum = 0.0
            lr_total = lr_hit = lr_contacts_sum = 0.0  # lh_pial vs rh_pial collisions

            if rank == 0:
                with torch.no_grad():
                    for batch in tqdm(val_loader, disable=False, desc=f"Val {epoch} [rank0]"):
                        vol = batch["vol"].to(device)
                        aff = batch["affine"].to(device)
                        shift = batch["shift_ijk"].to(device)

                        B = vol.shape[0]

                        per_counts_init = []
                        merged_init_list = []
                        for i in range(B):
                            v_all = []
                            counts = []
                            for s in surface_names:
                                v = batch["init_verts_vox"][i][s].to(device)
                                v_all.append(v)
                                counts.append(int(v.shape[0]))
                            per_counts_init.append(counts)
                            merged_init_list.append(torch.cat(v_all, dim=0))

                        lengths = torch.tensor([v.shape[0] for v in merged_init_list], device=device, dtype=torch.long)
                        Vmax = int(lengths.max().item())
                        padded_init = torch.zeros((B, Vmax, 3), device=device, dtype=merged_init_list[0].dtype)
                        for i in range(B):
                            padded_init[i, :lengths[i]] = merged_init_list[i]

                        pred_vox = net(padded_init, vol, int(cfg.model.n_steps))

                        for i in range(B):
                            A = aff[i]
                            sh = shift[i].view(1, 3)

                            pred_i = pred_vox[i, :lengths[i]]
                            splits = torch.split(pred_i, per_counts_init[i], dim=0)

                            pred_mm = {}
                            pred_f = {}

                            for j, s in enumerate(surface_names):
                                pv = splits[j]
                                gv = batch["gt_verts_vox"][i][s].to(device)

                                pv_mm = voxel_to_world(pv - sh, A)
                                gv_mm = voxel_to_world(gv - sh, A)

                                f = batch["init_faces"][i][s].to(device).long()
                                gf = batch["gt_faces"][i][s].to(device).long()

                                if mesh_is_valid(pv_mm, f):
                                    pred_mm[s] = pv_mm
                                    pred_f[s] = f

                                if (not mesh_is_valid(pv_mm, f)) or (not mesh_is_valid(gv_mm, gf)):
                                    continue

                                mp = Meshes(verts=[pv_mm], faces=[f])
                                mg = Meshes(verts=[gv_mm], faces=[gf])

                                pp = sample_points_from_meshes(mp, num_samples=Pval)
                                pg = sample_points_from_meshes(mg, num_samples=Pval)

                                csq, _ = chamfer_distance(pp, pg)  # scalar (mm^2)

                                val_csq_sum += float(csq.item())
                                val_count += 1.0
                                val_surf[s]["csq"] += float(csq.item())
                                val_surf[s]["count"] += 1.0

                            # collision checks
                            if do_collision_check and HAS_FCL:
                                if ("lh_white" in pred_mm) and ("lh_pial" in pred_mm):
                                    is_col, ncon = count_collisions_inmemory(
                                        pred_mm["lh_white"], pred_f["lh_white"],
                                        pred_mm["lh_pial"], pred_f["lh_pial"]
                                    )
                                    if is_col is not None:
                                        lh_total += 1.0
                                        lh_hit += 1.0 if is_col else 0.0
                                        lh_contacts_sum += float(ncon)

                                if ("rh_white" in pred_mm) and ("rh_pial" in pred_mm):
                                    is_col, ncon = count_collisions_inmemory(
                                        pred_mm["rh_white"], pred_f["rh_white"],
                                        pred_mm["rh_pial"], pred_f["rh_pial"]
                                    )
                                    if is_col is not None:
                                        rh_total += 1.0
                                        rh_hit += 1.0 if is_col else 0.0
                                        rh_contacts_sum += float(ncon)

                                if ("lh_pial" in pred_mm) and ("rh_pial" in pred_mm):
                                    is_col, ncon = count_collisions_inmemory(
                                        pred_mm["lh_pial"], pred_f["lh_pial"],
                                        pred_mm["rh_pial"], pred_f["rh_pial"]
                                    )
                                    if is_col is not None:
                                        lr_total += 1.0
                                        lr_hit += 1.0 if is_col else 0.0
                                        lr_contacts_sum += float(ncon)

                # log val
                if val_count > 0:
                    csq_mean = val_csq_sum / val_count
                    rmse_mm = math.sqrt(max(csq_mean, 0.0))

                    surf_str = ", ".join(
                        f"{s}={math.sqrt(max(val_surf[s]['csq']/max(val_surf[s]['count'],1.0),0.0)):.4f}mm"
                        for s in surface_names
                    )
                    log.info("Epoch %d [Val] | ChamferRMSE=%.4f mm | Surfaces: %s", epoch, rmse_mm, surf_str)

                    if do_collision_check:
                        if not HAS_FCL:
                            log.info("Epoch %d [Val] | Collision check skipped (python-fcl not available).", epoch)
                        else:
                            def fmt_stats(total, hit, csum):
                                if total <= 0:
                                    return "NA"
                                pct = 100.0 * (hit / total)
                                mean_all = csum / total
                                mean_hit = csum / max(hit, 1.0)
                                return f"{hit:.0f}/{total:.0f} ({pct:.2f}%) | MeanContacts(all)={mean_all:.2f} | MeanContacts(hit)={mean_hit:.2f}"

                            log.info("Epoch %d [Val] | White–Pial Collisions LH: %s", epoch, fmt_stats(lh_total, lh_hit, lh_contacts_sum))
                            log.info("Epoch %d [Val] | White–Pial Collisions RH: %s", epoch, fmt_stats(rh_total, rh_hit, rh_contacts_sum))
                            log.info("Epoch %d [Val] | Pial–Pial Collisions LR: %s", epoch, fmt_stats(lr_total, lr_hit, lr_contacts_sum))

                    scheduler.step(rmse_mm)

                    if tb_writer is not None:
                        tb_writer.add_scalar("val/rmse_mm", rmse_mm, epoch)

                        if do_collision_check and HAS_FCL:
                            total = lh_total + rh_total
                            hit = lh_hit + rh_hit
                            csum = lh_contacts_sum + rh_contacts_sum
                            if total > 0:
                                pct = 100.0 * (hit / total)
                                tb_writer.add_scalar("collisions/whitepial_pct_pairs_colliding_total", pct, epoch)
                                tb_writer.add_scalar("collisions/whitepial_num_pairs_colliding_total", hit, epoch)
                                tb_writer.add_scalar("collisions/whitepial_mean_contacts_all_total", csum / total, epoch)
                                tb_writer.add_scalar("collisions/whitepial_mean_contacts_hit_total", csum / max(hit, 1.0), epoch)

                            if lr_total > 0:
                                pct_lr = 100.0 * (lr_hit / lr_total)
                                tb_writer.add_scalar("collisions/piallr_pct_pairs_colliding", pct_lr, epoch)
                                tb_writer.add_scalar("collisions/piallr_num_pairs_colliding", lr_hit, epoch)
                                tb_writer.add_scalar("collisions/piallr_mean_contacts_all", lr_contacts_sum / lr_total, epoch)
                                tb_writer.add_scalar("collisions/piallr_mean_contacts_hit", lr_contacts_sum / max(lr_hit, 1.0), epoch)

                    # best + early stop
                    if rmse_mm < (best_val - early_delta):
                        best_val = rmse_mm
                        no_improve = 0
                        ckpt = os.path.join(out_root, "checkpoints", "deform_best_rmse.pth")
                        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                        torch.save(net.state_dict(), ckpt)
                        log.info("🌟 Best model updated at epoch %d | RMSE=%.4f mm -> %s", epoch, rmse_mm, ckpt)
                    else:
                        no_improve += 1

                    if early_patience > 0 and no_improve >= early_patience:
                        log.info("🛑 Early stopping after %d validations without improvement.", early_patience)
                        stop_tensor.fill_(1)

            net.train()

        # Sync early-stop decision across ranks
        if is_distributed:
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()

        if stop_tensor.item() == 1:
            break

    if tb_writer is not None:
        tb_writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
