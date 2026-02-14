import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.metrics import compute_surface_dice
from omegaconf import OmegaConf

from nipoppy_custom_pipeline.seg.data.dataloader import EvalSegDataset


def setup_logger(log_dir: str, filename: str = "seg_eval.log"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / filename
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def dice_np(gt, pred, num_classes: int, exclude_background: bool = True, eps: float = 1e-6) -> float:
    dices = []
    start_cls = 1 if exclude_background else 0
    for c in range(start_cls, num_classes):
        gt_c = (gt == c)
        pred_c = (pred == c)
        inter = np.logical_and(gt_c, pred_c).sum()
        union = gt_c.sum() + pred_c.sum()
        if union == 0:
            continue
        dices.append((2.0 * inter + eps) / (union + eps))
    return float(np.mean(dices)) if dices else 0.0


def accuracy_np(gt, pred) -> float:
    return float((gt == pred).sum() / gt.size)


def nsd_monai(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    tolerance_vox: float = 1.0,
    include_background: bool = False,
    spacing=(1.0, 1.0, 1.0),
) -> float:
    gt_t = torch.from_numpy(gt).long().unsqueeze(0)
    pred_t = torch.from_numpy(pred).long().unsqueeze(0)

    gt_1h = F.one_hot(gt_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    pred_1h = F.one_hot(pred_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    class_thresholds = [float(tolerance_vox)] * (num_classes if include_background else (num_classes - 1))

    nsd_per_class = compute_surface_dice(
        y_pred=pred_1h,
        y=gt_1h,
        class_thresholds=class_thresholds,
        include_background=include_background,
        distance_metric="euclidean",
        spacing=spacing,
        use_subvoxels=False,
    )[0]

    vals = nsd_per_class[~torch.isnan(nsd_per_class)]
    return float(vals.mean().item()) if vals.numel() else 0.0


@hydra.main(version_base="1.3", config_path="pkg://nipoppy_custom_pipeline.configs.seg", config_name="eval")
def main(cfg):
    setup_logger(cfg.outputs.log_dir, "seg_eval.log")
    logging.info("=== Segmentation Eval config ===")
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    space = getattr(cfg.dataset, "space", "MNI152")
    ses_label = getattr(cfg.dataset, "session_label", "01")

    ds = EvalSegDataset(
        deriv_root=cfg.dataset.path,
        split_csv=cfg.dataset.split_file,
        pred_root=cfg.outputs.pred_root,
        split_name=cfg.dataset.split_name,
        session_label=ses_label,
        space=space,
    )
    logging.info(f"Evaluating {len(ds)} subjects on split={cfg.dataset.split_name}")

    num_classes = cfg.evaluation.num_classes
    exclude_bg = cfg.evaluation.exclude_background
    nsd_tol = float(getattr(cfg.evaluation, "nsd_tolerance_vox", 1.0))

    spacing = (1.0, 1.0, 1.0)  # MNI 1mm

    records = []
    for i in range(len(ds)):
        gt9, pred_arr, sub, ses = ds[i]

        d = dice_np(gt9, pred_arr, num_classes=num_classes, exclude_background=exclude_bg)
        acc = accuracy_np(gt9, pred_arr)
        nsd = nsd_monai(
            gt9,
            pred_arr,
            num_classes=num_classes,
            tolerance_vox=nsd_tol,
            include_background=False,
            spacing=spacing,
        )

        records.append({"subject": sub, "session": ses, "dice": d, "accuracy": acc, "nsd": nsd})
        logging.info(f"{sub} {ses}: Dice={d:.4f}, Acc={acc:.4f}, NSD={nsd:.4f}")

    if not records:
        logging.warning("No subjects evaluated.")
        return

    df = pd.DataFrame(records)
    logging.info(
        f"MEAN Dice={df['dice'].mean():.4f} (SD={df['dice'].std():.4f}), "
        f"MEAN Acc={df['accuracy'].mean():.4f} (SD={df['accuracy'].std():.4f}), "
        f"MEAN NSD={df['nsd'].mean():.4f} (SD={df['nsd'].std():.4f})"
    )

    out_csv = Path(cfg.outputs.eval_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved per-subject metrics to {out_csv}")


if __name__ == "__main__":
    main()
