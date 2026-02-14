import os
import logging
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from nipoppy_custom_pipeline.seg.data.dataloader import SegDataset
from nipoppy_custom_pipeline.seg.models.unet import Unet


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def get_state_dict(model: nn.Module):
    """Handle plain, DataParallel, or DistributedDataParallel models."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def compute_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    exclude_background: bool = True,
    eps: float = 1e-6,
) -> float:

    with torch.no_grad():
        preds = logits.argmax(dim=1)  # [B,D,H,W]

        preds_oh = F.one_hot(preds, num_classes=num_classes)      # [B,D,H,W,C]
        targs_oh = F.one_hot(targets, num_classes=num_classes)

        preds_oh = preds_oh.permute(0, 4, 1, 2, 3).float()        # [B,C,D,H,W]
        targs_oh = targs_oh.permute(0, 4, 1, 2, 3).float()

        if exclude_background and num_classes > 1:
            preds_oh = preds_oh[:, 1:, ...]
            targs_oh = targs_oh[:, 1:, ...]

        pred_flat = preds_oh.reshape(preds_oh.shape[0], preds_oh.shape[1], -1)
        targ_flat = targs_oh.reshape(targs_oh.shape[0], targs_oh.shape[1], -1)

        inter = (pred_flat * targ_flat).sum(-1)
        union = pred_flat.sum(-1) + targ_flat.sum(-1)

        dice_per_class = (2 * inter + eps) / (union + eps)
        return dice_per_class.mean().item()


class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-6,
        exclude_background: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.exclude_background = exclude_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # [B,C,D,H,W]

        # one-hot targets: [B,D,H,W] -> [B,D,H,W,C] -> [B,C,D,H,W]
        targets_oh = F.one_hot(targets, num_classes=self.num_classes)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        if self.exclude_background and self.num_classes > 1:
            probs = probs[:, 1:, ...]
            targets_oh = targets_oh[:, 1:, ...]

        # flatten: [B,C,D,H,W] -> [B,C,N]
        probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_oh.reshape(targets_oh.shape[0], targets_oh.shape[1], -1)

        intersection = (probs_flat * targets_flat).sum(-1)  # [B,C]
        union = probs_flat.sum(-1) + targets_flat.sum(-1)   # [B,C]

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_class  # [B,C]

        return dice_loss.mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
 
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


# -----------------------------------------------------------
# DDP helpers
# -----------------------------------------------------------
def setup_ddp(cfg) -> Tuple[int, int, bool, int]:

    use_ddp = getattr(cfg.trainer, "use_ddp", False)
    if use_ddp and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        return rank, world_size, True, local_rank
    else:
        return 0, 1, False, 0


def is_main_process(rank: int) -> bool:
    return rank == 0


# -----------------------------------------------------------
# Main training (Hydra)
# -----------------------------------------------------------
@hydra.main(version_base="1.3", config_path="pkg://nipoppy_custom_pipeline.configs.seg", config_name="train")
def main(cfg):

    rank, world_size, is_ddp, local_rank = setup_ddp(cfg)

    # Device
    if torch.cuda.is_available():
        if is_ddp:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(cfg.trainer.device)
    else:
        device = torch.device("cpu")

    # Make sure dirs exist (only on rank 0)
    if is_main_process(rank):
        os.makedirs(cfg.outputs.ckpt_dir, exist_ok=True)
        os.makedirs(cfg.outputs.log_dir, exist_ok=True)

    # ---------- Logging ----------
    log_file = os.path.join(cfg.outputs.log_dir, "train_seg.log")
    logging.basicConfig(
        filename=log_file if is_main_process(rank) else None,
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        force=True,  # reset handlers
    )
    if is_main_process(rank):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger("").addHandler(console)

        logging.info("=== Segmentation config ===")
        logging.info("\n" + OmegaConf.to_yaml(cfg))

    writer = SummaryWriter(log_dir=cfg.outputs.log_dir) if is_main_process(rank) else None

    # ---------- Dataset & Dataloaders ----------
    do_aug = getattr(cfg.dataset, "augment", False)
    pin_memory=torch.cuda.is_available()

    train_dataset = SegDataset(
        deriv_root=cfg.dataset.path,
        split_csv=cfg.dataset.split_file,
        split=cfg.dataset.train_split,
        session_label=cfg.dataset.session_label,
        space=cfg.dataset.space,
        pad_mult=cfg.dataset.pad_mult,
        augment=do_aug,
    )

    val_dataset = SegDataset(
        deriv_root=cfg.dataset.path,
        split_csv=cfg.dataset.split_file,
        split=cfg.dataset.val_split,
        session_label=cfg.dataset.session_label,
        space=cfg.dataset.space,
        pad_mult=cfg.dataset.pad_mult,
        augment=False,
    )


    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None


    if is_main_process(rank):
        logging.info(
            f"Train subjects: {len(train_dataset)}, Val subjects: {len(val_dataset)}"
        )
        logging.info(
            f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}"
        )

    # ---------- Model, Loss, Optimizer ----------
    num_classes = cfg.model.out_channels

    model = Unet(c_in=cfg.model.in_channels, c_out=num_classes).to(device)

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        if torch.cuda.device_count() > 1 and getattr(cfg.trainer, "data_parallel", False):
            logging.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLoss(num_classes=num_classes, exclude_background=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.learning_rate)

    best_val_dice = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    # ---------- Training Loop ----------
    for epoch in range(1, cfg.trainer.num_epochs + 1):
        model.train()
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_totals = torch.zeros(4, device=device)
        # 0: loss_sum, 1: dice_metric_sum, 2: acc_sum, 3: n_samples

        train_pbar = tqdm(
            train_loader,
            desc=f"[rank {rank}] Epoch {epoch} [train]",
            total=len(train_loader),
            leave=False,
            disable=not is_main_process(rank),
        )

        for vol, seg in train_pbar:
            vol = vol.to(device, non_blocking=True)  # [B,1,D,H,W]
            seg = seg.to(device, non_blocking=True)  # [B,D,H,W]
            bsz = vol.size(0)

            optimizer.zero_grad()
            logits = model(vol)  # [B,C,D,H,W]

            # --- loss ---
            loss_ce = ce_loss(logits, seg)
            dice_loss_term = dice_loss_fn(logits, seg)       
            loss = loss_ce + dice_loss_term * cfg.trainer.dice_weight

            loss.backward()
            optimizer.step()

            dice_metric = compute_dice_score(logits, seg, num_classes)
            acc = compute_accuracy(logits, seg)

            train_totals[0] += loss.item() * bsz
            train_totals[1] += dice_metric * bsz
            train_totals[2] += acc * bsz
            train_totals[3] += bsz

            if is_main_process(rank):
                train_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    dice=f"{dice_metric:.3f}",
                    acc=f"{acc:.3f}",
                )


        if is_ddp:
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)

        total_loss, total_dice, total_acc, total_samples = train_totals.tolist()
        mean_train_loss = total_loss / max(total_samples, 1)
        mean_train_dice = total_dice / max(total_samples, 1)
        mean_train_acc  = total_acc  / max(total_samples, 1)

        if is_main_process(rank):
            logging.info(
                f"[Epoch {epoch:03d}] "
                f"Train Loss={mean_train_loss:.4f}, "
                f"Dice={mean_train_dice:.4f}, "
                f"Acc={mean_train_acc:.4f}"
            )

            if writer is not None:
                writer.add_scalar("train/loss",     mean_train_loss, epoch)
                writer.add_scalar("train/dice",     mean_train_dice, epoch)
                writer.add_scalar("train/accuracy", mean_train_acc,  epoch)

        # =================== VALIDATION ===================
        if epoch % cfg.trainer.validation_interval == 0:
            model.eval()

            val_totals = torch.zeros(4, device=device)
            # 0: loss_sum, 1: dice_metric_sum, 2: acc_sum, 3: n_samples

            val_pbar = tqdm(
                val_loader,
                desc=f"[rank {rank}] Epoch {epoch} [val]",
                total=len(val_loader),
                leave=False,
                disable=not is_main_process(rank),
            )

            with torch.no_grad():
                for vol, seg in val_pbar:
                    vol = vol.to(device, non_blocking=True)
                    seg = seg.to(device, non_blocking=True)
                    bsz = vol.size(0)

                    logits = model(vol)

                    loss_ce = ce_loss(logits, seg)
                    dice_loss_term = dice_loss_fn(logits, seg)
                    loss = loss_ce + dice_loss_term * cfg.trainer.dice_weight

                    dice_metric = compute_dice_score(logits, seg, num_classes)
                    acc = compute_accuracy(logits, seg)

                    val_totals[0] += loss.item() * bsz
                    val_totals[1] += dice_metric * bsz
                    val_totals[2] += acc * bsz
                    val_totals[3] += bsz

                    if is_main_process(rank):
                        val_pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            dice=f"{dice_metric:.3f}",
                            acc=f"{acc:.3f}",
                        )

            if is_ddp:
                dist.all_reduce(val_totals, op=dist.ReduceOp.SUM)

            total_vloss, total_vdice, total_vacc, total_vsamp = val_totals.tolist()
            mean_val_loss = total_vloss / max(total_vsamp, 1)
            mean_val_dice = total_vdice / max(total_vsamp, 1)
            mean_val_acc  = total_vacc  / max(total_vsamp, 1)

            if is_main_process(rank):
                if writer is not None:
                    writer.add_scalar("val/loss",     mean_val_loss, epoch)
                    writer.add_scalar("val/dice",     mean_val_dice, epoch)
                    writer.add_scalar("val/accuracy", mean_val_acc,  epoch)

                logging.info(
                    f"[Epoch {epoch:03d}] "
                    f"VAL Loss={mean_val_loss:.4f}, "
                    f"Dice={mean_val_dice:.4f}, "
                    f"Acc={mean_val_acc:.4f}"
                )

                if mean_val_dice > best_val_dice:
                    best_val_dice = mean_val_dice
                    best_val_loss = mean_val_loss
                    best_epoch    = epoch

                    best_ckpt_path = os.path.join(
                        cfg.outputs.ckpt_dir, "seg_best_dice.pt"
                    )
                    os.makedirs(cfg.outputs.ckpt_dir, exist_ok=True)
                    torch.save(get_state_dict(model), best_ckpt_path)

                    logging.info(
                        f"New BEST model at epoch {epoch:03d}: "
                        f"Dice={best_val_dice:.4f}, Loss={best_val_loss:.4f}. "
                        f"Saved to {best_ckpt_path}"
                    )

    if writer is not None:
        writer.close()
    if is_ddp:
        dist.destroy_process_group()

    if is_main_process(rank):
        logging.info(
            f"Training finished. Best epoch={best_epoch}, "
            f"best val dice={best_val_dice:.4f}, best val loss={best_val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
