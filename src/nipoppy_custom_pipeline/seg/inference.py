import os
import logging
from pathlib import Path

import torch
import nibabel as nib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import numpy as np

from nipoppy_custom_pipeline.seg.models.unet import Unet
from nipoppy_custom_pipeline.seg.data.dataloader import PredictSegDataset


def setup_logger(log_dir: str, filename: str = "inference.log"):
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


def _strip_module_prefix(state_dict):
    # allows loading checkpoints saved from DataParallel/DDP that may contain "module."
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


@hydra.main(
    version_base="1.3",
    config_path="pkg://nipoppy_custom_pipeline.configs.seg",
    config_name="inference",
)
def main(cfg):
    setup_logger(cfg.outputs.log_dir, "inference.log")
    logging.info("=== Inference config ===")
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    # Dataset / DataLoader (NEW layout)
    ds = PredictSegDataset(
        deriv_root=cfg.dataset.path,
        split_csv=cfg.dataset.split_file,
        split_name=cfg.dataset.split_name,
        session_label=cfg.dataset.session_label,
        space=cfg.dataset.space,
        pad_mult=cfg.dataset.pad_mult,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        pin_memory=pin_memory,
    )

    # Output root (NEW convention)
    pred_root = Path(cfg.outputs.pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    # Model
    model = Unet(c_in=cfg.model.in_channels, c_out=cfg.model.out_channels).to(device)
    logging.info(f"Loading checkpoint from: {cfg.model.ckpt_path}")
    state = torch.load(cfg.model.ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    writer = SummaryWriter(cfg.outputs.log_dir)

    with torch.no_grad():
        pbar = tqdm(dl, desc="Inferring", total=len(dl))
        processed = 0

        for step, batch in enumerate(pbar):
            # NEW: (vol, sub, ses, affine, orig_shape)
            vol, sub, ses, affine, orig_shape = batch
            vol = vol.to(device, non_blocking=True)  # [B,1,D',H',W']

            shapes = orig_shape.cpu().numpy() if isinstance(orig_shape, torch.Tensor) else np.array(orig_shape)
            affines = affine.cpu().numpy() if isinstance(affine, torch.Tensor) else np.array(affine)

            logits = model(vol)                         # [B,C,D',H',W']
            pred = logits.argmax(dim=1).cpu().numpy()   # [B,D',H',W']

            for b in range(pred.shape[0]):
                sid = sub[b]
                ses_b = ses[b]
                D, H, W = shapes[b].tolist()
                pred_b = pred[b, :D, :H, :W].astype(np.int16)

                out_dir = pred_root / sid / ses_b / "anat"
                out_dir.mkdir(parents=True, exist_ok=True)

                stem = f"{sid}_{ses_b}"
                out_path = out_dir / f"{stem}_space-{cfg.dataset.space}_desc-seg9_pred.nii.gz"

                out_img = nib.Nifti1Image(pred_b, affines[b])
                nib.save(out_img, str(out_path))

                logging.info(f"Saved prediction: {out_path}")
                processed += 1

            writer.add_scalar("inference/processed_subjects", processed, step)

    writer.close()
    logging.info("Inference finished.")


if __name__ == "__main__":
    main()
