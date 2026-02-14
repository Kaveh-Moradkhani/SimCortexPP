from __future__ import annotations

import sys
import subprocess
import typer

from nipoppy_custom_pipeline.preproc.fs_to_mni import app as fs_to_mni_app

app = typer.Typer(help="SimCortexPP (SCPP) CLI")

# Preprocessing
app.add_typer(fs_to_mni_app, name="fs-to-mni", help="FreeSurfer -> MNI preprocessing")

# Segmentation group
seg_app = typer.Typer(help="Segmentation (3D U-Net) stage")
app.add_typer(seg_app, name="seg")


@seg_app.command("train")
def seg_train(
    overrides: list[str] = typer.Argument(None, help="Hydra overrides, e.g. dataset.path=... trainer.use_ddp=true"),
    torchrun: bool = typer.Option(False, "--torchrun", help="Launch training with torchrun for DDP"),
    nproc_per_node: int = typer.Option(1, "--nproc-per-node", help="GPUs per node for torchrun"),
):
    """
    Run segmentation training.
    - Single GPU: scpp seg train ...
    - Multi GPU:  scpp seg train --torchrun --nproc-per-node 2 trainer.use_ddp=true ...
    """
    if torchrun:
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            "-m",
            "nipoppy_custom_pipeline.seg.train",
        ]
    else:
        cmd = [sys.executable, "-m", "nipoppy_custom_pipeline.seg.train"]

    cmd += (overrides or [])
    raise typer.Exit(subprocess.call(cmd))


@seg_app.command("infer")
def seg_infer(overrides: list[str] = typer.Argument(None)):
    """Run segmentation inference (Hydra)."""
    cmd = [sys.executable, "-m", "nipoppy_custom_pipeline.seg.inference"] + (overrides or [])
    raise typer.Exit(subprocess.call(cmd))


@seg_app.command("eval")
def seg_eval(overrides: list[str] = typer.Argument(None)):
    """Run segmentation evaluation (Hydra)."""
    cmd = [sys.executable, "-m", "nipoppy_custom_pipeline.seg.eval"] + (overrides or [])
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
