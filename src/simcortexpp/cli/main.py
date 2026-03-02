from __future__ import annotations

import sys
import subprocess
import typer
from typing import List

from simcortexpp.preproc.fs_to_mni import app as fs_to_mni_app
from simcortexpp.initsurf.generate import main as initsurf_generate


app = typer.Typer(help="SimCortexPP (SCPP) CLI")

def run_module(module: str, overrides: List[str], torchrun: bool = False, nproc: int = 1) -> int:
    if torchrun:
        cmd = ["torchrun", f"--nproc_per_node={nproc}", "-m", module]
    else:
        cmd = [sys.executable, "-m", module]
    cmd += (overrides or [])
    return subprocess.call(cmd)

# Preprocessing
app.add_typer(fs_to_mni_app, name="fs-to-mni", help="FreeSurfer -> MNI preprocessing")

# Segmentation group
seg_app = typer.Typer(help="Segmentation (3D U-Net) stage")
app.add_typer(seg_app, name="seg")

initsurf_app = typer.Typer(help="Initial surface generation (from seg predictions)")

@seg_app.command("train")
def seg_train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides"),
    torchrun: bool = typer.Option(False, "--torchrun", help="Launch training with torchrun for DDP"),
    nproc_per_node: int = typer.Option(1, "--nproc-per-node", help="GPUs per node for torchrun"),
):
    code = run_module("simcortexpp.seg.train", overrides or [], torchrun=torchrun, nproc=nproc_per_node)
    raise typer.Exit(code)

@seg_app.command("infer")
def seg_infer(overrides: List[str] = typer.Argument(None, help="Hydra overrides")):
    code = run_module("simcortexpp.seg.inference", overrides or [])
    raise typer.Exit(code)

@seg_app.command("eval")
def seg_eval(overrides: List[str] = typer.Argument(None, help="Hydra overrides")):
    code = run_module("simcortexpp.seg.eval", overrides or [])
    raise typer.Exit(code)

@initsurf_app.command("generate")
def initsurf_generate_cmd(overrides: List[str] = typer.Argument(None, help="Hydra overrides")):
    code = run_module("simcortexpp.initsurf.generate", overrides or [])
    raise typer.Exit(code)

app.add_typer(initsurf_app, name="initsurf")

# --- add near seg_app / initsurf_app ---
deform_app = typer.Typer(help="Stage 4 — Deformation: train / infer / eval")
app.add_typer(deform_app, name="deform")

@deform_app.command("train")
def deform_train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides"),
    torchrun: bool = typer.Option(False, "--torchrun", help="Launch training with torchrun for DDP"),
    nproc_per_node: int = typer.Option(1, "--nproc-per-node", help="GPUs per node for torchrun"),
):
    code = run_module("simcortexpp.deform.train", overrides or [], torchrun=torchrun, nproc=nproc_per_node)
    raise typer.Exit(code)

@deform_app.command("infer")
def deform_infer(overrides: List[str] = typer.Argument(None, help="Hydra overrides")):
    code = run_module("simcortexpp.deform.inference", overrides or [])
    raise typer.Exit(code)

@deform_app.command("eval")
def deform_eval(overrides: List[str] = typer.Argument(None, help="Hydra overrides")):
    code = run_module("simcortexpp.deform.eval", overrides or [])
    raise typer.Exit(code)

if __name__ == "__main__":
    app()
