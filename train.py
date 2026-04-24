"""
Training and validation loop.
Student 1 primary contribution; provided here for integration.
"""

import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Dict, List


def _spectral_loss(recon: torch.Tensor, clean: torch.Tensor, device: str) -> torch.Tensor:
    """
    Spectral L1 loss: L1 between magnitude spectra.

    rfft is not yet supported on MPS (PyTorch ≤ 2.4).  When training on MPS
    we compute the FFT on CPU — gradients flow back through the device transfer
    so the model weights on MPS are updated correctly.  The FFT itself is a
    small fraction of total compute, so the cross-device hop is negligible.
    """
    fft_dev = "cpu" if device == "mps" else device
    # clean has no learnable parameters → detach before moving to save memory
    c_f = clean.detach().float().to(fft_dev)
    r_f = recon.float().to(fft_dev)          # gradient flows through .to()
    clean_fft = torch.fft.rfft(c_f, dim=-1, norm="ortho")
    recon_fft = torch.fft.rfft(r_f, dim=-1, norm="ortho")
    freq_loss = nn.functional.l1_loss(recon_fft.abs(), clean_fft.abs())
    return freq_loss.to(device)              # bring scalar loss back to training device


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    noise_fn: Callable,
    device: str,
    alpha: float = 0.1,
    scaler = None,
) -> float:
    """Returns train_loss."""
    if len(loader.dataset) == 0:
        return float("inf")
    model.train()
    total_loss = 0.0
    for batch in loader:
        clean = batch.to(device)
        noisy = noise_fn(clean)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                recon    = model(noisy)
                mse_loss = nn.functional.mse_loss(recon, clean)
                freq_loss = _spectral_loss(recon, clean, device)
                loss = mse_loss + alpha * freq_loss

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            recon    = model(noisy)
            mse_loss = nn.functional.mse_loss(recon, clean)
            freq_loss = _spectral_loss(recon, clean, device)
            loss = mse_loss + alpha * freq_loss

            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * len(clean)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    noise_fn: Callable,
    device: str,
    alpha: float = 0.1,
) -> tuple:
    """Returns (hybrid_loss, pure_mse) so callers can track both metrics."""
    if len(loader.dataset) == 0:
        return float("inf"), float("inf")
    model.eval()
    total_hybrid = 0.0
    total_mse = 0.0
    for batch in loader:
        clean = batch.to(device)
        noisy = noise_fn(clean)
        recon = model(noisy)

        mse_loss  = nn.functional.mse_loss(recon, clean)
        freq_loss = _spectral_loss(recon, clean, device)
        hybrid    = mse_loss + alpha * freq_loss
        total_hybrid += hybrid.item() * len(clean)
        total_mse    += mse_loss.item() * len(clean)

    n = len(loader.dataset)
    return total_hybrid / n, total_mse / n


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    noise_fn: Callable,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str = None,
    verbose: bool = True,
    alpha: float = 0.1,
) -> Dict[str, List[float]]:
    """
    Full training loop with LR scheduling and optional checkpointing.
    Returns history dict with 'train_loss' and 'val_loss' lists.
    """
    model.to(device)

    # torch.compile — fuses ops and generates optimised GPU kernels.
    # Skip for the WaveletDenoiser (no trainable params, pure NumPy forward).
    #
    # torch.compile() is lazy: it doesn't actually build kernels until the
    # first forward pass.  On HPC clusters the Triton CUDA backend's one-time
    # C-extension build can fail because Python development headers (Python.h)
    # are absent from GCC's include path.  The failure normally surfaces as a
    # noisy InductorError mid-training, not at the torch.compile() call site.
    #
    # Fix: use _triton_ok() to probe the Triton driver *before* compile so the
    # check happens exactly once and silently, without invoking gcc at all.
    # If the probe fails, torch.compile is skipped and training proceeds on
    # the uncompiled model (AMP and cuDNN benchmark still active).
    def _triton_ok() -> bool:
        """
        Return True only if the environment can actually build Triton kernels.

        Triton's CUDA driver compiles a small C extension (cuda_utils.c) on
        first use; this requires Python development headers (Python.h).
        We check for the header BEFORE touching Triton so that gcc is never
        invoked and no temp-file noise appears in the logs.
        """
        import sysconfig
        inc = sysconfig.get_path("include")          # e.g. /usr/include/python3.9
        if inc:
            import pathlib
            if not (pathlib.Path(inc) / "Python.h").exists():
                return False                         # headers missing → skip
        # Headers present (or unknown); do a cheap Triton driver probe.
        try:
            import triton.runtime.driver as _td
            _ = _td.active.get_current_target()
            return True
        except Exception:
            return False

    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        import config as _cfg
        _use_compile = getattr(_cfg, "USE_COMPILE", False)
    except ImportError:
        _use_compile = False
    if _use_compile and trainable and device == "cuda" and hasattr(torch, "compile"):
        if _triton_ok():
            try:
                model = torch.compile(model, mode="reduce-overhead")
                if verbose:
                    print("  torch.compile enabled (reduce-overhead)")
            except Exception as _e:
                if verbose:
                    print(f"  torch.compile skipped: {_e}")
        else:
            if verbose:
                print("  torch.compile skipped: Triton CUDA driver unavailable "
                      "(Python.h missing on this node — training without compile)")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history = {"train_loss": [], "val_loss": [], "val_mse": []}
    best_val = float("inf")
    
    # Initialize AMP Scaler if on CUDA GPU (e.g. RTX 2080 Ti)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    start_time = time.time()
    epochs_iter = tqdm(range(1, epochs + 1), desc="Training") if verbose else range(1, epochs + 1)

    for epoch in epochs_iter:
        train_loss = train_one_epoch(model, train_loader, optimizer, noise_fn, device, alpha=alpha, scaler=scaler)
        val_loss, val_mse = validate(model, val_loader, noise_fn, device, alpha=alpha)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)

        if val_loss < best_val:
            best_val = val_loss
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            epochs_iter.set_postfix({"train": f"{train_loss:.3f}", "val": f"{val_loss:.3f}", "mse": f"{val_mse:.3f}", "lr": f"{lr_now:.1e}"})

    total_time = time.time() - start_time
    if verbose:
        print(f"  Took: {total_time:.1f}s ({total_time/epochs:.3f}s / epoch)")

    return history
