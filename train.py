"""
Training and validation loop.
Student 1 primary contribution; provided here for integration.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Dict, List


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
                recon = model(noisy)
                mse_loss = nn.functional.mse_loss(recon, clean)
                clean_fft = torch.fft.rfft(clean, dim=-1, norm="ortho")
                recon_fft = torch.fft.rfft(recon, dim=-1, norm="ortho")
                freq_loss = nn.functional.l1_loss(torch.abs(recon_fft), torch.abs(clean_fft))
                loss = mse_loss + alpha * freq_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            recon = model(noisy)
            mse_loss = nn.functional.mse_loss(recon, clean)
            clean_fft = torch.fft.rfft(clean, dim=-1, norm="ortho")
            recon_fft = torch.fft.rfft(recon, dim=-1, norm="ortho")
            freq_loss = nn.functional.l1_loss(torch.abs(recon_fft), torch.abs(clean_fft))
            loss = mse_loss + alpha * freq_loss
            
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

        mse_loss = nn.functional.mse_loss(recon, clean)
        clean_fft = torch.fft.rfft(clean, dim=-1, norm="ortho")
        recon_fft = torch.fft.rfft(recon, dim=-1, norm="ortho")
        freq_loss = nn.functional.l1_loss(torch.abs(recon_fft), torch.abs(clean_fft))

        hybrid = mse_loss + alpha * freq_loss
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
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history = {"train_loss": [], "val_loss": [], "val_mse": []}
    best_val = float("inf")
    
    # Initialize AMP Scaler if on CUDA GPU (e.g. RTX 2080 Ti)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    for epoch in range(1, epochs + 1):
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

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | val_mse={val_mse:.6f} | lr={lr_now:.2e}"
            )

    return history
