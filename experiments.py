"""
Experiment runners for Student 2's analysis.

Experiments implemented:
  1. Architecture comparison (FC vs. CNN vs. LSTM)
  2. Latent dimension sweep
  3. Noise robustness (train with one noise, test on varying levels)
  4. Hyperparameter grid search (lr × batch_size)
"""

import os
import json
import torch
import numpy as np
from typing import Callable, Dict, List
from torch.utils.data import DataLoader, Dataset

from models import build_model
from metrics import evaluate_model
from train import train
from noise import make_noise_fn, gaussian_noise
from visualize import (
    plot_training_curves,
    plot_latent_dim_results,
    plot_noise_robustness,
    plot_noise_type_matrix,
    plot_signals,
)
import config


def _get_device() -> str:
    return config.DEVICE


@torch.no_grad()
def _save_reconstruction_example(
    model: torch.nn.Module,
    test_loader,
    noise_fn: Callable,
    device: str,
    save_path: str,
    title: str = "Signal Reconstruction",
):
    """Grab one test sample, denoise it, and save a multi-channel comparison plot."""
    model.eval()
    batch = next(iter(test_loader))
    clean = batch[:1].to(device)       # (1, C, L)
    noisy = noise_fn(clean)
    recon = model(noisy)
    plot_signals(
        clean[0].cpu().numpy(),
        noisy[0].cpu().numpy(),
        recon[0].cpu().numpy(),
        title=title,
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────
# Experiment 1: Architecture Comparison
# ─────────────────────────────────────────────────────────────────
def run_architecture_comparison(
    train_loader,
    val_loader,
    test_loader,
    noise_fn: Callable,
    results_dir: str = "results",
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
) -> Dict[str, Dict]:
    """Train FC, CNN, and LSTM autoencoders and compare test metrics."""
    device = _get_device()
    os.makedirs(results_dir, exist_ok=True)
    summary = {}

    for arch in ["fc", "cnn", "lstm", "unet"]:
        print(f"\n{'='*50}")
        print(f"  Architecture: {arch.upper()}")
        print(f"{'='*50}")

        model = build_model(
            arch, config.WINDOW_SIZE, config.LATENT_DIM, config.NUM_CHANNELS
        )
        ckpt = os.path.join(results_dir, "checkpoints", f"{arch}_best.pt")

        history = train(
            model, train_loader, val_loader, noise_fn,
            epochs=epochs, lr=lr, device=device,
            checkpoint_path=ckpt, verbose=True,
        )

        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))

        test_metrics = evaluate_model(model, test_loader, noise_fn, device)
        summary[arch] = {"history": history, "test": test_metrics}

        print(
            f"  Test MSE={test_metrics['mse']:.6f} | "
            f"SNR_out={test_metrics['snr_out']:.2f} dB | "
            f"SNRi={test_metrics['snr_improvement']:.2f} dB\n"
            f"  Tremor Power MAE (4-6Hz): In={test_metrics['tremor_mae_in']:.2f} | Out={test_metrics['tremor_mae_out']:.2f}"
        )

        plot_training_curves(
            history,
            title=f"{arch.upper()} Training Curves",
            save_path=os.path.join(results_dir, f"{arch}_training_curves.png"),
        )

        _save_reconstruction_example(
            model, test_loader, noise_fn, device,
            save_path=os.path.join(results_dir, f"{arch}_reconstruction.png"),
            title=f"{arch.upper()} — Denoising Example (Gaussian σ={config.GAUSSIAN_SIGMA})",
        )

    _save_json(summary, os.path.join(results_dir, "architecture_comparison.json"))
    return summary


# ─────────────────────────────────────────────────────────────────
# Experiment 2: Latent Dimension Sweep
# ─────────────────────────────────────────────────────────────────
def run_latent_dim_experiment(
    train_loader,
    val_loader,
    test_loader,
    noise_fn: Callable,
    latent_dims: List[int] = None,
    archs: List[str] = None,
    results_dir: str = "results",
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
) -> Dict:
    """Sweep latent dimension for each architecture and record MSE/SNR."""
    if latent_dims is None:
        latent_dims = config.LATENT_DIM_SWEEP
    if archs is None:
        archs = ["fc", "cnn", "lstm", "unet"]

    device = _get_device()
    os.makedirs(results_dir, exist_ok=True)

    mse_results: Dict[str, List[float]] = {a: [] for a in archs}
    snr_results: Dict[str, List[float]] = {a: [] for a in archs}

    for latent_dim in latent_dims:
        print(f"\n--- Latent dim = {latent_dim} ---")
        for arch in archs:
            model = build_model(
                arch, config.WINDOW_SIZE, latent_dim, config.NUM_CHANNELS
            )
            ckpt = os.path.join(
                results_dir, "checkpoints", f"{arch}_latent{latent_dim}.pt"
            )
            train(
                model, train_loader, val_loader, noise_fn,
                epochs=epochs, lr=lr, device=device,
                checkpoint_path=ckpt, verbose=False,
            )
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device))

            m = evaluate_model(model, test_loader, noise_fn, device)
            mse_results[arch].append(m["mse"])
            snr_results[arch].append(m["snr_out"])
            print(
                f"  {arch.upper():4s} | latent={latent_dim:3d} | "
                f"MSE={m['mse']:.5f} | SNR={m['snr_out']:.2f} dB"
            )

    plot_latent_dim_results(
        latent_dims, mse_results, snr_results,
        save_path=os.path.join(results_dir, "latent_dim_sweep.png"),
    )

    summary = {"latent_dims": latent_dims, "mse": mse_results, "snr": snr_results}
    _save_json(summary, os.path.join(results_dir, "latent_dim_sweep.json"))
    return summary


# ─────────────────────────────────────────────────────────────────
# Experiment 3: Noise Robustness
# ─────────────────────────────────────────────────────────────────
def run_noise_robustness_experiment(
    train_loader,
    val_loader,
    test_loader,
    train_sigma: float = 0.1,
    test_sigmas: List[float] = None,
    archs: List[str] = None,
    results_dir: str = "results",
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
) -> Dict:
    """
    Train each model with a fixed Gaussian noise level (train_sigma),
    then evaluate on a range of test noise levels.
    """
    if test_sigmas is None:
        test_sigmas = config.NOISE_SIGMA_SWEEP
    if archs is None:
        archs = ["fc", "cnn", "lstm", "unet"]

    device = _get_device()
    os.makedirs(results_dir, exist_ok=True)

    train_noise = make_noise_fn("gaussian", sigma=train_sigma)
    snri_results: Dict[str, List[float]] = {a: [] for a in archs}

    for arch in archs:
        print(f"\n--- Noise robustness: {arch.upper()} (train σ={train_sigma}) ---")
        model = build_model(
            arch, config.WINDOW_SIZE, config.LATENT_DIM, config.NUM_CHANNELS
        )
        ckpt = os.path.join(results_dir, "checkpoints", f"{arch}_noise_robust.pt")

        train(
            model, train_loader, val_loader, train_noise,
            epochs=epochs, lr=lr, device=device,
            checkpoint_path=ckpt, verbose=False,
        )
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))

        for sigma in test_sigmas:
            test_noise = make_noise_fn("gaussian", sigma=sigma)
            m = evaluate_model(model, test_loader, test_noise, device)
            snri_results[arch].append(m["snr_improvement"])
            print(f"  σ={sigma:.2f} | SNRi={m['snr_improvement']:.2f} dB")

    plot_noise_robustness(
        test_sigmas, snri_results,
        save_path=os.path.join(results_dir, "noise_robustness.png"),
    )

    summary = {
        "train_sigma": train_sigma,
        "test_sigmas": test_sigmas,
        "snr_improvement": snri_results,
    }
    _save_json(summary, os.path.join(results_dir, "noise_robustness.json"))
    return summary


# ─────────────────────────────────────────────────────────────────
# Experiment 4: Noise-Type Cross-Evaluation
# ─────────────────────────────────────────────────────────────────
def run_noise_type_experiment(
    train_loader,
    val_loader,
    test_loader,
    archs: List[str] = None,
    results_dir: str = "results",
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
) -> Dict:
    """
    Train each architecture on each of the four noise types, then evaluate on
    all four noise types → 4×4 SNRi matrix per architecture.

    Rows   = noise type used during training.
    Columns = noise type used during evaluation.
    Diagonal = matched condition; off-diagonal = cross-noise generalisation.

    This directly addresses the spec requirement:
    "Train or test the models under multiple noise models and compare performance."
    """
    if archs is None:
        archs = ["fc", "cnn", "lstm", "unet"]

    noise_types = ["gaussian", "masking", "impulse", "sinusoidal"]
    noise_kwargs = {
        "gaussian":   {"sigma": config.GAUSSIAN_SIGMA},
        "masking":    {"mask_prob": config.MASK_PROB, "mask_len": config.MASK_LEN},
        "impulse":    {},
        "sinusoidal": {},
    }

    device = _get_device()
    os.makedirs(results_dir, exist_ok=True)
    all_results: Dict[str, Dict] = {}

    for arch in archs:
        print(f"\n{'='*50}")
        print(f"  Noise-type cross-eval: {arch.upper()}")
        print(f"{'='*50}")

        # snri_matrix[train_type][test_type]
        snri_matrix: Dict[str, Dict[str, float]] = {t: {} for t in noise_types}

        for train_type in noise_types:
            train_noise = make_noise_fn(train_type, **noise_kwargs[train_type])
            model = build_model(
                arch, config.WINDOW_SIZE, config.LATENT_DIM, config.NUM_CHANNELS
            )
            ckpt = os.path.join(
                results_dir, "checkpoints", f"{arch}_noisetype_{train_type}.pt"
            )

            print(f"  Training on [{train_type}] ...")
            train(
                model, train_loader, val_loader, train_noise,
                epochs=epochs, lr=lr, device=device,
                checkpoint_path=ckpt, verbose=False,
            )
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device))

            for test_type in noise_types:
                test_noise = make_noise_fn(test_type, **noise_kwargs[test_type])
                m = evaluate_model(model, test_loader, test_noise, device)
                snri_matrix[train_type][test_type] = m["snr_improvement"]

            row = "  ".join(
                f"{t[:4]}={snri_matrix[train_type][t]:.1f}dB" for t in noise_types
            )
            print(f"    test → {row}")

            # Save one reconstruction example per train-noise type
            _save_reconstruction_example(
                model, test_loader, train_noise, device,
                save_path=os.path.join(
                    results_dir, f"{arch}_noisetype_{train_type}_reconstruction.png"
                ),
                title=f"{arch.upper()} trained on {train_type} noise",
            )

        plot_noise_type_matrix(
            snri_matrix, noise_types, arch,
            save_path=os.path.join(results_dir, f"{arch}_noise_type_matrix.png"),
        )
        all_results[arch] = snri_matrix

    _save_json(all_results, os.path.join(results_dir, "noise_type_crosseval.json"))
    return all_results


# ─────────────────────────────────────────────────────────────────
# Experiment 5: Hyperparameter Grid Search
# ─────────────────────────────────────────────────────────────────
def run_hyperparameter_search(
    train_dataset: Dataset,
    val_dataset: Dataset,
    noise_fn: Callable,
    arch: str = "cnn",
    lr_list: List[float] = None,
    batch_list: List[int] = None,
    results_dir: str = "results",
    epochs: int = 50,
) -> Dict:
    """
    Grid search over learning rates and batch sizes.
    Accepts PyTorch Dataset objects for train and val so that DataLoaders
    can be rebuilt with each candidate batch size.
    Returns the best (lr, batch_size) pair and full results table.
    """
    if lr_list is None:
        lr_list = config.LR_SWEEP
    if batch_list is None:
        batch_list = config.BATCH_SWEEP

    device = _get_device()
    os.makedirs(results_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_config = None
    table = []

    loader_kwargs = dict(pin_memory=torch.cuda.is_available(), num_workers=0)

    for lr in lr_list:
        for bs in batch_list:
            train_loader = DataLoader(
                train_dataset, batch_size=bs, shuffle=True, **loader_kwargs
            )
            val_loader = DataLoader(
                val_dataset, batch_size=bs, shuffle=False, **loader_kwargs
            )
            model = build_model(
                arch, config.WINDOW_SIZE, config.LATENT_DIM, config.NUM_CHANNELS
            )

            history = train(
                model, train_loader, val_loader, noise_fn,
                epochs=epochs, lr=lr, device=device,
                checkpoint_path=None, verbose=False,
            )
            best_val = min(history["val_loss"])
            table.append({"lr": lr, "batch": bs, "best_val": best_val})

            if best_val < best_val_loss:
                best_val_loss = best_val
                best_config = {"lr": lr, "batch": bs}

            print(f"  lr={lr:.0e} | batch={bs:3d} | best_val={best_val:.6f}")

    print(f"\nBest config: lr={best_config['lr']:.0e}, batch={best_config['batch']}")
    summary = {"best": best_config, "table": table}
    _save_json(summary, os.path.join(results_dir, "hyperparam_search.json"))
    return summary


# ─────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────
def _save_json(data, path: str):
    """Recursively convert tensors/numpy to Python primitives and save JSON."""
    def _convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.ndarray, list)):
            return [_convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(data), f, indent=2)
    print(f"Saved: {path}")
