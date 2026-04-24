"""
Main entry point for Project 2: Denoising Autoencoders for 1D Time-Series.

Usage:
    python main.py                     # run all experiments
    python main.py --exp arch          # architecture comparison only
    python main.py --exp latent        # latent dimension sweep
    python main.py --exp noise         # noise robustness
    python main.py --exp noise_types   # noise-type cross-evaluation
    python main.py --exp hyperparam    # hyperparameter search

Each run is saved to results/<YYYYMMDD_HHMMSS>/ so runs never overwrite each other.
Pass --tag my_label to append a human-readable suffix: results/<timestamp>_my_label/.
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import torch

# Suppress PyTorch 2.x warning regarding FFT buffer resizing
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

import config
from noise import make_noise_fn
from experiments import (
    run_architecture_comparison,
    run_latent_dim_experiment,
    run_noise_robustness_experiment,
    run_noise_type_experiment,
    run_hyperparameter_search,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="all",
        choices=["all", "arch", "latent", "noise", "noise_types", "hyperparam"],
        help="Which experiment to run",
    )
    parser.add_argument("--epochs",  type=int,   default=config.EPOCHS)
    parser.add_argument("--lr",      type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch",   type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--results", type=str,   default=config.RESULTS_DIR,
                        help="Base directory; each run creates a timestamped subdirectory")
    parser.add_argument("--tag",     type=str,   default="",
                        help="Optional label appended to the run directory name")
    return parser.parse_args()


def _build_loaders(args):
    """Return (train_loader, val_loader, test_loader, train_dataset, val_dataset)."""
    if config.DATASET == "pads":
        from pads_dataset import build_pads_dataloaders
        print(f"Loading PADS dataset from '{config.DATA_DIR}' ...")
        train_loader, val_loader, test_loader, stats = build_pads_dataloaders(
            data_dir=config.DATA_DIR,
            batch_size=args.batch,
            window_size=config.WINDOW_SIZE,
            wrist=config.WRIST,
            cohorts=config.COHORTS,
            seed=42,
        )
    else:
        from dataset import generate_synthetic_signals, build_dataloaders, SignalDataset
        from torch.utils.data import random_split, DataLoader
        print("Generating synthetic signals ...")
        signals = generate_synthetic_signals(
            n_samples=5000, signal_length=config.SIGNAL_LENGTH
        )
        train_loader, val_loader, test_loader = build_dataloaders(
            signals, batch_size=args.batch
        )
        # Unsqueeze to (B, 1, L) so models receive (B, C, L) in synthetic mode too
        train_loader = _wrap_1d_loader(train_loader, args.batch, shuffle=True)
        val_loader   = _wrap_1d_loader(val_loader,   args.batch, shuffle=False)
        test_loader  = _wrap_1d_loader(test_loader,  args.batch, shuffle=False)

    return train_loader, val_loader, test_loader


def _wrap_1d_loader(loader, batch_size, shuffle):
    """Wrap a (B, L) DataLoader into a (B, 1, L) DataLoader for channel-first models."""
    from torch.utils.data import DataLoader, TensorDataset
    all_data = torch.cat([batch for batch in loader])           # (N, L)
    dataset  = TensorDataset(all_data.unsqueeze(1))             # (N, 1, L)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
        collate_fn=lambda batch: torch.cat([b[0].unsqueeze(0) for b in batch]),
    )


def _make_run_dir(base: str, tag: str) -> str:
    """Create and return results/<timestamp>[_tag]/ for this run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name  = f"{stamp}_{tag}" if tag else stamp
    run_dir = os.path.join(base, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_run_config(run_dir: str, args) -> None:
    """Snapshot CLI args + key config values into run_config.json."""
    cfg = {
        "exp":           args.exp,
        "epochs":        args.epochs,
        "lr":            args.lr,
        "batch":         args.batch,
        "tag":           args.tag,
        "dataset":       config.DATASET,
        "data_dir":      config.DATA_DIR,
        "wrist":         config.WRIST,
        "window_size":   config.WINDOW_SIZE,
        "num_channels":  config.NUM_CHANNELS,
        "latent_dim":    config.LATENT_DIM,
        "gaussian_sigma": config.GAUSSIAN_SIGMA,
        "device":        config.DEVICE,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    # Enable CuDNN benchmark for RTX 2080 Ti speedup (auto-tunes convolution algorithms)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args = parse_args()
    run_dir = _make_run_dir(args.results, args.tag)
    _save_run_config(run_dir, args)

    print("\n" + "=" * 60)
    print(f"SYSTEM INFO: Using device '{config.DEVICE}'")
    if config.DEVICE == "mps":
        print("  -> Apple Silicon GPU (Metal) enabled!")
    elif config.DEVICE == "cuda":
        print("  -> NVIDIA GPU enabled!")
    print("=" * 60 + "\n")

    train_loader, val_loader, test_loader = _build_loaders(args)

    noise_fn = make_noise_fn("gaussian", sigma=config.GAUSSIAN_SIGMA)

    run_all = args.exp == "all"

    if run_all or args.exp == "hyperparam":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Hyperparameter Search")
        print("=" * 60)
        run_hyperparameter_search(
            train_loader.dataset,
            val_loader.dataset,
            noise_fn,
            arch="cnn",
            results_dir=run_dir,
            epochs=min(args.epochs, 50),
        )

    if run_all or args.exp == "arch":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Architecture Comparison (FC / CNN / LSTM)")
        print("=" * 60)
        run_architecture_comparison(
            train_loader, val_loader, test_loader, noise_fn,
            results_dir=run_dir,
            epochs=args.epochs,
            lr=args.lr,
        )

    if run_all or args.exp == "latent":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Latent Dimension Sweep")
        print("=" * 60)
        run_latent_dim_experiment(
            train_loader, val_loader, test_loader, noise_fn,
            results_dir=run_dir,
            epochs=args.epochs,
            lr=args.lr,
        )

    if run_all or args.exp == "noise":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Noise Robustness")
        print("=" * 60)
        run_noise_robustness_experiment(
            train_loader, val_loader, test_loader,
            train_sigma=config.GAUSSIAN_SIGMA,
            results_dir=run_dir,
            epochs=args.epochs,
            lr=args.lr,
        )

    if run_all or args.exp == "noise_types":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Noise-Type Cross-Evaluation (4×4 matrix)")
        print("=" * 60)
        run_noise_type_experiment(
            train_loader, val_loader, test_loader,
            results_dir=run_dir,
            epochs=args.epochs,
            lr=args.lr,
        )

    print("\nAll experiments complete. Results saved to:", run_dir)


if __name__ == "__main__":
    main()
