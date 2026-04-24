"""
Dataset utilities — signal generation and windowed loading.
Student 1 primary contribution; synthetic generator provided here for standalone use.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SignalDataset(Dataset):
    """
    Wraps a numpy array of shape (N, signal_length) as a PyTorch Dataset.
    Each item is a float32 tensor of shape (signal_length,).
    """

    def __init__(self, signals: np.ndarray):
        self.data = torch.from_numpy(signals.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_synthetic_signals(
    n_samples: int = 5000,
    signal_length: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate random multi-tone sinusoidal signals normalized to [-1, 1].
    Each signal is a sum of 2-4 sine waves with random frequencies and amplitudes.
    """
    rng = np.random.default_rng(seed)
    signals = []
    for _ in range(n_samples):
        t = np.linspace(0, 1, signal_length)
        n_tones = rng.integers(2, 5)
        freqs = rng.uniform(5, 100, size=n_tones)
        amps  = rng.uniform(0.2, 1.0, size=n_tones)
        sig = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
        sig /= np.max(np.abs(sig)) + 1e-8   # normalize
        signals.append(sig)
    return np.array(signals, dtype=np.float32)


def segment_signal(signal: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Segment a 1D signal into overlapping windows of length `window`."""
    starts = range(0, len(signal) - window + 1, hop)
    return np.stack([signal[s: s + window] for s in starts])


def build_dataloaders(
    signals: np.ndarray,
    batch_size: int = 64,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple:
    """Split signals into train/val/test DataLoaders."""
    dataset = SignalDataset(signals)
    n = len(dataset)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    kwargs = dict(batch_size=batch_size, pin_memory=torch.cuda.is_available(), num_workers=0)
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds,  shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )
