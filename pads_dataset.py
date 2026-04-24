"""
PADS — Parkinson's Disease Smartwatch dataset loader.

Source: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
Paper:  Varghese et al., npj Parkinson's Disease (2023).

File layout expected under data_dir:
    movement/
        observation_<id>.json      # per-subject session metadata
        timeseries/
            <id>_<task>_<wrist>Wrist.txt   # comma-separated, no header
    patients/
        patient_<id>.json          # condition, demographics

Each timeseries file has 7 comma-separated columns (no header):
    Time, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z

We load 6 sensor channels (columns 1–6, skipping Time) for the chosen wrist.
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# First 50 samples contain a smartwatch vibration artifact at task start
_VIBRATION_SAMPLES = 50


# ──────────────────────────────────────────────────────────────────
# Subject-level split
# ──────────────────────────────────────────────────────────────────

def _subject_has_data(data_dir: str, subject_id: str) -> bool:
    """Return True if at least one timeseries .txt file exists for this subject."""
    ts_dir = os.path.join(data_dir, "movement", "timeseries")
    if not os.path.isdir(ts_dir):
        return False
    prefix = f"{subject_id}_"
    return any(f.startswith(prefix) and f.endswith(".txt") for f in os.listdir(ts_dir))


def load_subject_split(
    data_dir: str,
    cohorts: Tuple[str, ...] = ("Parkinson's", "Healthy"),
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[List[str], List[str], List[str]]:
    """
    Read patient JSONs, keep only subjects in ``cohorts`` **that have timeseries
    files on disk**, shuffle deterministically, and split by subject.

    Only subjects with downloaded data are included so that no split is empty
    while the full dataset is still being downloaded.

    Returns three lists of zero-padded subject ID strings (e.g. '001', '042').
    """
    patients_dir = os.path.join(data_dir, "patients")
    subject_ids: List[str] = []

    for fname in sorted(os.listdir(patients_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(patients_dir, fname)) as fh:
            meta = json.load(fh)
        if meta.get("condition") not in cohorts:
            continue
        sid = str(meta["id"]).zfill(3)
        if _subject_has_data(data_dir, sid):
            subject_ids.append(sid)

    if not subject_ids:
        raise RuntimeError(
            f"No subjects with downloaded timeseries found under '{data_dir}'. "
            "Ensure .txt files are present in movement/timeseries/."
        )

    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n = len(subject_ids)
    # Guarantee at least 1 subject per split; adjust when dataset is small
    n_val   = max(1, int(n * ratios[1]))
    n_test  = max(1, int(n * ratios[2]))
    n_train = max(1, n - n_val - n_test)
    # If total is too small to satisfy all three, collapse val into train
    if n_train + n_val + n_test > n:
        n_val   = max(0, n - n_train - n_test)

    train_ids = subject_ids[:n_train]
    val_ids   = subject_ids[n_train : n_train + n_val]
    test_ids  = subject_ids[n_train + n_val :]

    available = len(subject_ids)
    total_target = sum(int(n * r) for r in ratios)
    if available < 10:
        print(
            f"  Warning: only {available} subjects with data on disk. "
            "Splits will be small until the full download completes."
        )
    return train_ids, val_ids, test_ids


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

class PADSDataset(Dataset):
    """
    Loads wrist IMU recordings from PADS for a list of subject IDs.

    Each item is a float32 tensor of shape ``(num_channels, window_size)``
    (default: ``(6, 1024)``).

    Preprocessing applied per recording:
      1. Drop first ``_VIBRATION_SAMPLES`` (50) rows.
      2. Extract non-overlapping windows of ``window_size``; the last partial
         window is zero-padded to ``window_size``.
      3. If ``stats`` are provided, z-score normalise per channel:
         ``x = (x - mean[:, None]) / std[:, None]``.

    All windows are pre-loaded into memory (the 6-channel subset of the full
    dataset is roughly 120 MB for 355 subjects).
    """

    def __init__(
        self,
        data_dir: str,
        subject_ids: List[str],
        window_size: int = 1024,
        wrist: str = "Right",
        stats: Optional[Dict[str, torch.Tensor]] = None,
        overlap: bool = False,
    ):
        self.window_size = window_size
        self.stats = stats
        self.overlap = overlap

        movement_dir = os.path.join(data_dir, "movement")
        windows: List[torch.Tensor] = []

        for sid in subject_ids:
            obs_path = os.path.join(movement_dir, f"observation_{sid}.json")
            if not os.path.exists(obs_path):
                continue
            with open(obs_path) as fh:
                obs = json.load(fh)

            for session in obs["session"]:
                # Locate all files for the requested wrist(s)
                wrist_files: List[str] = []
                for record in session["records"]:
                    loc = record.get("device_location", "")
                    if wrist.lower() == "both" and loc in ("RightWrist", "LeftWrist"):
                        wrist_files.append(record["file_name"])
                    elif loc == f"{wrist}Wrist":
                        wrist_files.append(record["file_name"])

                for wrist_file in wrist_files:
                    full_path = os.path.join(movement_dir, wrist_file)
                    if not os.path.exists(full_path):
                        continue

                    # Load CSV (no header) — expected shape (rows, 7).
                    # Use pandas to skip any malformed rows gracefully.
                    try:
                        import pandas as pd
                        df = pd.read_csv(
                            full_path, header=None,
                            on_bad_lines="skip", engine="python",
                        )
                        if df.shape[1] < 7:
                            continue
                        data = df.iloc[:, :7].values.astype(np.float32)
                        # Drop rows that contain NaN (from skipped bad lines)
                        data = data[~np.isnan(data).any(axis=1)]
                    except Exception:
                        continue

                    if len(data) < _VIBRATION_SAMPLES + 1:
                        continue

                    # Columns 1–6: Accel X/Y/Z, Gyro X/Y/Z
                    signal = data[_VIBRATION_SAMPLES:, 1:7]  # (rows-50, 6)
                    signal = signal.T  # (6, rows-50) — channel-first

                    for win in self._extract_windows(signal):
                        windows.append(torch.tensor(win, dtype=torch.float32))

        self.windows = windows
        # Stacked tensor used for efficient stats computation
        self._stacked: torch.Tensor = (
            torch.stack(windows) if windows else torch.zeros(0, 6, window_size)
        )

    # ------------------------------------------------------------------
    def _extract_windows(self, signal: np.ndarray) -> List[np.ndarray]:
        """Split (C, L) signal into windows; pad last window. Uses 128-sample sliding hop if overlap=True."""
        C, L = signal.shape
        out: List[np.ndarray] = []
        start = 0
        hop_size = 128 if self.overlap else self.window_size
        
        while start < L:
            chunk = signal[:, start : start + self.window_size]
            if chunk.shape[1] < self.window_size:
                if chunk.shape[1] > 0:
                    pad = np.zeros((C, self.window_size - chunk.shape[1]), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=1)
                    out.append(chunk)
                break
            out.append(chunk)
            start += hop_size
        return out

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.windows[idx].clone()  # (C, W)
        if self.stats is not None:
            mean = self.stats["mean"]  # (C,)
            std = self.stats["std"]    # (C,)
            x = (x - mean[:, None]) / std[:, None]
        return x


# ──────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────

def build_pads_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    window_size: int = 1024,
    wrist: str = "Right",
    cohorts: Tuple[str, ...] = ("Parkinson's", "Healthy"),
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """
    Build subject-split, z-score-normalised train/val/test DataLoaders.

    Returns ``(train_loader, val_loader, test_loader, stats)`` where
    ``stats = {'mean': Tensor(C,), 'std': Tensor(C,)}`` is fit on the
    training split and applied to all three splits.
    """
    train_ids, val_ids, test_ids = load_subject_split(
        data_dir, cohorts=cohorts, seed=seed
    )

    print(
        f"  Subject split — train: {len(train_ids)}, "
        f"val: {len(val_ids)}, test: {len(test_ids)}"
    )
    # Guard against accidental leakage
    assert not (set(train_ids) & set(val_ids)), "Train/val subject overlap!"
    assert not (set(train_ids) & set(test_ids)), "Train/test subject overlap!"
    assert not (set(val_ids) & set(test_ids)), "Val/test subject overlap!"

    # Load training set without normalisation to compute channel statistics
    print("  Loading training windows (with overlapping data augmentation) ...")
    train_ds = PADSDataset(data_dir, train_ids, window_size, wrist, stats=None, overlap=True)

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training windows found under '{data_dir}/movement/timeseries/'. "
            "Check that the PADS timeseries .txt files have been downloaded."
        )

    # Per-channel mean and std over (batch, time) dimensions
    stack = train_ds._stacked  # (N, C, W)
    mean = stack.mean(dim=(0, 2))        # (C,)
    std = stack.std(dim=(0, 2)) + 1e-8   # (C,)
    stats: Dict[str, torch.Tensor] = {"mean": mean, "std": std}

    # Apply stats to training set in-place (avoids a second CSV read pass)
    train_ds.stats = stats

    print("  Loading val/test windows ...")
    val_ds = PADSDataset(data_dir, val_ids, window_size, wrist, stats=stats)
    test_ds = PADSDataset(data_dir, test_ids, window_size, wrist, stats=stats)

    print(
        f"  Windows — train: {len(train_ds)}, "
        f"val: {len(val_ds)}, test: {len(test_ds)}"
    )

    # pin_memory is only beneficial (and supported) on CUDA
    pin = torch.cuda.is_available()
    loader_kwargs = dict(pin_memory=pin, num_workers=0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        stats,
    )
