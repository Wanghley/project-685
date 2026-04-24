import os
import torch

# ── Dataset ──────────────────────────────────────────────────────
DATASET = "pads"          # "pads" | "synthetic"
DATA_DIR = "physionet.org/files/parkinsons-disease-smartwatch/1.0.0"
COHORTS = ("Parkinson's", "Healthy")
WRIST = "Both"

# ── Signal shape ─────────────────────────────────────────────────
NUM_CHANNELS = 6           # Accel X/Y/Z + Gyro X/Y/Z for one wrist
WINDOW_SIZE = 1024         # samples @ 100 Hz → 10.24 s
SIGNAL_LENGTH = WINDOW_SIZE  # alias kept for backward compatibility
SAMPLING_RATE = 100        # Hz

# ── Training ─────────────────────────────────────────────────────
BATCH_SIZE = 128           # larger batch → better GPU utilisation
LEARNING_RATE = 1e-3
EPOCHS = 100
LATENT_DIM = 64   # 6×1024 input → 64 latent = 96× compression; forces real feature learning
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ── Noise parameters ─────────────────────────────────────────────
# σ=0.1 on z-scored signals (unit std) gives SNR_in≈8 dB — enough noise
# that models cannot converge to the trivial identity solution (MSE≈σ²=0.01)
# and must learn genuine denoising to improve further.
# σ=0.05 was too small: the identity mapping achieves MSE≈0.0025 which the
# optimizer finds in the first few epochs, preventing real learning.
GAUSSIAN_SIGMA = 0.1
MASK_PROB = 0.1
MASK_LEN = 20

# ── Experiment sweep ranges ───────────────────────────────────────
LATENT_DIM_SWEEP = [8, 16, 32, 64, 128, 256]
NOISE_SIGMA_SWEEP = [0.05, 0.1, 0.2, 0.4]
LR_SWEEP = [1e-4, 5e-4, 1e-3]
BATCH_SWEEP = [32, 64, 128]

# ── Paths ─────────────────────────────────────────────────────────
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

# ── Performance ──────────────────────────────────────────────────
# Use (cpu_count - 1) workers so the main process has a dedicated core.
# Cap at 4 — empirically safe on both 8-core Apple Silicon and DCC/SLURM
# nodes where PyTorch warns above 4 workers; raise if running on a
# fat node with many cores and no SLURM CPU limit.
NUM_WORKERS    = min(4, max(0, (os.cpu_count() or 1) - 1))
PIN_MEMORY     = torch.cuda.is_available()   # only beneficial on CUDA
PREFETCH_FACTOR = 2                          # batches to prefetch per worker
# torch.compile: enabled by default; train.py does a probe-forward after
# compile() to catch environments where Triton/inductor kernel build fails
# (e.g., missing Python.h on some HPC clusters) and falls back gracefully.
USE_COMPILE    = True
