# =============================================================================
# TJEECC PROJECT - CELL 0: SESSION BOOTSTRAP  (portable / no Google Drive)
# Rank stability of 2D semiconductor channel screening
#
# Runs unchanged on: a plain laptop, an HPC login node, or Google Colab.
# Nothing here requires Google Drive. The project root is resolved as:
#
#   1. $TJEECC_ROOT                 if set
#   2. the repository directory     otherwise  (i.e. the parent of src/)
#
# Safe to re-run. Exported names are listed at the bottom of the file.
# =============================================================================

NOTEBOOK_NAME = "step01_ingest_c2db"   # <-- EDIT PER NOTEBOOK / STAGE
SEED = 20260815

# --- 0. stdlib only, before anything can break -------------------------------
import os, sys, subprocess, importlib, importlib.util, importlib.metadata as md
import logging, shutil, platform, tempfile, warnings
from pathlib import Path
from datetime import datetime, timezone

IN_COLAB = importlib.util.find_spec("google.colab") is not None

# --- 1. Project root ---------------------------------------------------------
# Resolution order: $TJEECC_ROOT -> repository directory (parent of this file).
# No Drive mount, no /content assumptions. If you are on Colab and *want*
# Drive, mount it yourself and export TJEECC_ROOT before running this cell.
_env_root = os.environ.get("TJEECC_ROOT", "").strip()
if _env_root:
    ROOT = Path(_env_root).expanduser().resolve()
    _root_src = "$TJEECC_ROOT"
else:
    try:
        _here = Path(__file__).resolve().parent
    except NameError:                       # exec'd / notebook cell, no __file__
        _here = Path.cwd()
        if _here.name != "src" and (_here / "src" / "cell00_bootstrap.py").exists():
            _here = _here / "src"
    ROOT = (_here.parent if _here.name == "src" else _here).resolve()
    _root_src = "repository directory"

SUBDIRS = {
    "raw":       ROOT / "data" / "raw",
    "interim":   ROOT / "data" / "interim",
    "processed": ROOT / "data" / "processed",
    "jarvis":    ROOT / "cache" / "jarvis",
    "figures":   ROOT / "figures",
    "models":    ROOT / "models",
    "logs":      ROOT / "logs",
}
for _name, _p in SUBDIRS.items():
    _p.mkdir(parents=True, exist_ok=True)

# Fast local scratch for the extracted C2DB tree. This holds ~86k small files
# and must NOT live on a network filesystem (Drive FUSE, NFS, SMB): extraction
# there takes hours instead of minutes. Override with $TJEECC_SCRATCH.
_env_scratch = os.environ.get("TJEECC_SCRATCH", "").strip()
if _env_scratch:
    SCRATCH = Path(_env_scratch).expanduser().resolve()
elif IN_COLAB and Path("/content").is_dir():
    SCRATCH = Path("/content/scratch")          # Colab local disk, not Drive
else:
    SCRATCH = Path(tempfile.gettempdir()) / "tjeecc_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
C2DB_TREE = SCRATCH / "c2db_tree"

# --- 2. Dependencies ---------------------------------------------------------
# Preferred: install from requirements.txt before running (see README).
# This block only *reports* what is missing; it installs nothing unless you
# opt in with TJEECC_AUTOINSTALL=1, so the script can never mutate a
# reviewer's environment behind their back.
REQUIRED = {
    "ase":          "ase",
    "pymatgen":     "pymatgen",
    "jarvis-tools": "jarvis",
    "SALib":        "SALib",
    "numpyro":      "numpyro",
    "arviz":        "arviz",
    "tqdm":         "tqdm",
    "pyarrow":      "pyarrow",
}

def _installed(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False

missing = [dist for dist, imp in REQUIRED.items() if not _installed(imp)]

if missing and os.environ.get("TJEECC_AUTOINSTALL", "") == "1":
    print(f"[bootstrap] TJEECC_AUTOINSTALL=1, installing: {', '.join(missing)}")
    # pymatgen is numpy-pin sensitive and can break a CUDA-matched jax build.
    # Install it alone and first so a failure is attributable.
    ordered = ([m for m in missing if m == "pymatgen"] +
               [m for m in missing if m != "pymatgen"])
    for dist in ordered:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "--disable-pip-version-check", "--no-input", dist]
        )
        importlib.invalidate_caches()
    missing = [d for d, i in REQUIRED.items() if not _installed(i)]
    print("[bootstrap] Install complete.")
elif missing:
    print(f"[bootstrap] MISSING packages: {', '.join(missing)}\n"
          f"[bootstrap] Install them with:  pip install -r requirements.txt\n"
          f"[bootstrap] (or re-run with TJEECC_AUTOINSTALL=1 to let this "
          f"script pip-install them)")
else:
    print("[bootstrap] All packages already present.")

# jax / jaxlib are deliberately NOT auto-installed. Colab and most HPC images
# ship a hardware-matched build, and pip-installing jax on top will silently
# drop you to CPU. numpyro uses whatever jax is already present.

# --- 3. Determinism ----------------------------------------------------------
import numpy as np
RNG = np.random.default_rng(SEED)   # pass RNG explicitly; avoid legacy global state
np.random.seed(SEED)                # legacy global, for libraries that still use it

import random as _pyrandom
_pyrandom.seed(SEED)

# Headless plotting. Must precede any jarvis-tools import or it raises on
# backend selection in a display-less runtime.
import matplotlib
matplotlib.use("Agg")

# --- 4. Logging: file + stdout ----------------------------------------------
LOG_PATH = SUBDIRS["logs"] / f"{NOTEBOOK_NAME}.log"

logger = logging.getLogger("tjeecc")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logger.info
log(f"=== SESSION START: {NOTEBOOK_NAME} ===")

# --- 5. Reproducibility record ----------------------------------------------
def _ver(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "NOT FOUND"

def _gb(n: int) -> float:
    return n / 1024 ** 3

log(f"UTC time        : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
log(f"Python          : {platform.python_version()} ({platform.platform()})")
log(f"Seed            : {SEED}")
log(f"ROOT            : {ROOT}   (from {_root_src})")
log(f"SCRATCH         : {SCRATCH}")
log(f"Log file        : {LOG_PATH}")
log(f"In Colab        : {IN_COLAB}")

log("--- package versions ---")
for dist in REQUIRED:
    log(f"  {dist:<14}: {_ver(dist)}")
for dist in ("numpy", "scipy", "pandas", "matplotlib", "jax", "jaxlib"):
    log(f"  {dist:<14}: {_ver(dist)}")

# --- 6. Hardware -------------------------------------------------------------
log("--- hardware ---")
for _label, _path in (("scratch disk", SCRATCH), ("project disk", ROOT)):
    try:
        _du = shutil.disk_usage(str(_path))
        log(f"  {_label:<13} : {_gb(_du.free):.1f} GB free of "
            f"{_gb(_du.total):.1f} GB")
    except OSError as e:
        log(f"  {_label:<13} : unreadable ({e})")

# RAM, from /proc/meminfo where available (no psutil dependency)
try:
    _mem = {}
    for _line in Path("/proc/meminfo").read_text().splitlines():
        _k, _, _v = _line.partition(":")
        _mem[_k] = int(_v.strip().split()[0]) * 1024
    log(f"  RAM           : {_gb(_mem['MemTotal']):.1f} GB total, "
        f"{_gb(_mem['MemAvailable']):.1f} GB available")
    if _gb(_mem["MemTotal"]) < 20:
        log("  NOTE: less than 20 GB RAM. Stage 2 (harvest of ~86k JSON "
            "files) and the gap calibration are the memory-hungry stages.")
except (OSError, KeyError):
    log("  RAM           : could not read /proc/meminfo (non-Linux host)")

log(f"  CPU cores     : {os.cpu_count()}")

# GPU (optional; the pipeline runs fine on CPU)
try:
    _smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    if _smi.returncode == 0 and _smi.stdout.strip():
        for _g in _smi.stdout.strip().splitlines():
            log(f"  GPU           : {_g.strip()}")
    else:
        log("  GPU           : none detected (CPU run)")
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    log("  GPU           : nvidia-smi unavailable (CPU run)")

# --- 7. JAX / NumPyro device configuration -----------------------------------
# set_host_device_count MUST run before jax initialises its backend, and it
# only affects CPU devices. On GPU, parallel chains come from
# chain_method="vectorized" instead.
MCMC_CHAIN_METHOD = "parallel"
try:
    import numpyro
    try:
        numpyro.set_host_device_count(4)
    except Exception as e:                                # already initialised
        log(f"  note: set_host_device_count skipped ({e})")

    import jax
    _devs = jax.devices()
    _backend = jax.default_backend()
    log(f"  JAX backend   : {_backend}  devices={_devs}")

    if _backend == "gpu":
        MCMC_CHAIN_METHOD = "vectorized"   # all 4 chains in one GPU kernel
        log("  MCMC plan     : chain_method='vectorized' (single GPU)")
    else:
        MCMC_CHAIN_METHOD = "parallel"     # one chain per CPU device
        log("  MCMC plan     : chain_method='parallel' (4 CPU devices)")

    # jax.default_backend() says "gpu"; numpyro.set_platform() only accepts
    # cpu / cuda / rocm / tpu / METAL. Map via the device repr: in jaxlib
    # 0.7.x the Python class is a generic pybind wrapper, but the repr is
    # reliably e.g. "CudaDevice(id=0)". This call is cosmetic - jax has
    # already initialised - and is wrapped so it can never abort the run.
    _devrepr = repr(_devs[0]).lower()
    if "cuda" in _devrepr or "nvidia" in _devrepr:
        NUMPYRO_PLATFORM = "cuda"
    elif "rocm" in _devrepr:
        NUMPYRO_PLATFORM = "rocm"
    elif "tpu" in _devrepr:
        NUMPYRO_PLATFORM = "tpu"
    else:
        NUMPYRO_PLATFORM = "cpu"
    try:
        numpyro.set_platform(NUMPYRO_PLATFORM)
        log(f"  numpyro plat  : {NUMPYRO_PLATFORM}")
    except (AssertionError, RuntimeError) as e:
        log(f"  numpyro plat  : set_platform skipped ({e})")

    # JAX defaults to float32. The gap calibration and the Fermi-Dirac
    # integrals both need float64, or you see sampler divergences that look
    # like model misspecification but are pure precision artefacts.
    jax.config.update("jax_enable_x64", True)
    log("  jax_enable_x64: True")

    # Consumer NVIDIA cards cripple float64: a T4 runs FP64 at 1/32 of FP32,
    # so a float64 NUTS fit can be SLOWER on the GPU than on 8 CPU cores.
    if _backend == "gpu":
        _kind = getattr(_devs[0], "device_kind", "")
        if any(t in _kind for t in ("T4", "P4", "P100", "V100")):
            log(f"  NOTE: {_kind} has weak FP64. With jax_enable_x64=True the "
                f"gap-calibration MCMC may run faster on CPU. Time both.")
except ImportError:
    log("  JAX/NumPyro   : not installed. The gap-calibration stage will not "
        "run; every later stage still works from the shipped posterior in "
        "data/processed/gap_posterior.npy.")

# --- 8. Input data check -----------------------------------------------------
# Raw C2DB is NOT redistributed with this repository (see README and
# download_data.py). These checks are advisory: every stage from 3 onward can
# be reproduced from the derived tables shipped in data/processed/.
RAW_EXPECTED = {
    "c2db.db":     (74_481_664,
                    "a1a96ebc00c27dafecd7be8a1ebf7f2cbdedc8160626a674855d22ff466aef2d"),
    "c2db.tar.gz": (449_561_496,
                    "e3b27bf0426ace94eb02e6ab31818c45cad5e619bbf3e1c3397fcdf2f35f9dc5"),
}
log("--- raw input files (optional; needed only for stages 1-2) ---")
for _label, (_bytes, _sha) in RAW_EXPECTED.items():
    _f = SUBDIRS["raw"] / _label
    if not _f.exists():
        log(f"  absent  {_label:<14} -> see download_data.py for how to obtain it")
    elif _f.stat().st_size != _bytes:
        log(f"  WARNING {_label:<14} is {_f.stat().st_size} bytes, expected "
            f"{_bytes}. Likely a truncated download or a different C2DB "
            f"snapshot; run 'python download_data.py --verify' before "
            f"extracting, or tarfile will fail mid-stream.")
    else:
        log(f"  OK      {_label:<14} {_gb(_f.stat().st_size):.2f} GB "
            f"(size matches the snapshot used in the paper)")

log("--- derived tables shipped with the repository ---")
_n_proc = len(list(SUBDIRS["processed"].glob("*")))
log(f"  data/processed contains {_n_proc} files")
if _n_proc == 0:
    log("  NOTE: empty. Either run the full pipeline, or restore the "
        "committed tables with:  git checkout -- data/processed")

log(f"=== BOOTSTRAP COMPLETE: {NOTEBOOK_NAME} ===\n")

# Names exported to the rest of the pipeline:
#   ROOT, SUBDIRS, SCRATCH, C2DB_TREE, RNG, SEED, log, logger,
#   LOG_PATH, MCMC_CHAIN_METHOD, IN_COLAB
