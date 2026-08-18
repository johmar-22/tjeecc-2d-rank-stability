# =============================================================================
# TJEECC - CELL 18: FIGURE 1, the pipeline schematic
#
# The last missing figure. Drawn programmatically so it inherits the same
# journal specification as the rest (600 dpi, Times New Roman or a metric
# substitute, 0.5-1.0 pt strokes, <= 16 x 20 cm) and so the counts shown on it
# are read from the real files rather than typed by hand and left stale.
#
# Run after cells 10-15.
# =============================================================================

import json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROC, FIGDIR = SUBDIRS["processed"], SUBDIRS["figures"]
CM, DPI = 1 / 2.54, 600

for cand in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
    if cand in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        SERIF = cand; break
else:
    SERIF = "serif"
mpl.rcParams.update({"font.family": "serif", "font.serif": [SERIF],
                     "mathtext.fontset": "stix", "font.size": 7.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# --- live counts, so the figure can never go stale --------------------------
dev = pd.read_parquet(PROC / "c2db_device_ready.parquet")
par = pd.read_parquet(PROC / "device_parameters.parquet")
ids = pd.read_csv(PROC / "gap_posterior_uids.csv")
cen = pd.read_csv(PROC / "gap_sensitivity_census.csv")
n_dev   = len(dev)
n_par   = len(par)
n_anch  = int(ids["has_gw_anchor"].sum()) if "has_gw_anchor" in ids else 0
n_gsens = int((cen["spr_gap_on_off"] > 0.30).sum())
try:
    n_match = int(pd.read_parquet(PROC / "c2db_jarvis_matched.parquet")["jid"].notna().sum())
except Exception:
    n_match = 0
log(f"[cell18] counts: device-ready {n_dev}, full params {n_par}, "
    f"G0W0 anchors {n_anch}, JARVIS matches {n_match}, gap-sensitive {n_gsens}")

# --- layout ------------------------------------------------------------------
BOXES = [
    # (x, y, w, h, title, body, colour)
    (0.02, 0.62, 0.20, 0.30, "C2DB",
     f"16,905 entries\nPBE / HSE06 / G$_0$W$_0$\nmasses, polarizability", "#e8eef7"),
    (0.02, 0.16, 0.20, 0.30, "JARVIS-DFT 2D",
     f"1,103 monolayers\n{n_match} structure matches\n$\\sigma$ for $\\varepsilon$", "#e8eef7"),
    (0.28, 0.40, 0.20, 0.36, "Screening",
     f"stability, gap > 0.3 eV\nnon-magnetic\nparabolic bands\n$\\rightarrow$ {n_dev} materials",
     "#eef3e8"),
    (0.54, 0.62, 0.20, 0.30, "Gap calibration",
     f"$E_\\mathrm{{g}}$ ~ $a$ + $b_1E_\\mathrm{{PBE}}$ + $b_2E_\\mathrm{{HSE}}$\n"
     f"{n_anch} G$_0$W$_0$ anchors\n10-fold coverage check", "#f7efe8"),
    (0.54, 0.16, 0.20, 0.30, "Uncertainty",
     "$E_\\mathrm{g}$: measured\n$\\varepsilon$: measured\n$m^*$: ASSUMED", "#f7efe8"),
    (0.80, 0.40, 0.18, 0.36, "Device model",
     f"ballistic, ambipolar\ndouble gate\n$L_\\mathrm{{g}}$ = 12 nm\n$\\rightarrow$ {n_par} materials",
     "#f7e8ee"),
]
ARROWS = [(0.22, 0.77, 0.28, 0.62), (0.22, 0.31, 0.28, 0.52),
          (0.48, 0.62, 0.54, 0.77), (0.48, 0.52, 0.54, 0.31),
          (0.74, 0.77, 0.80, 0.62), (0.74, 0.31, 0.80, 0.52)]

fig = plt.figure(figsize=(16.0 * CM, 7.4 * CM))
ax = fig.add_axes([0, 0.16, 1, 0.84]); ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

for x, y, w, h, title, body, col in BOXES:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                lw=0.7, ec="0.25", fc=col, transform=ax.transAxes))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
            fontsize=8, fontweight="bold", transform=ax.transAxes)
    ax.text(x + w / 2, y + h - 0.115, body, ha="center", va="top",
            fontsize=7, linespacing=1.45, transform=ax.transAxes)

for x0, y0, x1, y1 in ARROWS:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color="0.3", shrinkA=0, shrinkB=0))

# --- output strip ------------------------------------------------------------
axb = fig.add_axes([0, 0, 1, 0.17]); axb.set_axis_off()
axb.set_xlim(0, 1); axb.set_ylim(0, 1)
OUT = [(0.03, "Monte Carlo", "$10^3$ draws per material"),
       (0.28, "Rank stability", "dominance, Kendall $\\tau$, entropy"),
       (0.53, "Sensitivity", f"Sobol; $E_\\mathrm{{g}}$ acts in {n_gsens}/{n_par}"),
       (0.78, "Robust rule", "10th-percentile lower bound")]
for x, t, b in OUT:
    axb.add_patch(FancyBboxPatch((x, 0.18), 0.19, 0.62,
                                 boxstyle="round,pad=0.01,rounding_size=0.04",
                                 lw=0.7, ec="0.25", fc="#f2f2f2", transform=axb.transAxes))
    axb.text(x + 0.095, 0.63, t, ha="center", va="center", fontsize=7.5,
             fontweight="bold", transform=axb.transAxes)
    axb.text(x + 0.095, 0.36, b, ha="center", va="center", fontsize=6.8,
             transform=axb.transAxes)
for x in (0.22, 0.47, 0.72):
    axb.add_patch(FancyArrowPatch((x, 0.49), (x + 0.06, 0.49), transform=axb.transAxes,
                                  arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.3"))

for ext in ("tiff", "png"):
    kw = dict(pil_kwargs={"compression": "tiff_lzw"}) if ext == "tiff" else {}
    fig.savefig(FIGDIR / f"fig1_pipeline.{ext}", dpi=DPI, **kw)
w_cm, h_cm = fig.get_size_inches() * 2.54
log(f"  fig1_pipeline  {w_cm:.2f} x {h_cm:.2f} cm  {DPI/2.54:.0f} px/cm  "
    f"{'OK' if w_cm <= 16.01 and h_cm <= 20.01 else '*** SPEC FAIL'}")
plt.close(fig)
log("  Caption: 'Analysis pipeline. Materials data from C2DB and JARVIS-DFT "
    "are screened for stability and band parabolicity, the band gap is "
    "calibrated against G0W0 anchors with cross-validated predictive "
    "intervals, and the resulting uncertainty is propagated through a "
    "ballistic ambipolar double-gate device model to rank-stability and "
    "sensitivity metrics.'")
log("[cell18] DONE\n")
