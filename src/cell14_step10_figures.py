# =============================================================================
# TJEECC - CELL 14 / PLAN STEP 10: publication figures
#
# Specification satisfies BOTH the journal requirements and Nature-style
# conventions, which are compatible:
#   TJEECC : Times New Roman, line weight 0.5-1.0 pt, max 16 x 20 cm,
#            >=118 px/cm at 16 cm width (that is >=300 dpi)
#   Nature : 600 dpi, single column 89 mm / double column 183 mm,
#            sans or serif at 5-7 pt minimum, bold lower-case panel labels
#
# We use 600 dpi, Times New Roman with a metric-compatible fallback, 8 pt
# body text (comfortably above both minima and legible when printed at
# column width), and export TIFF (submission) plus PNG (drafting).
#
# Produces Figures 2-6. Figure 1 (pipeline schematic) is drawn by hand.
# Run cells 10-13 first.
# =============================================================================

import warnings
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.font_manager   # required: mpl.font_manager is not auto-imported
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

FIGDIR = SUBDIRS["figures"]; FIGDIR.mkdir(parents=True, exist_ok=True)
PROC   = SUBDIRS["processed"]

CM = 1 / 2.54
W1, W2 = 8.9 * CM, 16.0 * CM      # single and double column, within the 16 cm cap
DPI = 600

# --- font: Times New Roman, else a metrically identical substitute -----------
# Colab ships no Times New Roman. Try to install the metrically identical
# Liberation Serif once; harmless and fast if already present.
import subprocess, shutil
if not shutil.which("fc-list"):
    pass
try:
    subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-liberation"],
                   capture_output=True, timeout=120)
    mpl.font_manager._load_fontmanager(try_read_cache=False)
except Exception:
    pass

avail = {f.name for f in mpl.font_manager.fontManager.ttflist}
for cand in ("Times New Roman", "Liberation Serif", "Nimbus Roman",
             "DejaVu Serif"):
    if cand in avail:
        SERIF = cand
        break
else:
    SERIF = "serif"
if SERIF != "Times New Roman":
    log(f"  NOTE: Times New Roman unavailable; using '{SERIF}'. "
        f"Install with: apt-get install -qq fonts-liberation  "
        f"(Liberation Serif is metrically identical to Times New Roman). "
        f"Declare the substitution in your submission letter.")

mpl.rcParams.update({
    "font.family": "serif", "font.serif": [SERIF],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 0.9,      # within 0.5-1.0 pt
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "axes.grid": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.dpi": 120,
})

def panel(ax, letter, dx=-0.19, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")

def save(fig, name):
    for ext in ("tiff", "png"):
        p = FIGDIR / f"{name}.{ext}"
        kw = dict(pil_kwargs={"compression": "tiff_lzw"}) if ext == "tiff" else {}
        fig.savefig(p, dpi=DPI, **kw)
    w_cm, h_cm = fig.get_size_inches() * 2.54
    px_per_cm = DPI / 2.54
    ok = (w_cm <= 16.01 and h_cm <= 20.01 and px_per_cm >= 118)
    log(f"  {name:<28} {w_cm:5.2f} x {h_cm:5.2f} cm  {px_per_cm:.0f} px/cm  "
        f"{'OK' if ok else '*** SPEC FAIL'}")
    plt.close(fig)

par = pd.read_parquet(PROC / "device_parameters.parquet")
dev = pd.read_parquet(PROC / "c2db_device_ready.parquet")

# Defensive: an earlier cell11 wrote this parquet BEFORE attaching the
# figure-of-merit columns. If they are absent, recompute them here rather than
# forcing a re-run of cell11.
if "I_ON" not in par.columns:
    log("  NOTE: device_parameters.parquet has no I_ON column (written by an "
        "older cell11). Recomputing figures of merit in place.")
    _c = compute_fom(Eg=par["E_mean"].to_numpy()[:, None],
                     m_dos_e=par["cbm_m_dos_file"].to_numpy()[:, None],
                     m_dos_h=par["vbm_m_dos_file"].to_numpy()[:, None],
                     m_cond_e=par["cbm_m_cond"].to_numpy()[:, None],
                     m_cond_h=par["vbm_m_cond"].to_numpy()[:, None],
                     eps_ch=par["eps_ch"].to_numpy()[:, None],
                     t_ch=par["t_ch_m"].to_numpy()[:, None])
    for _k in ("I_ON", "I_OFF", "on_off", "SS", "DIBL", "tau", "EDP"):
        par[_k] = _c[_k].ravel()


# =============================================================================
# FIGURE 2  band gap and effective mass dispersion
# =============================================================================
fig, axs = plt.subplots(1, 2, figsize=(W2, 6.2 * CM))
a = axs[0]
g = pd.to_numeric(dev["gap"], errors="coerce")
h = pd.to_numeric(dev["gap_hse"], errors="coerce")
w = pd.to_numeric(dev["gap_gw"], errors="coerce")
a.scatter(g, h, s=3, c="0.55", lw=0, label="HSE06 (n=%d)" % h.notna().sum())
m = w.notna()
a.scatter(g[m], w[m], s=7, facecolors="none", edgecolors="C3", lw=0.6,
          label="G$_0$W$_0$ (n=%d)" % m.sum())
lim = [0, max(g.max(), h.max(), w.max()) * 1.03]
a.plot(lim, lim, "k--", lw=0.7, label="1:1")
a.set(xlim=lim, ylim=lim, xlabel="PBE band gap (eV)",
      ylabel="Higher-level band gap (eV)")
a.legend(loc="upper left", handletextpad=0.5, borderpad=0.2)
panel(a, "a")

b = axs[1]
E = np.load(PROC / "gap_posterior.npy")
b.hist(E.std(axis=1), bins=40, color="0.4", lw=0)
b.axvline(0.263, color="C3", lw=0.9, ls="--")
b.text(0.263, b.get_ylim()[1] * 0.92, "  calibration\n  residual $\\sigma$",
       color="C3", fontsize=7)
b.set(xlabel="Per-material predictive s.d. of $E_\\mathrm{g}$ (eV)",
      ylabel="Number of materials")
panel(b, "b")
fig.tight_layout(pad=0.4)
save(fig, "fig2_gap_dispersion")

# =============================================================================
# FIGURE 3  I_ON vs on/off with propagated uncertainty
# =============================================================================
fig, ax = plt.subplots(figsize=(W1, 7.0 * CM))
ion, oo = par["I_ON"].to_numpy(), par["on_off"].to_numpy()
ax.scatter(oo, ion, s=4, c="0.6", lw=0)
top = np.argsort(-ion)[:30]
ax.scatter(oo[top], ion[top], s=11, facecolors="none", edgecolors="C0", lw=0.7,
           label="top 30 by $I_\\mathrm{ON}$")
ax.axvline(1e4, color="C3", lw=0.7, ls=":")
ax.text(1.15e4, ion.min() * 1.4, "IRDS HP target", color="C3", fontsize=7,
        rotation=90, va="bottom")
ax.set(xscale="log", yscale="log",
       xlabel="$I_\\mathrm{ON}/I_\\mathrm{OFF}$",
       ylabel="$I_\\mathrm{ON}$ ($\\mu$A $\\mu$m$^{-1}$)")
ax.legend(loc="lower left")
fig.tight_layout(pad=0.4)
save(fig, "fig3_ion_ioff")

# =============================================================================
# FIGURE 4  dominance matrix and rank entropy
# =============================================================================
dom = pd.read_csv(PROC / "fig4_dominance_matrix.csv", index_col=0)
ent = pd.read_csv(PROC / "step8_rank_entropy.csv")
fig, axs = plt.subplots(1, 2, figsize=(W2, 7.0 * CM),
                        gridspec_kw={"width_ratios": [1.15, 1]})
a = axs[0]
im = a.imshow(dom.to_numpy(), cmap="RdBu_r", vmin=0, vmax=1,
              interpolation="nearest")
a.set(xlabel="material $B$ (rank)", ylabel="material $A$ (rank)")
a.set_xticks([0, 9, 19, 29]); a.set_xticklabels([1, 10, 20, 30])
a.set_yticks([0, 9, 19, 29]); a.set_yticklabels([1, 10, 20, 30])
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
cb.set_label("$P(I_\\mathrm{ON}^{A} > I_\\mathrm{ON}^{B})$", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.7)
panel(a, "a")

b = axs[1]
b.hist(ent["rank_entropy"], bins=40, color="0.55", lw=0, label="all materials")
b.hist(ent.nsmallest(30, "rank_mean")["rank_entropy"], bins=20, color="C0",
       alpha=0.85, lw=0, label="top 30")
b.set(xlabel="Normalised rank entropy", ylabel="Number of materials")
b.legend(loc="upper left")
panel(b, "b")
fig.tight_layout(pad=0.4)
save(fig, "fig4_dominance_entropy")

# =============================================================================
# FIGURE 5  Sobol total-order indices
# =============================================================================
sob = pd.read_csv(PROC / "fig5_sobol_indices.csv")
LBL = {"Eg": "$E_\\mathrm{g}$", "m_dos_e": "$m_\\mathrm{d,e}$",
       "m_cond_e": "$m_\\mathrm{c,e}$", "m_dos_h": "$m_\\mathrm{d,h}$",
       "m_cond_h": "$m_\\mathrm{c,h}$", "eps": "$\\varepsilon_\\mathrm{ch}$"}
order = ["Eg", "m_cond_e", "m_dos_e", "m_cond_h", "m_dos_h", "eps"]
fig, ax = plt.subplots(figsize=(W1, 6.0 * CM))
sub = sob[sob.fom == "I_ON"].groupby("param")[["ST", "ST_conf"]].median()
y = np.arange(len(order))
vals = [sub.loc[p, "ST"] if p in sub.index else 0 for p in order]
errs = [sub.loc[p, "ST_conf"] if p in sub.index else 0 for p in order]
cols = ["C0" if v - e > 0.02 else "0.75" for v, e in zip(vals, errs)]
ax.barh(y, vals, xerr=errs, color=cols, height=0.62,
        error_kw=dict(lw=0.7, capsize=1.8, capthick=0.7))
ax.set_yticks(y); ax.set_yticklabels([LBL[p] for p in order])
ax.invert_yaxis()
ax.set(xlabel="Total-order Sobol index $S_{T}$ for $I_\\mathrm{ON}$",
       xlim=(0, 1.05))
ax.text(0.97, 0.06, "grey: CI covers zero\n(not influential)", fontsize=7,
        transform=ax.transAxes, ha="right", color="0.35")
fig.tight_layout(pad=0.4)
save(fig, "fig5_sobol")

# =============================================================================
# FIGURE 6  uncertainty amplification and rank stability vs transport regime
# =============================================================================
reg = pd.read_csv(PROC / "transport_regime_rank_stability.csv")
base = reg[reg.sigma_ln_mass == 0.20].sort_values("B_median")
fig, ax = plt.subplots(figsize=(W2 * 0.62, 6.4 * CM))
ax.plot(base.B_median, base.spread_med, "o-", ms=3.2, color="C0",
        label="$I_\\mathrm{ON}$ spread")
ax.set(xlabel="Ballisticity $B$",
       ylabel="$\\pm1\\sigma$ spread in $\\log_{10} I_\\mathrm{ON}$ (dec)")
ax2 = ax.twinx()
ax2.plot(base.B_median, base.kendall_tau, "s--", ms=3.2, color="C3",
         label="Kendall $\\tau$")
ax2.set_ylabel("Kendall $\\tau$ vs noise-free ranking", color="C3")
ax2.tick_params(axis="y", colors="C3", width=0.7)
ax2.spines["right"].set_color("C3"); ax2.spines["right"].set_linewidth(0.7)
ax2.set_ylim(0.5, 1.0)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=1)
amp = base.spread_med.iloc[0] / base.spread_med.iloc[-1]
ax.text(0.04, 0.06, f"amplification $\\times${amp:.1f}\n(analytic $\\times$4.0)",
        transform=ax.transAxes, fontsize=7)
fig.tight_layout(pad=0.4)
save(fig, "fig6_transport_regime")

log("--- figure export complete ---")
log(f"  TIFF (submission) and PNG (drafting) in {FIGDIR}")
log("  Every panel is <= 16 cm wide and 600 dpi = 236 px/cm, twice the "
    "journal minimum of 118 px/cm.")
log("  Remaining: Figure 1, the pipeline schematic, drawn by hand.")
log("[cell14] DONE\n")
