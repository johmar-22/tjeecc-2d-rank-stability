# =============================================================================
# TJEECC - CELL 15
#   PART 1: Sobol split by regime (saturated vs gap-sensitive) -> Figure 5
#   PART 2: literature reference table for the Step 6 validation gate
#
# WHY PART 1 EXISTS. cell13 selected its 20 representative materials by I_ON
# quantile. Almost all of them have a calibrated gap above ~1.4 eV, where
# I_OFF is pinned at the spec and the gap cannot influence anything. The
# resulting S_T(E_g) = 0.000 is an artefact of that selection, not a physical
# statement, and publishing it in a band-gap paper invites an obvious
# objection. Splitting the sample by regime turns the artefact into the figure
# that demonstrates the paper's mechanism.
#
# Run cells 10-13 first.
# =============================================================================

import json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.font_manager
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

PROC   = SUBDIRS["processed"]
FIGDIR = SUBDIRS["figures"]
CENSUS = PROC / "gap_sensitivity_census.csv"
OUT    = PROC / "fig5_sobol_by_regime.csv"

SIG_LN_M, SIG_LN_EPS = 0.20, 0.19
NSOB, NREP = 1024, 20
NAMES = ["Eg", "m_dos_e", "m_cond_e", "m_dos_h", "m_cond_h", "eps"]

par = pd.read_parquet(PROC / "device_parameters.parquet").reset_index(drop=True)

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

ids = pd.read_csv(PROC / "gap_posterior_uids.csv")
Ep  = np.load(PROC / "gap_posterior.npy")
gi  = np.array([{u: i for i, u in enumerate(ids["uid"])}[u] for u in par["uid"]])

# regime membership, from the census
cen = pd.read_csv(CENSUS)
gap_sens = set(cen.loc[cen["spr_gap_on_off"] > 0.30, "uid"]) if \
           "spr_gap_on_off" in cen.columns else set()
par["gap_sensitive"] = par["uid"].isin(gap_sens)
log(f"[cell15] gap-sensitive {par['gap_sensitive'].sum()} / {len(par)}")
if par["gap_sensitive"].sum() < 8:
    log("  WARNING: too few gap-sensitive materials for a stable Sobol "
        "estimate. Widen the criterion (e.g. spread > 0.15 dec) and say so.")

def sobol_for(idx, label):
    out = []
    for mi in idx:
        r = par.iloc[mi]
        sd = float(Ep[gi[mi]].std())
        prob = {"num_vars": 6, "names": NAMES, "bounds": [
            [max(0.05, r.E_mean - 2*sd), r.E_mean + 2*sd],
            [r.cbm_m_dos_file*np.exp(-2*SIG_LN_M), r.cbm_m_dos_file*np.exp(2*SIG_LN_M)],
            [r.cbm_m_cond   *np.exp(-2*SIG_LN_M), r.cbm_m_cond   *np.exp(2*SIG_LN_M)],
            [r.vbm_m_dos_file*np.exp(-2*SIG_LN_M), r.vbm_m_dos_file*np.exp(2*SIG_LN_M)],
            [r.vbm_m_cond   *np.exp(-2*SIG_LN_M), r.vbm_m_cond   *np.exp(2*SIG_LN_M)],
            [r.eps_ch*np.exp(-2*SIG_LN_EPS), r.eps_ch*np.exp(2*SIG_LN_EPS)]]}
        X = sobol_sample.sample(prob, NSOB, calc_second_order=False)
        res = compute_fom(Eg=X[:, 0][:, None], m_dos_e=X[:, 1][:, None],
                          m_cond_e=X[:, 2][:, None], m_dos_h=X[:, 3][:, None],
                          m_cond_h=X[:, 4][:, None], eps_ch=X[:, 5][:, None],
                          t_ch=np.full((len(X), 1), r.t_ch_m))
        for fom in ("I_ON", "on_off"):
            y = np.log10(np.maximum(res[fom].ravel(), 1e-300))
            if y.std() < 1e-12:
                for nm in NAMES:
                    out.append({"regime": label, "fom": fom, "param": nm,
                                "ST": 0.0, "ST_conf": 0.0, "uid": r.uid})
                continue
            Si = sobol_analyze.analyze(prob, y, calc_second_order=False,
                                       print_to_console=False)
            for j, nm in enumerate(NAMES):
                out.append({"regime": label, "fom": fom, "param": nm,
                            "ST": Si["ST"][j], "ST_conf": Si["ST_conf"][j],
                            "uid": r.uid})
    return out

rows = []
for flag, label in ((False, "saturated"), (True, "gap-sensitive")):
    pool = np.where(par["gap_sensitive"].to_numpy() == flag)[0]
    if len(pool) == 0:
        continue
    q = np.linspace(0.02, 0.98, min(NREP, len(pool)))
    pick = pool[np.argsort(-par["I_ON"].to_numpy()[pool])]
    pick = pick[np.clip((q * (len(pick) - 1)).astype(int), 0, len(pick) - 1)]
    log(f"  Sobol on {len(pick)} {label} materials...")
    rows += sobol_for(pick, label)

sob = pd.DataFrame(rows)
sob.to_csv(OUT, index=False)

log("--- median total-order indices ---")
for fom in ("I_ON", "on_off"):
    for label in ("saturated", "gap-sensitive"):
        s = sob[(sob.fom == fom) & (sob.regime == label)]
        if not len(s):
            continue
        g = s.groupby("param")[["ST", "ST_conf"]].median()
        top = g["ST"].idxmax()
        log(f"  {fom:<7} {label:<14} dominant={top:<10} "
            f"ST={g.loc[top,'ST']:.3f}   E_g ST={g.loc['Eg','ST']:.3f}")

# =============================================================================
# FIGURE 5 (revised): two panels, saturated vs gap-sensitive
# =============================================================================
for cand in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
    if cand in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        SERIF = cand; break
else:
    SERIF = "serif"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": [SERIF], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

CM = 1/2.54
LBL = {"Eg": "$E_\\mathrm{g}$", "m_dos_e": "$m_\\mathrm{d,e}$",
       "m_cond_e": "$m_\\mathrm{c,e}$", "m_dos_h": "$m_\\mathrm{d,h}$",
       "m_cond_h": "$m_\\mathrm{c,h}$", "eps": "$\\varepsilon_\\mathrm{ch}$"}
ORDER = ["Eg", "m_cond_e", "m_dos_e", "m_cond_h", "m_dos_h", "eps"]

fig, axs = plt.subplots(1, 2, figsize=(16.0*CM, 6.0*CM), sharex=True)
for ax, label, letter in zip(axs, ("saturated", "gap-sensitive"), "ab"):
    s = sob[(sob.fom == "on_off") & (sob.regime == label)]
    if not len(s):
        ax.set_axis_off(); continue
    g = s.groupby("param")[["ST", "ST_conf"]].median()
    y = np.arange(len(ORDER))
    v = [g.loc[p, "ST"] if p in g.index else 0 for p in ORDER]
    e = [g.loc[p, "ST_conf"] if p in g.index else 0 for p in ORDER]
    c = ["C0" if vi - ei > 0.02 else "0.75" for vi, ei in zip(v, e)]
    ax.barh(y, v, xerr=e, color=c, height=0.62,
            error_kw=dict(lw=0.7, capsize=1.8, capthick=0.7))
    ax.set_yticks(y); ax.set_yticklabels([LBL[p] for p in ORDER])
    ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel("Total-order Sobol index $S_T$ for $I_\\mathrm{ON}/I_\\mathrm{OFF}$")
    n_here = s["uid"].nunique()
    ax.set_title(f"{label} ($n$ = {n_here})", fontsize=8)
    ax.text(-0.30, 1.06, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
fig.tight_layout(pad=0.4)
for ext in ("tiff", "png"):
    kw = dict(pil_kwargs={"compression": "tiff_lzw"}) if ext == "tiff" else {}
    fig.savefig(FIGDIR / f"fig5_sobol_by_regime.{ext}", dpi=600, **kw)
plt.close(fig)
log(f"  Figure 5 (revised) -> {FIGDIR/'fig5_sobol_by_regime.tiff'}")
log("  Caption to write: 'The band gap has no influence once E_g exceeds the "
    "ambipolar limit (a), and becomes the dominant term below it (b). "
    "Effective mass governs I_ON in both regimes.'")

# =============================================================================
# PART 2: reference table template for Table 8
# =============================================================================
TEMPLATE = PROC / "table8_references_TEMPLATE.csv"
pd.DataFrame([
    {"formula": "MoS2",  "I_ON_uA_um": "", "L_g_nm": "", "V_DD_V": "",
     "I_OFF_nA_um": "", "gate_geometry": "", "method": "DFT-NEGF ballistic",
     "citation": "", "doi": "", "table_or_figure": "", "notes": ""},
    {"formula": "WSe2",  "I_ON_uA_um": "", "L_g_nm": "", "V_DD_V": "",
     "I_OFF_nA_um": "", "gate_geometry": "", "method": "DFT-NEGF ballistic",
     "citation": "", "doi": "", "table_or_figure": "", "notes": ""},
    {"formula": "HfS2",  "I_ON_uA_um": "", "L_g_nm": "", "V_DD_V": "",
     "I_OFF_nA_um": "", "gate_geometry": "", "method": "DFT-NEGF ballistic",
     "citation": "", "doi": "", "table_or_figure": "", "notes": ""},
    {"formula": "ZrS2",  "I_ON_uA_um": "", "L_g_nm": "", "V_DD_V": "",
     "I_OFF_nA_um": "", "gate_geometry": "", "method": "DFT-NEGF ballistic",
     "citation": "", "doi": "", "table_or_figure": "", "notes": ""},
    {"formula": "P",     "I_ON_uA_um": "", "L_g_nm": "", "V_DD_V": "",
     "I_OFF_nA_um": "", "gate_geometry": "", "method": "DFT-NEGF ballistic",
     "citation": "", "doi": "", "table_or_figure": "", "notes": ""},
]).to_csv(TEMPLATE, index=False)
log(f"[cell15] reference template -> {TEMPLATE}")
log("  Fill EVERY column from the paper itself. A row missing L_g, V_DD or "
    "I_OFF cannot be used: without the off-current specification you are "
    "comparing at different threshold voltages, which is the error that "
    "produces order-of-magnitude discrepancies.")
log("[cell15] DONE\n")
