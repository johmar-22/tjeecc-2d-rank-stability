# =============================================================================
# TJEECC - CELL 6: diagnose the impossible JARVIS<->C2DB scale ratios
#
# Cell 5 returned median ratios of 344 (electron mass), 286 (hole mass) and
# 0.287 (permittivity). Real cross-code disagreement is tens of percent, not
# factors of 300. These are unit/definition mismatches and must be resolved
# before any of them is used as a sigma in Step 4.
#
# Hypotheses under test:
#   H1  JARVIS avg_elec_mass is not in units of m_0, or is not a band mass.
#   H2  JARVIS epsx is vacuum-diluted by the supercell height L, so the
#       correct comparison is polarizability, not permittivity:
#           alpha_jarvis = (eps - 1) * L / (4*pi)   vs   C2DB alphax_el
#
# Requires cell00_bootstrap.py.
# =============================================================================

import json
import numpy as np, pandas as pd
from pymatgen.core import Composition

MATCH_PQ    = SUBDIRS["processed"] / "c2db_jarvis_matched.parquet"
SCALES_JSON = SUBDIRS["processed"] / "uncertainty_scales.json"

mt = pd.read_parquet(MATCH_PQ)
mt = mt[mt["jid"].notna()].copy()
jd = json.loads((SUBDIRS["jarvis"] / "dft_2d.json").read_text())
jdf = pd.DataFrame(jd).set_index("jid")

for c in ("avg_elec_mass", "avg_hole_mass", "epsx", "epsy", "epsz",
          "mbj_bandgap", "optb88vdw_bandgap"):
    if c in jdf.columns:
        jdf[c] = pd.to_numeric(jdf[c], errors="coerce")

# supercell height L from the JARVIS lattice (needed for H2)
def cheight(row):
    try:
        return float(np.linalg.norm(np.array(row["atoms"]["lattice_mat"])[2]))
    except Exception:
        return np.nan
jdf["L_c"] = pd.DataFrame(jd).set_index("jid").apply(cheight, axis=1)

for c in ("avg_elec_mass", "avg_hole_mass", "epsx", "epsy", "L_c"):
    if c in jdf.columns:
        mt[f"j_{c}"] = mt["jid"].map(jdf[c])

# =============================================================================
# H1: what ARE the JARVIS mass numbers?
# =============================================================================
log("=" * 70)
log("H1: JARVIS effective-mass units / definition")

for col, c2 in (("j_avg_elec_mass", "cbm_m_dos_file"),
                ("j_avg_hole_mass", "vbm_m_dos_file")):
    s = pd.to_numeric(mt[col], errors="coerce").dropna()
    log(f"--- {col} ---")
    log(f"  n={len(s)}  min={s.min():.4g}  p25={s.quantile(.25):.4g}  "
        f"median={s.median():.4g}  p75={s.quantile(.75):.4g}  max={s.max():.4g}")
    log(f"  negative values: {(s<0).sum()}   |values|>100: {(s.abs()>100).sum()}")
    c = pd.to_numeric(mt[c2], errors="coerce").dropna()
    log(f"  C2DB {c2}: median={c.median():.4g}  range=[{c.min():.4g}, {c.max():.4g}]")

# Ground truth check on materials whose effective masses are textbook values.
# Monolayer MoS2: m_e* ~ 0.45-0.48 m_0, m_h* ~ 0.55-0.65 m_0 (PBE).
log("--- ground-truth spot check (literature m* in m_0) ---")
LIT = {"MoS2": (0.47, 0.60), "WS2": (0.31, 0.42), "WSe2": (0.34, 0.44),
       "MoSe2": (0.55, 0.64), "BN": (0.90, 1.10)}
def ckey(f):
    rc = Composition(f).reduced_composition.get_el_amt_dict()
    return "-".join(f"{e}{int(round(n))}" for e, n in sorted(rc.items()))

log(f"  {'mat':<7} {'lit m_e':>8} {'C2DB m_dos':>11} {'JARVIS avg':>11} "
    f"{'J/C2DB':>9} {'J/lit':>8}")
for f, (me, mh) in LIT.items():
    sub = mt[mt["comp_key"] == ckey(f)] if "comp_key" in mt else mt.iloc[0:0]
    if not len(sub):
        log(f"  {f:<7} not in matched set")
        continue
    r = sub.iloc[0]
    c2 = float(r.get("cbm_m_dos_file", np.nan))
    jv = float(r.get("j_avg_elec_mass", np.nan))
    log(f"  {f:<7} {me:>8.2f} {c2:>11.4g} {jv:>11.4g} "
        f"{jv/c2 if c2 else np.nan:>9.3g} {jv/me:>8.3g}")

log("  READ THIS: if C2DB m_dos sits near the literature value and the JARVIS "
    "column is 2-3 orders larger, JARVIS avg_elec_mass is NOT a band mass in "
    "m_0 for this dataset. It cannot be used to set sigma on m*. "
    "Do not 'rescale' it to force agreement; that would be fitting a unit "
    "conversion to make a number look right.")

# =============================================================================
# H2: permittivity is vacuum-diluted; compare polarizability instead
# =============================================================================
log("=" * 70)
log("H2: vacuum dilution of the JARVIS dielectric constant")

# In a periodic supercell of height L containing a slab of polarizability
# alpha, the computed in-plane permittivity is eps = 1 + 4*pi*alpha/L.
# C2DB reports alpha directly, so invert JARVIS eps back to alpha and compare
# like with like.
mt["j_eps_ip"] = 0.5 * (mt["j_epsx"] + mt["j_epsy"])
mt["j_alpha_ip"] = (mt["j_eps_ip"] - 1.0) * mt["j_L_c"] / (4.0 * np.pi)
mt["c_alpha_ip"] = 0.5 * (mt["alphax_el"] + mt["alphay_el"])
mt["c_eps_ip"] = 1.0 + 4.0 * np.pi * mt["c_alpha_ip"] / mt["thickness"]

d = mt[["j_eps_ip", "j_L_c", "j_alpha_ip", "c_alpha_ip", "c_eps_ip",
        "thickness"]].replace([np.inf, -np.inf], np.nan).dropna()
d = d[(d["j_alpha_ip"] > 0) & (d["c_alpha_ip"] > 0)]
log(f"  usable pairs: {len(d)}")
log(f"  JARVIS supercell height L: median={d['j_L_c'].median():.1f} A  "
    f"range=[{d['j_L_c'].min():.1f}, {d['j_L_c'].max():.1f}]")
log(f"  C2DB slab thickness t    : median={d['thickness'].median():.1f} A")
log(f"  predicted dilution t/L   : {(d['thickness']/d['j_L_c']).median():.3f}")
log(f"  observed eps ratio J/C2DB: {(d['j_eps_ip']/d['c_eps_ip']).median():.3f}")
log("  If those last two agree, H2 is confirmed: the 0.287 was a supercell "
    "convention, not a physical disagreement.")

lr = np.log(d["j_alpha_ip"] / d["c_alpha_ip"])
log("--- polarizability, the like-for-like comparison ---")
log(f"  n={len(lr)}  median ratio={np.exp(np.median(lr)):.3f}  "
    f"sigma_ln={lr.std(ddof=1):.3f}  "
    f"robust sigma_ln={np.median(np.abs(lr-np.median(lr)))*1.4826:.3f}")
log("  A median ratio near 1 with sigma_ln of order 0.1-0.3 is a credible "
    "cross-code uncertainty and IS usable for Step 4.")

# =============================================================================
# 3. Rewrite the scales file with only defensible entries
# =============================================================================
log("=" * 70)
scales = json.loads(SCALES_JSON.read_text()) if SCALES_JSON.exists() else {}

# Purge the two that failed physical plausibility.
for k in ("electron_mass_c2db_vs_jarvis", "hole_mass_c2db_vs_jarvis",
          "permittivity_c2db_vs_jarvis"):
    if k in scales:
        scales[k]["REJECTED"] = ("implausible ratio; unit or convention "
                                 "mismatch, not cross-code uncertainty")

if len(lr) >= 20:
    scales["polarizability_c2db_vs_jarvis"] = {
        "n": int(len(lr)),
        "median_ratio": float(np.exp(np.median(lr))),
        "sigma_ln": float(lr.std(ddof=1)),
        "mad_ln": float(np.median(np.abs(lr - np.median(lr))) * 1.4826),
        "note": "JARVIS eps inverted to polarizability via alpha=(eps-1)L/4pi "
                "to remove supercell vacuum dilution before comparison",
    }

# Effective mass has no surviving empirical handle. Declare the assumption
# rather than hiding it; Step 4 must sweep it.
scales["effective_mass_ASSUMED"] = {
    "sigma_ln": 0.20,
    "basis": "assumed, not measured. JARVIS avg_elec_mass was rejected (see "
             "above). 0.20 corresponds to about 20% 1-sigma spread, the order "
             "of magnitude reported for effective-mass differences between "
             "semilocal functionals.",
    "REQUIRED_ACTION": "state explicitly in Methods as an assumption and "
                       "report results at sigma_ln = 0.10, 0.20, 0.40 as a "
                       "robustness check",
}
SCALES_JSON.write_text(json.dumps(scales, indent=2))
log(f"  rewrote {SCALES_JSON}")
log("--- final Step 4 inputs ---")
for k, v in scales.items():
    tag = "REJECTED" if "REJECTED" in v else ("ASSUMED" if "ASSUMED" in k else "measured")
    log(f"  {k:<38} [{tag}]  sigma_ln={v.get('sigma_ln', float('nan')):.3f}")
log("[cell6] DONE\n")
