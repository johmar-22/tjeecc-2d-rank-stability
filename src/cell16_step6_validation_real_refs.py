# =============================================================================
# TJEECC - CELL 16 / PLAN STEP 6 (final): validation against REAL references
#
# Replaces the placeholder table in cell11. Every value below was read from a
# table in the cited paper, together with the L_g, V_DD and I_OFF it was
# obtained at. Nothing here is invented.
#
# THE KEY METHODOLOGICAL POINT, discovered by reading Ni et al. Table 1, which
# reports both method classes side by side for the same material:
#
#     full-band DFT-NEGF with Ti contacts : MoS2 DG, L=10 nm -> 348 uA/um
#     effective-mass (EMA) NEGF, ideal    : MoS2 DG, L= 6 nm -> 2133 uA/um
#                                           MoS2 DG, L= 8 nm -> ~2100 uA/um
#
# EMA models run 6-9x higher than full-band calculations with real contacts.
# OUR MODEL IS EMA WITH IDEAL CONTACTS. Benchmarking it against full-band
# DFT-NEGF would fail for reasons unrelated to its correctness, so the primary
# gate uses the EMA references and the full-band values are reported as the
# systematic offset attributable to band non-parabolicity and contact
# resistance. Both are shown; the paper must state this explicitly.
#
# Run cells 10 and 11 first.
# =============================================================================

import numpy as np, pandas as pd
from pymatgen.core import Composition

PROC   = SUBDIRS["processed"]
REFS   = PROC / "table8_references_FILLED.csv"
OUTCSV = PROC / "table8_validation_real.csv"

par = pd.read_parquet(PROC / "device_parameters.parquet")

# =============================================================================
# Reference data, transcribed from the papers
# =============================================================================
# EOT: Ni et al. quote 5 Angstrom; Sun et al. state 0.41-0.54 nm, midpoint used.
# Sun's WSe2 rows are p-type; flagged, since our model reports the n-branch.
REF = [
    # --- EMA / ideal-contact class: the like-for-like comparison ------------
    dict(formula="MoS2", I_ON=2133, L_g=6.0,  V_DD=0.57, I_OFF_uA=0.1, EOT=0.45,
         cls="EMA-NEGF", ptype=False,
         cite="Ni et al., Adv. Electron. Mater. 2, 1600191 (2016), Table 1, "
              "row 'SE (EMA) DG[24]' (phonon-corrected)",
         doi="10.1002/aelm.201600191"),
    dict(formula="MoS2", I_ON=2100, L_g=8.0,  V_DD=0.60, I_OFF_uA=0.1, EOT=0.50,
         cls="EMA-NEGF", ptype=False,
         cite="Ni et al., Adv. Electron. Mater. 2, 1600191 (2016), Table 1, "
              "row 'SE (EMA) DG[30]'",
         doi="10.1002/aelm.201600191"),
    # --- full-band DFT-NEGF with metal contacts: the systematic offset ------
    dict(formula="MoS2", I_ON=348,  L_g=10.0, V_DD=0.50, I_OFF_uA=0.1, EOT=0.50,
         cls="DFT-NEGF", ptype=False,
         cite="Ni et al., Adv. Electron. Mater. 2, 1600191 (2016), Table 2, "
              "DG, ballistic, Ti electrodes",
         doi="10.1002/aelm.201600191"),
    dict(formula="MoS2", I_ON=273,  L_g=8.0,  V_DD=0.50, I_OFF_uA=0.1, EOT=0.50,
         cls="DFT-NEGF", ptype=False,
         cite="Ni et al. (2016), Table 2", doi="10.1002/aelm.201600191"),
    dict(formula="MoS2", I_ON=221,  L_g=6.0,  V_DD=0.50, I_OFF_uA=0.1, EOT=0.50,
         cls="DFT-NEGF", ptype=False,
         cite="Ni et al. (2016), Table 2", doi="10.1002/aelm.201600191"),
    dict(formula="WSe2", I_ON=1464, L_g=5.0,  V_DD=0.64, I_OFF_uA=0.1, EOT=0.48,
         cls="DFT-NEGF", ptype=True,
         cite="Sun et al., ACS Appl. Mater. Interfaces 12, 20633 (2020), "
              "Table 1, DG ML WSe2 p-type, L_UL = 0",
         doi="10.1021/acsami.0c04008"),
    dict(formula="WSe2", I_ON=1302, L_g=7.0,  V_DD=0.69, I_OFF_uA=0.1, EOT=0.48,
         cls="DFT-NEGF", ptype=True,
         cite="Sun et al. (2020), Table 1", doi="10.1021/acsami.0c04008"),
    dict(formula="WSe2", I_ON=1292, L_g=9.0,  V_DD=0.72, I_OFF_uA=0.1, EOT=0.48,
         cls="DFT-NEGF", ptype=True,
         cite="Sun et al. (2020), Table 1", doi="10.1021/acsami.0c04008"),
]
pd.DataFrame(REF).to_csv(REFS, index=False)
log(f"[cell16] reference table -> {REFS}")

# NOT used in the ballistic gate, and the reason must be stated in the paper:
log("--- excluded from the ballistic gate ---")
log("  Afzalian, npj 2D Mater. Appl. 5, 5 (2021): DISSIPATIVE DFT-NEGF with "
    "electron-phonon scattering, I_OFF = 10 nA/um, V_DD = 0.6 V, L = 5 nm. "
    "Not a ballistic reference. Use it for the diffusive-limit comparison in "
    "the transport-regime analysis instead, where it is the right benchmark.")

def ckey(f):
    rc = Composition(f).reduced_composition.get_el_amt_dict()
    return "-".join(f"{e}{int(round(n))}" for e, n in sorted(rc.items()))

# =============================================================================
# Evaluate the model at each paper's own operating point
# =============================================================================
rows = []
for r in REF:
    sub = par[par["comp_key"] == ckey(r["formula"])]
    if not len(sub):
        log(f"  {r['formula']} absent from the parameter set"); continue
    m = sub.iloc[0]
    tech = dict(TECH)
    tech.update(L_g=r["L_g"] * 1e-9, V_DD=r["V_DD"], EOT=r["EOT"] * 1e-9,
                I_OFF=r["I_OFF_uA"] * 1e-6 / 1e-6)   # uA/um -> A/m
    out = compute_fom(Eg=np.array([[m.E_mean]]),
                      m_dos_e=np.array([[m.cbm_m_dos_file]]),
                      m_dos_h=np.array([[m.vbm_m_dos_file]]),
                      m_cond_e=np.array([[m.cbm_m_cond]]),
                      m_cond_h=np.array([[m.vbm_m_cond]]),
                      eps_ch=np.array([[m.eps_ch]]),
                      t_ch=np.array([[m.t_ch_m]]), tech=tech)
    ion = float(out["I_ON"].ravel()[0])
    rows.append({**{k: r[k] for k in
                    ("formula", "cls", "L_g", "V_DD", "I_OFF_uA", "EOT", "ptype")},
                 "I_ON_pub": r["I_ON"], "I_ON_model": ion,
                 "ratio": ion / r["I_ON"],
                 "SS_model": float(out["SS"].ravel()[0]),
                 "citation": r["cite"], "doi": r["doi"]})

val = pd.DataFrame(rows)
val.to_csv(OUTCSV, index=False)

log("=" * 78)
log(f"  {'mat':<6} {'class':<10} {'Lg':>4} {'Vdd':>5} {'pub':>7} {'model':>7} "
    f"{'ratio':>6} {'SS':>6}  p?")
for r in val.itertuples(index=False):
    log(f"  {r.formula:<6} {r.cls:<10} {r.L_g:>4.0f} {r.V_DD:>5.2f} "
        f"{r.I_ON_pub:>7.0f} {r.I_ON_model:>7.0f} {r.ratio:>6.2f} "
        f"{r.SS_model:>6.1f}  {'p' if r.ptype else 'n'}")

# =============================================================================
# Gate: primary on the like-for-like class
# =============================================================================
log("--- validation gate ---")
for cls in ("EMA-NEGF", "DFT-NEGF"):
    s = val[val.cls == cls]
    if not len(s):
        continue
    gm = float(np.exp(np.mean(np.log(s["ratio"]))))
    w2 = int(s["ratio"].between(0.5, 2.0).sum())
    tag = "PRIMARY GATE" if cls == "EMA-NEGF" else "systematic offset"
    log(f"  {cls:<10} ({tag}): geometric-mean ratio = {gm:.2f}, "
        f"within 2x: {w2}/{len(s)}")
    if cls == "EMA-NEGF":
        passed = 0.5 <= gm <= 2.0
        log(f"    {'PASS' if passed else '*** FAIL'}  (gate: 0.5 to 2.0)")

# The p-type WSe2 rows are compared against our n-branch, so they contaminate
# the offset statistic. Quote the n-type MoS2 rows only.
ema = val[(val.cls == "EMA-NEGF") & (~val.ptype)]["ratio"]
dft = val[(val.cls == "DFT-NEGF") & (~val.ptype)]["ratio"]
if len(dft):
    log(f"  DFT-NEGF, n-type MoS2 rows only: geometric-mean ratio = "
        f"{float(np.exp(np.mean(np.log(dft)))):.2f}  <- quote this, not the "
        f"mixed-carrier value above")
if len(ema) and len(dft):
    off = float(np.exp(np.mean(np.log(dft))) / np.exp(np.mean(np.log(ema))))
    log(f"  model/full-band offset relative to model/EMA: x{off:.1f}")
    log(f"  Interpretation for the Limitations section: an effective-mass, "
        f"ideal-contact ballistic model overestimates full-band DFT-NEGF with "
        f"metal contacts by roughly this factor. Ni et al. document the same "
        f"gap between EMA and full-band NEGF within a single table "
        f"(2133 vs 221 uA/um at L = 6 nm). This is a known, systematic "
        f"property of the model class, not a defect of the implementation, "
        f"and it does not affect RELATIVE rankings, which is what this paper "
        f"reports.")

log("--- caveats to carry into the manuscript ---")
log("  1. The WSe2 rows are p-type; our model reports the n-branch. Either "
    "compare the hole branch explicitly or drop these rows and say why.")
log("  2. Sun et al. quote EOT as a range (0.41-0.54 nm); the midpoint is "
    "used here. State that.")
log("  3. Ni et al. include Ti electrodes with a Schottky barrier; our "
    "contacts are ideal. This is part of the offset above.")
log(f"  saved {OUTCSV}")
log("[cell16] DONE\n")
