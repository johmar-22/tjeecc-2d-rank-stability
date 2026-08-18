# =============================================================================
# TJEECC - CELL 9: is the G0W0 anchor set representative of the full set?
#
# The 10-fold CV in cell08 validated INTERPOLATION inside the 129-material
# anchor cloud. It cannot validate EXTRAPOLATION to the other 956. Expensive
# GW calculations are not run on a random sample, so a selection effect is
# likely and would bias every downstream gap.
#
# Symptom that prompted this: the fitted map predicts ~2.90 eV at the median
# (PBE, HSE) of the full set, against a G0W0 median of 2.266 eV among the
# anchors, and ~3.40 eV for MoS2 where literature G0W0 is 2.6-2.8 eV.
#
# Requires cell00_bootstrap.py.
# =============================================================================

import json
import numpy as np, pandas as pd
from scipy import stats
from pymatgen.core import Composition

DEVICE_PQ = SUBDIRS["processed"] / "c2db_device_ready.parquet"
POST_NPY  = SUBDIRS["processed"] / "gap_posterior.npy"
POST_IDS  = SUBDIRS["processed"] / "gap_posterior_uids.csv"
OUT_CSV   = SUBDIRS["processed"] / "step4_anchor_representativeness.csv"

dev = pd.read_parquet(DEVICE_PQ)
ids = pd.read_csv(POST_IDS)
E    = np.load(POST_NPY)                       # (N, 1000)
d = dev.set_index("uid").loc[ids["uid"]].reset_index()
anchor = ids["has_gw_anchor"].to_numpy().astype(bool)
d["E_pred"] = E.mean(axis=1)
d["E_sd"]   = E.std(axis=1)
log(f"[cell9] N={len(d):,}  anchors={anchor.sum()}  non-anchors={(~anchor).sum()}")

# =============================================================================
# 1. Are the anchors drawn from the same (PBE, HSE) distribution?
# =============================================================================
log("--- distribution comparison: anchors vs non-anchors ---")
rows = []
for col, label in (("gap", "PBE gap"), ("gap_hse", "HSE gap"),
                   ("cbm_m_dos_file", "m_dos CBM"), ("thickness", "thickness")):
    a = pd.to_numeric(d.loc[anchor, col], errors="coerce").dropna()
    b = pd.to_numeric(d.loc[~anchor, col], errors="coerce").dropna()
    if len(a) < 10 or len(b) < 10:
        continue
    ks, p = stats.ks_2samp(a, b)
    log(f"  {label:<12} anchors: median={a.median():.3f} "
        f"[{a.quantile(.1):.2f}, {a.quantile(.9):.2f}]   "
        f"others: median={b.median():.3f} [{b.quantile(.1):.2f}, {b.quantile(.9):.2f}]")
    log(f"  {'':<12} KS={ks:.3f}  p={p:.2e}  "
        f"{'DIFFERENT' if p < 0.01 else 'compatible'}")
    rows.append({"variable": label, "anchor_median": float(a.median()),
                 "other_median": float(b.median()), "ks": float(ks), "p": float(p)})

# =============================================================================
# 2. How many materials extrapolate beyond the anchor cloud?
# =============================================================================
# Mahalanobis distance in (PBE, HSE) relative to the anchor covariance.
log("--- extrapolation check in (PBE, HSE) space ---")
X = d[["gap", "gap_hse"]].to_numpy(float)
Xa = X[anchor]
mu = Xa.mean(axis=0)
S  = np.cov(Xa.T)
Si = np.linalg.inv(S)
dm = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, Si, X - mu))
d["mahalanobis"] = dm
thr = np.quantile(dm[anchor], 0.95)       # 95th percentile of the anchor cloud
frac_out = float((dm[~anchor] > thr).mean())
log(f"  anchor cloud 95th-pct Mahalanobis distance = {thr:.2f}")
log(f"  non-anchor materials beyond it             = {frac_out*100:.1f}%")
log(f"  max non-anchor distance                    = {dm[~anchor].max():.2f}")

# simple box check too, easier to state in a paper
lo_p, hi_p = d.loc[anchor, "gap"].min(), d.loc[anchor, "gap"].max()
lo_h, hi_h = d.loc[anchor, "gap_hse"].min(), d.loc[anchor, "gap_hse"].max()
inbox = ((d["gap"].between(lo_p, hi_p)) & (d["gap_hse"].between(lo_h, hi_h)))
log(f"  anchor PBE range [{lo_p:.2f}, {hi_p:.2f}]  HSE range [{lo_h:.2f}, {hi_h:.2f}]")
log(f"  materials inside the anchor box: {int(inbox.sum()):,} / {len(d):,} "
    f"({inbox.mean()*100:.1f}%)")
d["in_anchor_box"] = inbox

# =============================================================================
# 3. Spot check against literature G0W0 monolayer gaps
# =============================================================================
# If the map is unbiased these should agree within roughly the residual sigma
# (0.26 eV). A systematic overshoot across all of them indicates bias.
log("--- predicted vs literature G0W0 (eV) ---")
LIT_GW = {"MoS2": 2.70, "WS2": 2.85, "WSe2": 2.50, "MoSe2": 2.35,
          "MoTe2": 1.85, "BN": 6.80, "P": 2.10}
def ckey(f):
    rc = Composition(f).reduced_composition.get_el_amt_dict()
    return "-".join(f"{e}{int(round(n))}" for e, n in sorted(rc.items()))

log(f"  {'mat':<7} {'PBE':>6} {'HSE':>6} {'C2DB GW':>8} {'pred':>7} {'+/-':>6} "
    f"{'lit GW':>7} {'pred-lit':>9}")
errs = []
for f, lit in LIT_GW.items():
    sub = d[d["comp_key"] == ckey(f)]
    if not len(sub):
        log(f"  {f:<7} absent"); continue
    r = sub.iloc[0]
    gw = r["gap_gw"] if pd.notna(r["gap_gw"]) else np.nan
    diff = r["E_pred"] - lit
    errs.append(diff)
    log(f"  {f:<7} {r['gap']:>6.2f} {r['gap_hse']:>6.2f} "
        f"{gw if pd.notna(gw) else float('nan'):>8.2f} {r['E_pred']:>7.2f} "
        f"{r['E_sd']:>6.2f} {lit:>7.2f} {diff:>+9.2f}")
if errs:
    errs = np.array(errs)
    log(f"  mean signed error = {errs.mean():+.2f} eV   "
        f"mean |error| = {np.abs(errs).mean():.2f} eV   "
        f"(calibration residual sigma is 0.26 eV)")
    if errs.mean() > 0.30:
        log("  *** SYSTEMATIC OVERSHOOT. The calibration is biased high on "
            "well-known monolayers. Most likely the anchor set is not "
            "representative. Options, in order of preference:")
        log("      (a) Restrict the primary analysis to the anchor box "
            "(materials where the map interpolates rather than extrapolates) "
            "and report the restricted N.")
        log("      (b) Use HSE alone as the predictor: b1 contributed almost "
            "nothing (residual 0.266 vs 0.261 eV) and a one-predictor map "
            "extrapolates far more safely than a two-predictor one.")
        log("      (c) Keep the full set but report the bias explicitly and "
            "add it to the uncertainty budget as a systematic term.")
    else:
        log("  No large systematic bias. The map is usable on the full set.")

# =============================================================================
# 4. Verdict
# =============================================================================
log("--- verdict ---")
same_dist = all(r["p"] >= 0.01 for r in rows if r["variable"].endswith("gap"))
if not same_dist:
    log("  The anchors are NOT a random sample of the working set. The 10-fold "
        "coverage result is valid for interpolation and must be described that "
        "way in the paper. State plainly that G0W0 availability in C2DB is not "
        "random and that predictions outside the anchor region carry "
        "unvalidated uncertainty.")
else:
    log("  Anchors look compatible with the working set; the coverage result "
        "carries over to the full population.")

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
d[["uid", "comp_key", "gap", "gap_hse", "gap_gw", "E_pred", "E_sd",
   "mahalanobis", "in_anchor_box"]].to_csv(
    SUBDIRS["processed"] / "step4_per_material_gap.csv", index=False)
log(f"[cell9] saved {OUT_CSV}")
log("[cell9] DONE\n")
