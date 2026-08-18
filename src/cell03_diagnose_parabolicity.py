# =============================================================================
# TJEECC - CELL 3: DIAGNOSE THE m_dos SELF-CONSISTENCY FAILURE
#
# Cell 2's gate reported 83.4% within 2%, below the 95% threshold. Before
# treating that as a defect, test whether the deviation is explained by band
# warping / non-parabolicity. If it is, the failing materials are ones where
# the parabolic-band device model does not apply, and excluding them is a
# scope decision rather than a bug fix.
#
# Requires cell00_bootstrap.py. Reads c2db_enriched.parquet.
# =============================================================================

import numpy as np, pandas as pd
from pathlib import Path

ENRICHED_PQ = SUBDIRS["processed"] / "c2db_enriched.parquet"
OUT_PQ      = SUBDIRS["processed"] / "c2db_device_ready.parquet"
DIAG_CSV    = SUBDIRS["processed"] / "parabolicity_diagnosis.csv"

PARABOLIC_TOL = 0.02          # |sqrt(m1*m2) - m_dos| / m_dos

df = pd.read_parquet(ENRICHED_PQ)
log(f"[cell3] loaded {len(df):,} screened materials")

# --- 1. Relative deviation per edge -----------------------------------------
for edge in ("cbm", "vbm"):
    f, c = f"{edge}_m_dos_file", f"{edge}_m_dos_check"
    df[f"{edge}_rel"] = (df[c] - df[f]).abs() / df[f]
    df[f"{edge}_ratio"] = df[c] / df[f]
    df[f"{edge}_parabolic"] = df[f"{edge}_rel"] < PARABOLIC_TOL

df["parabolic"] = df["cbm_parabolic"] & df["vbm_parabolic"]

log("--- deviation summary ---")
for edge in ("cbm", "vbm"):
    s = df[f"{edge}_rel"].dropna()
    log(f"  {edge}: median={s.median():.2e}  p90={s.quantile(0.90):.3f}  "
        f"p99={s.quantile(0.99):.3f}  frac<2%={(s<PARABOLIC_TOL).mean()*100:.1f}%")

# --- 2. Is the deviation explained by warping? -------------------------------
# H0: rel error is unrelated to warping -> the extraction is buggy.
# H1: rel error tracks |warping| and anisotropy -> it is band non-parabolicity.
log("--- hypothesis test: does warping explain the deviation? ---")
from scipy.stats import spearmanr, mannwhitneyu

for edge in ("cbm", "vbm"):
    sub = df.dropna(subset=[f"{edge}_rel", f"{edge}_warping", f"{edge}_anisotropy"])
    if len(sub) < 50:
        log(f"  {edge}: too few rows ({len(sub)})")
        continue
    r_w, p_w = spearmanr(sub[f"{edge}_rel"], sub[f"{edge}_warping"].abs())
    r_a, p_a = spearmanr(sub[f"{edge}_rel"], sub[f"{edge}_anisotropy"])
    log(f"  {edge}: Spearman(rel, |warping|)    = {r_w:+.3f}  p={p_w:.2e}")
    log(f"  {edge}: Spearman(rel, anisotropy)   = {r_a:+.3f}  p={p_a:.2e}")

    # barrier_found flags an extremum whose fitting region is truncated
    if f"{edge}_barrier" in sub.columns:
        g1 = sub.loc[sub[f"{edge}_barrier"] == True,  f"{edge}_rel"].dropna()
        g0 = sub.loc[sub[f"{edge}_barrier"] == False, f"{edge}_rel"].dropna()
        if len(g1) > 10 and len(g0) > 10:
            u, p = mannwhitneyu(g1, g0, alternative="greater")
            log(f"  {edge}: barrier_found=True median rel={g1.median():.3f} "
                f"(n={len(g1)}) vs False={g0.median():.3f} (n={len(g0)}), "
                f"Mann-Whitney p={p:.2e}")

    # anisotropy contrast between the two groups is the clearest single number
    par = sub.loc[sub[f"{edge}_parabolic"], f"{edge}_anisotropy"]
    non = sub.loc[~sub[f"{edge}_parabolic"], f"{edge}_anisotropy"]
    log(f"  {edge}: median anisotropy  parabolic={par.median():.2f}  "
        f"non-parabolic={non.median():.2f}")

log("  INTERPRETATION: strong positive Spearman correlations, and a higher "
    "median anisotropy in the non-parabolic group, support non-parabolicity "
    "rather than an extraction bug. Weak or absent correlation means we have "
    "a real bug and must re-examine the emass.json parsing.")

# --- 3. Consequence for the working set --------------------------------------
log("--- device-ready set ---")
n_all  = len(df)
n_par  = int(df["parabolic"].sum())
sub    = df[df["parabolic"]].copy()
n_pol  = int(sub["alphax_el"].notna().sum())
n_gw   = int(sub["gap_gw"].notna().sum())

log(f"  screened (Cell 2)                        : {n_all:,}")
log(f"  parabolic at both band edges (< 2%)      : {n_par:,} "
    f"({n_par/n_all*100:.1f}%)")
log(f"  ... and with polarizability (full model) : {n_pol:,}")
log(f"  ... and with a G0W0 gap (calibration)    : {n_gw:,}")

if n_gw < 60:
    log(f"  WARNING: only {n_gw} G0W0 anchors. The Step 4 hierarchical model "
        f"may struggle to identify the PBE/HSE offsets. Consider relaxing "
        f"PARABOLIC_TOL to 0.05 and reporting sensitivity to that choice.")

# --- 4. Persist ---------------------------------------------------------------
diag_cols = [c for c in df.columns if any(
    c.endswith(s) for s in ("_rel", "_ratio", "_warping", "_anisotropy",
                            "_barrier", "_parabolic"))] + ["uid", "formula"]
df[diag_cols].to_csv(DIAG_CSV, index=False)
sub.to_parquet(OUT_PQ)
log(f"  diagnosis -> {DIAG_CSV}")
log(f"  device-ready set -> {OUT_PQ}")

# --- 5. Sanity: use the FILE m_dos downstream, never the recomputed one ------
# C2DB's m_dos comes from a proper DOS integration and is the authoritative
# value. sqrt(min*max) was only ever a cross-check. Downstream code must read
# cbm_m_dos_file / vbm_m_dos_file. The conductivity mass from the harmonic
# mean remains an ellipsoidal-band approximation, which is exactly why we
# restrict to the parabolic subset.
log("--- downstream convention (do not deviate) ---")
log("  DOS mass          m_d  = <edge>_m_dos_file   (authoritative, C2DB)")
log("  conductivity mass m_c  = <edge>_m_cond       (harmonic mean, valid on "
    "the parabolic subset only)")
log("  n_s and C_Q use m_d ; the Natori current uses m_c")
log("[cell3] DONE\n")
