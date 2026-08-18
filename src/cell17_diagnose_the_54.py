# =============================================================================
# TJEECC - CELL 17: what actually makes the 54 "gap-sensitive" materials
#                   gap-sensitive, and why Sobol disagrees
#
# CONTRADICTION TO RESOLVE:
#   cell11 census : 54 materials show >0.3 decade spread in on/off from the
#                   band gap alone.
#   cell15 Sobol  : E_g total-order index = 0.000 on those same materials.
#
# Both cannot be right. This cell inspects the 54 directly instead of
# theorising. An earlier hypothesis of mine (an interpolation artefact at the
# reachability boundary) was wrong: `reachable` already guarantees the
# crossing lies above the ambipolar minimum, so that guard was inert.
#
# Run cells 10 and 11 first.
# =============================================================================

import numpy as np, pandas as pd

PROC = SUBDIRS["processed"]
par  = pd.read_parquet(PROC / "device_parameters.parquet")
cen  = pd.read_csv(PROC / "gap_sensitivity_census.csv")
ids  = pd.read_csv(PROC / "gap_posterior_uids.csv")
Ep   = np.load(PROC / "gap_posterior.npy")
gi   = {u: i for i, u in enumerate(ids["uid"])}

gs = cen[cen["spr_gap_on_off"] > 0.30].copy()
log(f"[cell17] {len(gs)} flagged gap-sensitive")

# --- 1. What do they look like? ---------------------------------------------
log("--- distribution of the flagged set ---")
for c, lbl in (("E_mean", "calibrated E_g (eV)"),
               ("vbm_m_dos_file", "hole DOS mass"),
               ("cbm_m_dos_file", "electron DOS mass"),
               ("I_OFF", "central I_OFF (A/m)"),
               ("on_off", "central on/off")):
    if c not in gs.columns:
        continue
    v = pd.to_numeric(gs[c], errors="coerce").dropna()
    a = pd.to_numeric(cen.loc[cen["spr_gap_on_off"] <= 0.30, c],
                      errors="coerce").dropna()
    log(f"  {lbl:<22} flagged: med={v.median():.4g} [{v.min():.3g},{v.max():.3g}]"
        f"   others: med={a.median():.4g}")

# THE decisive split: is the off-current spec reachable at the central gap?
TARGET = TECH["I_OFF"]
gs["floor_above_spec"] = gs["I_OFF"] > TARGET * 1.001
n_floor = int(gs["floor_above_spec"].sum())
log(f"--- reachability ---")
log(f"  flagged materials whose central I_OFF exceeds the 100 nA/um spec: "
    f"{n_floor} / {len(gs)}")
oth = cen[cen["spr_gap_on_off"] <= 0.30]
log(f"  same for the non-flagged set: "
    f"{int((oth['I_OFF'] > TARGET*1.001).sum())} / {len(oth)}")
log("  If the flagged set is exactly the ambipolar-limited set, the census is "
    "right and the mechanism is real: for these materials I_OFF is NOT pinned "
    "at the spec, so the gap moves it directly.")

# --- 2. Trace one flagged material end to end -------------------------------
log("--- per-material trace, 6 flagged examples ---")
log(f"  {'uid':<16} {'E_g':>6} {'E_sd':>6} {'I_OFF':>10} {'on/off':>10} "
    f"{'spread':>7}  mechanism")
for _, r in gs.nlargest(6, "spr_gap_on_off").iterrows():
    E = Ep[gi[r["uid"]]]
    d = compute_fom(Eg=np.clip(E[None, :256], 0.05, None),
                    m_dos_e=np.full((1, 256), r.cbm_m_dos_file),
                    m_dos_h=np.full((1, 256), r.vbm_m_dos_file),
                    m_cond_e=np.full((1, 256), r.cbm_m_cond),
                    m_cond_h=np.full((1, 256), r.vbm_m_cond),
                    eps_ch=np.full((1, 256), r.eps_ch),
                    t_ch=np.full((1, 256), r.t_ch_m))
    off = d["I_OFF"].ravel()
    frac_pinned = float(np.mean(off <= TARGET * 1.001))
    mech = ("I_OFF pinned in all draws -> gap inert" if frac_pinned > 0.99 else
            f"I_OFF floats in {100*(1-frac_pinned):.0f}% of draws -> gap acts")
    log(f"  {r['uid']:<16} {r.E_mean:>6.2f} {float(E.std()):>6.3f} "
        f"{r.I_OFF:>10.3e} {r.on_off:>10.3e} {r.spr_gap_on_off:>7.3f}  {mech}")

# --- 3. Why Sobol sees nothing ----------------------------------------------
# Sobol samples E_g UNIFORMLY on [E-2sd, E+2sd]; the census uses the actual
# posterior draws. If the response is a threshold (flat, then a sharp knee),
# a uniform design that straddles the knee still produces variance, so this
# alone should not zero the index. The likelier cause is that cell15 picked
# its 20 representatives by I_ON quantile WITHIN the flagged pool, and the
# high-I_ON members of that pool may be the ones whose gaps are large.
log("--- reconciling with Sobol ---")
sob_pick = gs.nlargest(20, "I_ON") if "I_ON" in gs.columns else gs.head(20)
log(f"  cell15 selects by I_ON quantile within the flagged pool.")
log(f"  flagged pool  E_g: median={gs['E_mean'].median():.2f} eV")
log(f"  the 20 highest-I_ON of them: median={sob_pick['E_mean'].median():.2f} eV")
log(f"  fraction of the pool with E_g < 1.4 eV (ambipolar window): "
    f"{100*float((gs['E_mean'] < 1.4).mean()):.0f}%")
log(f"  fraction of the I_ON-selected 20 with E_g < 1.4 eV: "
    f"{100*float((sob_pick['E_mean'] < 1.4).mean()):.0f}%")
log("  If the second number is far below the first, cell15 sampled the WRONG "
    "materials: it picked the high-current members of the flagged pool, which "
    "are wide-gap, so E_g was inert in the Sobol design. Fix: select the "
    "Sobol representatives by E_g, not by I_ON.")

# --- 4. Verdict --------------------------------------------------------------
log("--- verdict ---")
if n_floor > 0.5 * len(gs):
    log("  The census is RIGHT. The flagged materials are ambipolar-limited: "
        "their off-current floor sits above the 100 nA/um spec, so the gap "
        "sets I_OFF directly and uncertainty in it propagates. Report this "
        "subset as the gap-sensitive population, and REDO the Sobol with "
        "representatives selected by E_g.")
else:
    log("  The census flag is NOT explained by ambipolar limitation. Inspect "
        "the per-material traces above before using the 54 in the paper; if "
        "the mechanism cannot be identified, drop the subset and report only "
        "that the gap is inert at this technology corner.")
gs.to_csv(PROC / "step8_flagged_gap_sensitive.csv", index=False)
log(f"  saved {PROC/'step8_flagged_gap_sensitive.csv'}")
log("[cell17] DONE\n")
