# =============================================================================
# TJEECC - CELL 12: rank stability as a function of transport regime
#
# THE PAPER'S MAIN RESULT.
#
# The ballistic census (cell11) showed almost nothing moves: gap-sensitive
# 7.2%, mass- and eps-sensitive 0%. That is real physics, not a modelling
# shortfall: at EOT = 0.6 nm the on-state is charge-limited, so ballistic
# I_ON ~ 1/sqrt(m_c) and a 20% mass uncertainty buys only ~0.09 decades.
#
# But the mass exponent is regime-dependent:
#     d log I_ON / d log m = -0.5 - 1.5 * 2l/(2l + lambda)
#     -> -0.5 ballistic, -2.0 diffusive  (verified against the model)
# so mass uncertainty is amplified EXACTLY 4x toward the diffusive limit.
#
# This cell measures, across the ballisticity range: the induced spread in
# I_ON, and the rank stability of the resulting shortlist (Kendall tau and
# top-k retention against the noise-free ranking).
#
# Run cell10 (device model) and cell11 (parameter table) first.
# =============================================================================

import json
import numpy as np, pandas as pd
from scipy.stats import kendalltau

PARAM_PQ  = SUBDIRS["processed"] / "device_parameters.parquet"
POST_NPY  = SUBDIRS["processed"] / "gap_posterior.npy"
POST_IDS  = SUBDIRS["processed"] / "gap_posterior_uids.csv"
OUT_CSV   = SUBDIRS["processed"] / "transport_regime_rank_stability.csv"
FIG_CSV   = SUBDIRS["processed"] / "fig_regime_sensitivity.csv"

NS = 256                      # Monte Carlo draws per material
SIG_LN_EPS = 0.19             # measured (C2DB vs JARVIS polarizability)
MASS_SIGMAS = (0.10, 0.20, 0.40)   # ASSUMED; the sweep is load-bearing
LAMBDAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 100.0, None)  # None = ballistic

par = pd.read_parquet(PARAM_PQ)
ids = pd.read_csv(POST_IDS)
Epost = np.load(POST_NPY)
row_of = {u: i for i, u in enumerate(ids["uid"])}
gi = np.array([row_of[u] for u in par["uid"]])
n = len(par)
log(f"[cell12] {n:,} materials, {NS} draws, "
    f"{len(LAMBDAS)} regimes x {len(MASS_SIGMAS)} mass sigmas")

rng = np.random.default_rng(SEED)
tile = lambda a: np.repeat(np.asarray(a, float)[:, None], NS, axis=1)
base = dict(
    md_e=par["cbm_m_dos_file"].to_numpy(), md_h=par["vbm_m_dos_file"].to_numpy(),
    mc_e=par["cbm_m_cond"].to_numpy(),     mc_h=par["vbm_m_cond"].to_numpy(),
    eps=par["eps_ch"].to_numpy(),          t=par["t_ch_m"].to_numpy(),
    Eg=par["E_mean"].to_numpy(),
)

gsel = rng.choice(Epost.shape[1], NS, replace=False)
Eg_draws = np.clip(Epost[gi][:, gsel], 0.05, None)

def ranking(x):
    """Descending rank; ties broken by value order. Higher FoM = better."""
    return np.argsort(np.argsort(-x))

rows, figrows = [], []
for lam in LAMBDAS:
    lam_lbl = "ballistic" if lam is None else f"{lam:g}"
    # noise-free reference ranking for this regime
    ref = compute_fom(Eg=base["Eg"][:, None], m_dos_e=base["md_e"][:, None],
                      m_dos_h=base["md_h"][:, None], m_cond_e=base["mc_e"][:, None],
                      m_cond_h=base["mc_h"][:, None], eps_ch=base["eps"][:, None],
                      t_ch=base["t"][:, None], lambda0_over_l=lam)
    ion_ref = ref["I_ON"].ravel()
    B_med = float(np.median(ref["B"]))
    rank_ref = ranking(ion_ref)

    for sig_m in MASS_SIGMAS:
        lm = rng.normal(0.0, sig_m, (n, NS))
        le = rng.normal(0.0, SIG_LN_EPS, (n, NS))
        d = compute_fom(
            Eg=Eg_draws,
            m_dos_e=tile(base["md_e"]) * np.exp(lm),
            m_dos_h=tile(base["md_h"]) * np.exp(lm),
            m_cond_e=tile(base["mc_e"]) * np.exp(lm),
            m_cond_h=tile(base["mc_h"]) * np.exp(lm),
            eps_ch=tile(base["eps"]) * np.exp(le),
            t_ch=tile(base["t"]), lambda0_over_l=lam)

        lg = np.log10(np.maximum(d["I_ON"], 1e-300))
        spread = np.percentile(lg, 84, axis=1) - np.percentile(lg, 16, axis=1)

        # rank stability: each draw gives a ranking; compare to the noise-free one
        taus, top10, top20 = [], [], []
        ref10 = set(np.argsort(-ion_ref)[:10])
        ref20 = set(np.argsort(-ion_ref)[:20])
        for k in range(0, NS, 8):                     # 32 draws is plenty
            rk = ranking(d["I_ON"][:, k])
            taus.append(kendalltau(rank_ref, rk).statistic)
            top10.append(len(ref10 & set(np.argsort(-d["I_ON"][:, k])[:10])) / 10)
            top20.append(len(ref20 & set(np.argsort(-d["I_ON"][:, k])[:20])) / 20)

        rec = dict(regime=lam_lbl, B_median=B_med, sigma_ln_mass=sig_m,
                   spread_med=float(np.median(spread)),
                   spread_p90=float(np.percentile(spread, 90)),
                   frac_gt_2x=float((spread > 0.30).mean()),
                   kendall_tau=float(np.mean(taus)),
                   top10_retention=float(np.mean(top10)),
                   top20_retention=float(np.mean(top20)))
        rows.append(rec)
        if sig_m == 0.20:
            figrows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)
pd.DataFrame(figrows).to_csv(FIG_CSV, index=False)

log("=" * 78)
log("RANK STABILITY vs TRANSPORT REGIME   (sigma_ln mass = 0.20, the base case)")
log(f"  {'regime':>10} {'B':>6} {'spread':>8} {'>2x':>7} {'tau':>7} "
    f"{'top10':>7} {'top20':>7}")
for r in res[res.sigma_ln_mass == 0.20].itertuples(index=False):
    log(f"  {r.regime:>10} {r.B_median:>6.3f} {r.spread_med:>8.3f} "
        f"{r.frac_gt_2x*100:>6.1f}% {r.kendall_tau:>7.3f} "
        f"{r.top10_retention*100:>6.1f}% {r.top20_retention*100:>6.1f}%")

log("--- mass-sigma sensitivity (the ASSUMED input) ---")
for sig in MASS_SIGMAS:
    sub = res[res.sigma_ln_mass == sig]
    bal = sub[sub.regime == "ballistic"].iloc[0]
    dif = sub[sub.regime == "0.1"].iloc[0]
    log(f"  sigma_ln={sig:.2f}:  ballistic tau={bal.kendall_tau:.3f} "
        f"top10={bal.top10_retention*100:.0f}%   |   "
        f"diffusive tau={dif.kendall_tau:.3f} top10={dif.top10_retention*100:.0f}%")

bal = res[(res.regime == "ballistic") & (res.sigma_ln_mass == 0.20)].iloc[0]
dif = res[(res.regime == "0.1") & (res.sigma_ln_mass == 0.20)].iloc[0]
amp = dif.spread_med / max(bal.spread_med, 1e-9)
log("--- headline numbers ---")
log(f"  I_ON spread   ballistic {bal.spread_med:.3f} dec -> diffusive "
    f"{dif.spread_med:.3f} dec   (amplification x{amp:.1f}, analytic 4.0)")
log(f"  Kendall tau   {bal.kendall_tau:.3f} -> {dif.kendall_tau:.3f}")
log(f"  top-10 keep   {bal.top10_retention*100:.0f}% -> {dif.top10_retention*100:.0f}%")
log("  The claim to make: screening rankings are robust at the ballistic limit "
    "and degrade predictably as the channel becomes diffusive, with a mass "
    "sensitivity exponent moving from -0.5 to -2.0. Report the ballisticity at "
    "which top-10 retention drops below 80% as the practical design rule.")
log(f"  saved {OUT_CSV}")
log("[cell12] DONE\n")
