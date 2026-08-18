# =============================================================================
# TJEECC - CELL 13 / PLAN STEP 8 (completion): near-tie test, dominance,
#                    rank entropy, Sobol indices, robust shortlist
#
# cell12 established that top-10 retention sits at 65-72% in EVERY transport
# regime: about a third of a shortlist is not reproducible under propagated
# first-principles uncertainty, and that is regime-invariant because the mass
# exponent scales the between-material variation and the induced spread
# equally.
#
# Part A is a NULL TEST that can invalidate that headline. If materials ranked
# 8-15 are within a few percent of one another, shuffling the top 10 is
# trivially expected and means nothing. Run it before believing the result.
#
# Produces Figure 4 (dominance + retention), Figure 5 (Sobol) and Table 10.
# Run cell10, cell11, cell12 first.
# =============================================================================

import json
import numpy as np, pandas as pd
from scipy.stats import kendalltau

PARAM_PQ = SUBDIRS["processed"] / "device_parameters.parquet"
POST_NPY = SUBDIRS["processed"] / "gap_posterior.npy"
POST_IDS = SUBDIRS["processed"] / "gap_posterior_uids.csv"

NEARTIE_CSV = SUBDIRS["processed"] / "step8_neartie_test.csv"
DOM_CSV     = SUBDIRS["processed"] / "fig4_dominance_matrix.csv"
ENTROPY_CSV = SUBDIRS["processed"] / "step8_rank_entropy.csv"
SOBOL_CSV   = SUBDIRS["processed"] / "fig5_sobol_indices.csv"
TABLE10_CSV = SUBDIRS["processed"] / "table10_robust_shortlist.csv"

NS         = 2000        # draws for dominance / entropy
SIG_LN_M   = 0.20        # ASSUMED
SIG_LN_EPS = 0.19        # measured
LAMBDA     = None        # ballistic base case
FOM        = "I_ON"

par  = pd.read_parquet(PARAM_PQ).reset_index(drop=True)
ids  = pd.read_csv(POST_IDS)
Ep   = np.load(POST_NPY)
gi   = np.array([{u: i for i, u in enumerate(ids["uid"])}[u] for u in par["uid"]])
n    = len(par)
rng  = np.random.default_rng(SEED)
tile = lambda a: np.repeat(np.asarray(a, float)[:, None], NS, axis=1)

cols = dict(md_e=par["cbm_m_dos_file"].to_numpy(), md_h=par["vbm_m_dos_file"].to_numpy(),
            mc_e=par["cbm_m_cond"].to_numpy(),     mc_h=par["vbm_m_cond"].to_numpy(),
            eps=par["eps_ch"].to_numpy(),          t=par["t_ch_m"].to_numpy(),
            Eg=par["E_mean"].to_numpy())

ref = compute_fom(Eg=cols["Eg"][:, None], m_dos_e=cols["md_e"][:, None],
                  m_dos_h=cols["md_h"][:, None], m_cond_e=cols["mc_e"][:, None],
                  m_cond_h=cols["mc_h"][:, None], eps_ch=cols["eps"][:, None],
                  t_ch=cols["t"][:, None], lambda0_over_l=LAMBDA)
fom_ref = ref[FOM].ravel()
order   = np.argsort(-fom_ref)
par["fom_ref"] = fom_ref

# =============================================================================
# PART A. NULL TEST: are the shortlist boundaries actually separated?
# =============================================================================
log("=" * 78)
log("A. NEAR-TIE TEST  (can invalidate the whole rank-instability finding)")
top = order[:25]
log(f"  {'rank':>4} {'uid':<18} {'I_ON':>10} {'gap to next':>12}")
for i in range(min(15, len(top))):
    a = fom_ref[top[i]]
    b = fom_ref[top[i + 1]] if i + 1 < len(top) else np.nan
    log(f"  {i+1:>4} {par['uid'].iloc[top[i]]:<18} {a:>10.1f} "
        f"{(a-b)/a*100 if np.isfinite(b) else np.nan:>11.2f}%")

sep_10 = (fom_ref[order[9]] - fom_ref[order[10]]) / fom_ref[order[9]]
spread_top10 = (fom_ref[order[0]] - fom_ref[order[9]]) / fom_ref[order[0]]
# typical uncertainty on a single material, for comparison
lm = rng.normal(0.0, SIG_LN_M, (n, 64))
le = rng.normal(0.0, SIG_LN_EPS, (n, 64))
gs = rng.choice(Ep.shape[1], 64, replace=False)
dd = compute_fom(Eg=np.clip(Ep[gi][:, gs], 0.05, None),
                 m_dos_e=np.repeat(cols["md_e"][:, None], 64, 1) * np.exp(lm),
                 m_dos_h=np.repeat(cols["md_h"][:, None], 64, 1) * np.exp(lm),
                 m_cond_e=np.repeat(cols["mc_e"][:, None], 64, 1) * np.exp(lm),
                 m_cond_h=np.repeat(cols["mc_h"][:, None], 64, 1) * np.exp(lm),
                 eps_ch=np.repeat(cols["eps"][:, None], 64, 1) * np.exp(le),
                 t_ch=np.repeat(cols["t"][:, None], 64, 1), lambda0_over_l=LAMBDA)
rel_unc = float(np.median(dd[FOM].std(axis=1) / dd[FOM].mean(axis=1)))

log(f"  rank-10 to rank-11 separation : {sep_10*100:.2f}%")
log(f"  rank-1 to rank-10 spread      : {spread_top10*100:.2f}%")
log(f"  median per-material 1-sigma   : {rel_unc*100:.2f}%")
verdict = "REAL" if sep_10 * 100 > rel_unc * 100 * 0.5 else "NEAR-TIE ARTEFACT"
log(f"  VERDICT: {verdict}")
log("  If the rank-10/11 separation is far below the per-material sigma, the "
    "shortlist boundary is a coin flip and 'one third of the top 10 changes' "
    "is a statement about ties, not about screening reliability. In that case "
    "report dominance probabilities and DROP the top-k retention headline.")
pd.DataFrame([{"sep_10_11_pct": sep_10*100, "spread_1_10_pct": spread_top10*100,
               "median_sigma_pct": rel_unc*100, "verdict": verdict}]
             ).to_csv(NEARTIE_CSV, index=False)

# =============================================================================
# PART B. Monte Carlo, dominance probability, rank entropy
# =============================================================================
log("=" * 78)
log("B. DOMINANCE AND RANK ENTROPY")
lm = rng.normal(0.0, SIG_LN_M, (n, NS))
le = rng.normal(0.0, SIG_LN_EPS, (n, NS))
gs = rng.choice(Ep.shape[1], NS, replace=True)
draws = compute_fom(Eg=np.clip(Ep[gi][:, gs], 0.05, None),
                    m_dos_e=tile(cols["md_e"]) * np.exp(lm),
                    m_dos_h=tile(cols["md_h"]) * np.exp(lm),
                    m_cond_e=tile(cols["mc_e"]) * np.exp(lm),
                    m_cond_h=tile(cols["mc_h"]) * np.exp(lm),
                    eps_ch=tile(cols["eps"]) * np.exp(le),
                    t_ch=tile(cols["t"]), lambda0_over_l=LAMBDA)
F = draws[FOM]                                    # (n, NS)

TOPK = 30
sel = order[:TOPK]
Fs = F[sel]                                       # (TOPK, NS)
dom = (Fs[:, None, :] > Fs[None, :, :]).mean(axis=2)   # P(row beats col)
pd.DataFrame(dom, index=par["uid"].iloc[sel], columns=par["uid"].iloc[sel]
             ).to_csv(DOM_CSV)
log(f"  dominance matrix for the top {TOPK} -> {DOM_CSV.name} (Figure 4a)")
offdiag = dom[~np.eye(TOPK, dtype=bool)]
log(f"  pairs with P(A>B) between 0.4 and 0.6 (statistically tied): "
    f"{(np.abs(offdiag-0.5)<0.1).mean()*100:.1f}%")
log(f"  pairs with P(A>B) > 0.95 (clearly ordered): "
    f"{(offdiag>0.95).mean()*100:.1f}%")

# rank entropy, normalised
ranks = np.argsort(np.argsort(-F, axis=0), axis=0)      # (n, NS)
ent = np.zeros(n)
for i in range(n):
    counts = np.bincount(ranks[i], minlength=n).astype(float)
    p = counts[counts > 0] / NS
    ent[i] = -(p * np.log(p)).sum() / np.log(n)
par["rank_entropy"] = ent
par["rank_mean"] = ranks.mean(axis=1)
log(f"  rank entropy: median={np.median(ent):.4f}  "
    f"top-30 median={np.median(ent[sel]):.4f}  (0 = perfectly determined)")
par[["uid", "comp_key", "fom_ref", "rank_mean", "rank_entropy"]] \
    .sort_values("rank_mean").to_csv(ENTROPY_CSV, index=False)

# =============================================================================
# PART C. Sobol indices (Figure 5)
# =============================================================================
log("=" * 78)
log("C. SOBOL SENSITIVITY")
try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    # 20 materials spanning the parameter space, chosen by I_ON quantile
    qs = np.linspace(0.02, 0.98, 20)
    reps = order[np.clip((qs * (n - 1)).astype(int), 0, n - 1)]
    NSOB = 1024
    names = ["Eg", "m_dos_e", "m_cond_e", "m_dos_h", "m_cond_h", "eps"]
    out = []
    for mi in reps:
        c = {k: v[mi] for k, v in cols.items()}
        sd_gap = float(Ep[gi[mi]].std())
        problem = {"num_vars": 6, "names": names, "bounds": [
            [max(0.05, c["Eg"] - 2*sd_gap), c["Eg"] + 2*sd_gap],
            [c["md_e"]*np.exp(-2*SIG_LN_M), c["md_e"]*np.exp(2*SIG_LN_M)],
            [c["mc_e"]*np.exp(-2*SIG_LN_M), c["mc_e"]*np.exp(2*SIG_LN_M)],
            [c["md_h"]*np.exp(-2*SIG_LN_M), c["md_h"]*np.exp(2*SIG_LN_M)],
            [c["mc_h"]*np.exp(-2*SIG_LN_M), c["mc_h"]*np.exp(2*SIG_LN_M)],
            [c["eps"]*np.exp(-2*SIG_LN_EPS), c["eps"]*np.exp(2*SIG_LN_EPS)]]}
        X = sobol_sample.sample(problem, NSOB, calc_second_order=False)
        r = compute_fom(Eg=X[:, 0][:, None], m_dos_e=X[:, 1][:, None],
                        m_cond_e=X[:, 2][:, None], m_dos_h=X[:, 3][:, None],
                        m_cond_h=X[:, 4][:, None], eps_ch=X[:, 5][:, None],
                        t_ch=np.full((len(X), 1), c["t"]), lambda0_over_l=LAMBDA)
        for fom_name in ("I_ON", "on_off"):
            y = np.log10(np.maximum(r[fom_name].ravel(), 1e-300))
            if y.std() < 1e-12:
                continue
            Si = sobol_analyze.analyze(problem, y, calc_second_order=False,
                                       print_to_console=False)
            for j, nm in enumerate(names):
                out.append({"uid": par["uid"].iloc[mi], "fom": fom_name,
                            "param": nm, "S1": Si["S1"][j], "S1_conf": Si["S1_conf"][j],
                            "ST": Si["ST"][j], "ST_conf": Si["ST_conf"][j]})
    sob = pd.DataFrame(out)
    sob.to_csv(SOBOL_CSV, index=False)
    log(f"  {len(reps)} representative materials x {NSOB} Saltelli base samples")
    for fom_name in ("I_ON", "on_off"):
        s = sob[sob.fom == fom_name].groupby("param")[["ST", "ST_conf"]].median()
        log(f"  --- {fom_name}: median total-order index ---")
        for nm in names:
            if nm in s.index:
                st, cf = s.loc[nm, "ST"], s.loc[nm, "ST_conf"]
                tag = "influential" if st - cf > 0.02 else "NOT influential (CI covers 0)"
                log(f"    {nm:<10} ST={st:>6.3f} +/- {cf:.3f}   {tag}")
    log("  Report indices whose confidence interval covers zero as NOT "
        "influential, rather than ranking noise.")
except ImportError:
    log("  SALib unavailable; skipping Sobol.")

# =============================================================================
# PART D. Robust shortlist  ->  TABLE 10
# =============================================================================
log("=" * 78)
log("D. ROBUST SELECTION RULE (Table 10)")
mean_fom = F.mean(axis=1)
lcb_fom  = np.percentile(F, 10, axis=1)          # 10th-percentile lower bound
top_mean = np.argsort(-mean_fom)[:10]
top_lcb  = np.argsort(-lcb_fom)[:10]
overlap  = len(set(top_mean) & set(top_lcb))

t10 = pd.DataFrame({
    "rank": np.arange(1, 11),
    "by_mean_uid":  par["uid"].iloc[top_mean].to_numpy(),
    "by_mean_comp": par["comp_key"].iloc[top_mean].to_numpy(),
    "by_mean_I_ON": mean_fom[top_mean],
    "by_LCB_uid":   par["uid"].iloc[top_lcb].to_numpy(),
    "by_LCB_comp":  par["comp_key"].iloc[top_lcb].to_numpy(),
    "by_LCB_I_ON":  lcb_fom[top_lcb],
    "LCB_entropy":  ent[top_lcb],
})
t10.to_csv(TABLE10_CSV, index=False)
log(f"  {'rk':>3} {'by posterior MEAN':<26} {'by 10th-pct LCB':<26} {'H':>6}")
for r in t10.itertuples(index=False):
    log(f"  {r.rank:>3} {r.by_mean_comp:<26} {r.by_LCB_comp:<26} {r.LCB_entropy:>6.3f}")
log(f"  overlap between the two shortlists: {overlap}/10")
log(f"  saved {TABLE10_CSV}")
log("  If the overlap is high, the robust rule changes little and should be "
    "reported as a null result rather than sold as a contribution.")
log("[cell13] DONE\n")
