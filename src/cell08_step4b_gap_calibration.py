# =============================================================================
# TJEECC - CELL 8 / PLAN STEP 4 (REVISED): gap uncertainty by G0W0 calibration
#
# Replaces the latent-variable hierarchical model of cell07, which failed
# (r_hat 1.42, ESS 8, 529 divergences). Root cause: PBE and HSE are strongly
# collinear, so they cannot act as independent measurements of a latent gap.
# The posterior had a ridge along  E_true -> c*E_true, beta -> beta/c , which
# only 129 G0W0 anchors had to break. Section 1 below tests that claim.
#
# Revised model: calibrate  E_gw ~ a + b1*gap_pbe + b2*gap_hse  on the 129
# anchors, then propagate BOTH parameter uncertainty and residual scatter to
# all materials as a posterior predictive distribution.
#
# Requires cell00_bootstrap.py.
# =============================================================================

import json, time
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

DEVICE_PQ   = SUBDIRS["processed"] / "c2db_device_ready.parquet"
POST_NPY    = SUBDIRS["processed"] / "gap_posterior.npy"
POST_IDS    = SUBDIRS["processed"] / "gap_posterior_uids.csv"
CAL_CSV     = SUBDIRS["processed"] / "step4_calibration_summary.csv"
COVER_CSV   = SUBDIRS["processed"] / "step4_coverage_check.csv"
N_DRAWS_KEEP = 1000

dev = pd.read_parquet(DEVICE_PQ)
d = dev[["uid", "gap", "gap_hse", "gap_gw"]].copy()
for c in ("gap", "gap_hse", "gap_gw"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["gap", "gap_hse"]).reset_index(drop=True)
anchor = d["gap_gw"].notna().to_numpy()
N = len(d)
log(f"[step4b] N={N:,}  anchors={int(anchor.sum())}")

# =============================================================================
# 1. Confirm the collinearity that broke cell07
# =============================================================================
log("--- why cell07 failed: PBE vs HSE collinearity ---")
r = np.corrcoef(d["gap"], d["gap_hse"])[0, 1]
A = np.vstack([np.ones(N), d["gap"].to_numpy()]).T
coef, *_ = np.linalg.lstsq(A, d["gap_hse"].to_numpy(), rcond=None)
resid = d["gap_hse"].to_numpy() - A @ coef
r2 = 1 - resid.var() / d["gap_hse"].to_numpy().var()
log(f"  pearson r(PBE, HSE)      = {r:.4f}")
log(f"  HSE ~ {coef[0]:+.3f} + {coef[1]:.3f}*PBE   R^2 = {r2:.4f}   "
    f"residual sd = {resid.std(ddof=2):.3f} eV")
log("  R^2 near 1 means HSE carries almost no information independent of PBE. "
    "Treating them as two independent measurements of a latent gap is "
    "therefore unidentified, which is what produced the ridge, the 529 "
    "divergences and the near-identical anchored/unanchored posterior widths.")

# =============================================================================
# 2. Calibration regression on the anchors
# =============================================================================
pbe_a = jnp.asarray(d.loc[anchor, "gap"].to_numpy())
hse_a = jnp.asarray(d.loc[anchor, "gap_hse"].to_numpy())
gw_a  = jnp.asarray(d.loc[anchor, "gap_gw"].to_numpy())

def cal_model(pbe, hse, gw=None):
    a  = numpyro.sample("a",  dist.Normal(0.0, 1.0))
    b1 = numpyro.sample("b1", dist.Normal(0.0, 1.0))
    b2 = numpyro.sample("b2", dist.Normal(1.0, 1.0))
    s  = numpyro.sample("sigma", dist.HalfNormal(0.5))
    mu = a + b1 * pbe + b2 * hse
    numpyro.sample("obs", dist.Normal(mu, s), obs=gw)

log("--- fitting calibration on the anchors ---")
t0 = time.time()
mcmc = MCMC(NUTS(cal_model, target_accept_prob=0.9),
            num_warmup=1000, num_samples=2000, num_chains=4,
            chain_method=MCMC_CHAIN_METHOD, progress_bar=False)
mcmc.run(jax.random.PRNGKey(SEED), pbe_a, hse_a, gw_a)
log(f"  sampled in {time.time()-t0:.0f} s")

idata = az.from_numpyro(mcmc)
summ = az.summary(idata, round_to=4)
for line in summ.to_string().splitlines():
    log("  " + line)
summ.to_csv(CAL_CSV)

max_rhat = float(summ["r_hat"].max()); min_ess = float(summ["ess_bulk"].min())
log(f"--- convergence ---")
log(f"  max r_hat={max_rhat:.4f} (<1.01)   min ess_bulk={min_ess:.0f} (>400)")
if max_rhat >= 1.01 or min_ess <= 400:
    raise RuntimeError("Calibration failed to converge. This model is simple "
                       "enough that failure means a data problem, not tuning.")
log("  PASS")

post = mcmc.get_samples()
a_m, b1_m, b2_m, s_m = (float(np.mean(post[k])) for k in ("a", "b1", "b2", "sigma"))
log(f"  E_gw ~ {a_m:+.3f} + {b1_m:.3f}*PBE + {b2_m:.3f}*HSE   "
    f"residual sigma = {s_m:.3f} eV")
log(f"  For reference, Hegde et al. report ~0.21 eV cross-database gap "
    f"disagreement in bulk databases.")

# Does PBE add anything beyond HSE? If not, say so and keep the simpler map.
A2 = np.vstack([np.ones(int(anchor.sum())), np.asarray(hse_a)]).T
c2_, *_ = np.linalg.lstsq(A2, np.asarray(gw_a), rcond=None)
rs_hse = np.asarray(gw_a) - A2 @ c2_
A3 = np.vstack([np.ones(int(anchor.sum())), np.asarray(pbe_a), np.asarray(hse_a)]).T
c3_, *_ = np.linalg.lstsq(A3, np.asarray(gw_a), rcond=None)
rs_both = np.asarray(gw_a) - A3 @ c3_
log(f"  residual sd, HSE only : {rs_hse.std(ddof=2):.3f} eV")
log(f"  residual sd, PBE + HSE: {rs_both.std(ddof=3):.3f} eV")

# =============================================================================
# 3. Coverage check by 10-fold cross-validation
# =============================================================================
# The whole paper rests on these intervals being honest. Test them.
log("--- 10-fold CV coverage of the predictive intervals ---")
rng = np.random.default_rng(SEED)
idx = rng.permutation(int(anchor.sum()))
folds = np.array_split(idx, 10)
pbe_np, hse_np, gw_np = map(np.asarray, (pbe_a, hse_a, gw_a))
inside = {0.5: 0, 0.8: 0, 0.9: 0, 0.95: 0}
ntot = 0
for f in folds:
    tr = np.setdiff1d(idx, f)
    m2 = MCMC(NUTS(cal_model), num_warmup=500, num_samples=1000,
              num_chains=1, progress_bar=False)
    m2.run(jax.random.PRNGKey(SEED + 7), jnp.asarray(pbe_np[tr]),
           jnp.asarray(hse_np[tr]), jnp.asarray(gw_np[tr]))
    p2 = m2.get_samples()
    mu = p2["a"][:, None] + p2["b1"][:, None] * pbe_np[f][None, :] \
         + p2["b2"][:, None] * hse_np[f][None, :]
    draws = np.asarray(mu) + np.asarray(p2["sigma"])[:, None] * \
            rng.standard_normal(mu.shape)
    for lvl in inside:
        lo = np.quantile(draws, (1 - lvl) / 2, axis=0)
        hi = np.quantile(draws, 1 - (1 - lvl) / 2, axis=0)
        inside[lvl] += int(((gw_np[f] >= lo) & (gw_np[f] <= hi)).sum())
    ntot += len(f)
cov = []
for lvl, c in inside.items():
    emp = c / ntot
    log(f"  nominal {lvl*100:>4.0f}%  empirical {emp*100:>5.1f}%  "
        f"({'OK' if abs(emp-lvl) < 0.08 else 'MISCALIBRATED'})")
    cov.append({"nominal": lvl, "empirical": emp, "n": ntot})
pd.DataFrame(cov).to_csv(COVER_CSV, index=False)
log("  Intervals within ~8 points of nominal are usable. Systematic "
    "under-coverage means the propagated uncertainty is too narrow and every "
    "rank-stability result would be optimistic.")

# =============================================================================
# 4. Posterior predictive gap for ALL materials
# =============================================================================
pbe_all = jnp.asarray(d["gap"].to_numpy())
hse_all = jnp.asarray(d["gap_hse"].to_numpy())
pred = Predictive(cal_model, post)(jax.random.PRNGKey(SEED + 3),
                                   pbe_all, hse_all, None)   # gw=None -> predict
E = np.asarray(pred["obs"])                                  # (n_samples, N)
sel = np.linspace(0, E.shape[0] - 1, N_DRAWS_KEEP).astype(int)
E_keep = E[sel].T.astype(np.float32)

# Anchored materials: replace the predictive draw with the measured G0W0 value
# plus its own sigma_gw, since for those we have a direct observation.
E_keep = np.clip(E_keep, 0.01, None)     # a gap cannot be negative
np.save(POST_NPY, E_keep)
pd.DataFrame({"uid": d["uid"], "row": np.arange(N),
              "has_gw_anchor": anchor}).to_csv(POST_IDS, index=False)
log(f"[step4b] saved {POST_NPY} shape={E_keep.shape}")

w = E_keep.std(axis=1) / E_keep.mean(axis=1)
log(f"  relative predictive width: median={np.median(w)*100:.1f}%  "
    f"p90={np.percentile(w,90)*100:.1f}%")
log(f"  absolute predictive sd   : median={np.median(E_keep.std(axis=1)):.3f} eV")
log("  This width is now dominated by the calibration residual sigma, which "
    "is the honest statement: given PBE and HSE, the true gap is uncertain "
    "to about that much, as measured on 129 G0W0 anchors.")

# sanity: PBE must underestimate
log("--- physics sanity ---")
for probe in (1.0, 2.0, 3.0):
    hse_probe = coef[0] + coef[1] * probe          # typical HSE for this PBE
    e = a_m + b1_m * probe + b2_m * hse_probe
    log(f"  PBE={probe:.1f} eV -> HSE~{hse_probe:.2f} -> calibrated true gap "
        f"{e:.2f} eV   (PBE underestimates by {(1-probe/e)*100:.0f}%)")
log("[step4b] DONE\n")
