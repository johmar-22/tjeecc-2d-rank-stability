# =============================================================================
# TJEECC - CELL 7 / PLAN STEP 4: hierarchical measurement-error model for the
#                                band gap across PBE, HSE06 and G0W0
#
# Treats the three functionals as noisy measurements of one latent gap per
# material, with G0W0 fixed as the reference scale so the model is
# identifiable. The 129 G0W0 anchors calibrate the PBE and HSE offsets, which
# are then applied WITH uncertainty to the materials that have only PBE+HSE.
#
# This is the axis Chen et al. (2025) could not build: they varied a single
# continuous alpha_HFX knob as a proxy for functional choice. We use three
# genuinely distinct functionals as measurements of a latent truth.
#
# Requires cell00_bootstrap.py (MCMC_CHAIN_METHOD, jax_enable_x64).
# =============================================================================

import json, time
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam
import arviz as az
from pymatgen.core import Composition

DEVICE_PQ  = SUBDIRS["processed"] / "c2db_device_ready.parquet"
POST_NPY   = SUBDIRS["processed"] / "gap_posterior.npy"
POST_IDS   = SUBDIRS["processed"] / "gap_posterior_uids.csv"
SUMMARY_CSV= SUBDIRS["processed"] / "step4_posterior_summary.csv"
SCALES_JSON= SUBDIRS["processed"] / "uncertainty_scales.json"

N_DRAWS_KEEP = 1000

dev = pd.read_parquet(DEVICE_PQ)
log(f"[step4] device-ready: {len(dev):,}")

# =============================================================================
# 0. PREFLIGHT: are C2DB effective masses right in the centre?
# =============================================================================
# C2DB is now the ONLY source of m*, and its sigma is assumed rather than
# measured. The least we can do is confirm the central values against
# literature for well-characterised monolayers.
log("--- preflight: C2DB m_dos vs literature (m_0) ---")
LIT = {"MoS2": 0.47, "WS2": 0.31, "WSe2": 0.34, "MoSe2": 0.55,
       "MoTe2": 0.55, "BN": 0.90}
def ckey(f):
    rc = Composition(f).reduced_composition.get_el_amt_dict()
    return "-".join(f"{e}{int(round(n))}" for e, n in sorted(rc.items()))

ok = 0
for f, lit in LIT.items():
    sub = dev[dev["comp_key"] == ckey(f)]
    if not len(sub):
        log(f"  {f:<7} absent"); continue
    m = float(sub.iloc[0]["cbm_m_dos_file"])
    rel = abs(m - lit) / lit
    flag = "OK " if rel < 0.35 else "OFF"
    ok += rel < 0.35
    log(f"  {flag} {f:<7} C2DB={m:.3f}  lit={lit:.2f}  rel={rel*100:.0f}%")
log(f"  {ok}/{len(LIT)} within 35% of literature. PBE masses are typically "
    f"within 10-30% of experiment, so this is the expected band. If most are "
    f"OFF, stop: the mass column is wrong and Step 5 will inherit it.")

# =============================================================================
# 1. Assemble observations
# =============================================================================
d = dev[["uid", "gap", "gap_hse", "gap_gw"]].copy()
for c in ("gap", "gap_hse", "gap_gw"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["gap", "gap_hse"]).reset_index(drop=True)

gw_mask = d["gap_gw"].notna().to_numpy()
# CRITICAL: NaN in an observation poisons the gradient even under a mask,
# because the likelihood is still evaluated before masking. Fill first.
gap_gw_filled = np.nan_to_num(d["gap_gw"].to_numpy(), nan=0.0)

y_pbe = jnp.asarray(d["gap"].to_numpy())
y_hse = jnp.asarray(d["gap_hse"].to_numpy())
y_gw  = jnp.asarray(gap_gw_filled)
m_gw  = jnp.asarray(gw_mask)
N = len(d)
log(f"[step4] N={N:,} materials, {int(gw_mask.sum())} with a G0W0 anchor")
log(f"  PBE  gap: median={d['gap'].median():.3f}  range=[{d['gap'].min():.2f},{d['gap'].max():.2f}]")
log(f"  HSE  gap: median={d['gap_hse'].median():.3f}")
log(f"  G0W0 gap: median={d.loc[gw_mask,'gap_gw'].median():.3f}")

# =============================================================================
# 2. Model
# =============================================================================
# E_true[m] ~ LogNormal(mu_pop, tau_pop)      positive, right-skewed like gaps
# y_k[m]    ~ Normal(alpha_k + beta_k * E_true[m], sigma_k)
# G0W0 is the reference: alpha_gw = 0, beta_gw = 1. Stated as an assumption in
# Methods; this is a RELATIVE uncertainty model, GW is not error-free.
#
# sigma priors are informed by Hegde et al. (2023): cross-database band-gap
# median relative absolute difference ~9%, about 0.21 eV. Applied here as a
# scale, noting it derives from bulk 3D databases.
HEGDE_SIGMA = 0.21

def model(y_pbe, y_hse, y_gw, m_gw):
    n = y_pbe.shape[0]
    mu_pop  = numpyro.sample("mu_pop",  dist.Normal(0.3, 1.0))
    tau_pop = numpyro.sample("tau_pop", dist.HalfNormal(1.0))

    # non-centred: sampling E_true directly gives a funnel and divergences
    with numpyro.plate("mat", n):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    E_true = numpyro.deterministic("E_true", jnp.exp(mu_pop + tau_pop * z))

    a_pbe = numpyro.sample("alpha_pbe", dist.Normal(0.0, 0.5))
    b_pbe = numpyro.sample("beta_pbe",  dist.LogNormal(0.0, 0.3))
    a_hse = numpyro.sample("alpha_hse", dist.Normal(0.0, 0.5))
    b_hse = numpyro.sample("beta_hse",  dist.LogNormal(0.0, 0.3))
    s_pbe = numpyro.sample("sigma_pbe", dist.HalfNormal(HEGDE_SIGMA * 2))
    s_hse = numpyro.sample("sigma_hse", dist.HalfNormal(HEGDE_SIGMA * 2))
    s_gw  = numpyro.sample("sigma_gw",  dist.HalfNormal(HEGDE_SIGMA * 2))

    numpyro.sample("obs_pbe", dist.Normal(a_pbe + b_pbe * E_true, s_pbe), obs=y_pbe)
    numpyro.sample("obs_hse", dist.Normal(a_hse + b_hse * E_true, s_hse), obs=y_hse)
    with numpyro.handlers.mask(mask=m_gw):
        numpyro.sample("obs_gw", dist.Normal(E_true, s_gw), obs=y_gw)


# =============================================================================
# 3. Fit
# =============================================================================
log(f"[step4] NUTS: 4 chains, 1000 warmup, 2000 samples, "
    f"chain_method='{MCMC_CHAIN_METHOD}', backend={jax.default_backend()}")
t0 = time.time()
kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=10)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4,
            chain_method=MCMC_CHAIN_METHOD, progress_bar=True)
mcmc.run(jax.random.PRNGKey(SEED), y_pbe, y_hse, y_gw, m_gw)
dt = time.time() - t0
log(f"[step4] sampling finished in {dt/60:.1f} min")

# =============================================================================
# 4. Diagnostics - hard gate
# =============================================================================
idata = az.from_numpyro(mcmc)
GLOBALS = ["mu_pop", "tau_pop", "alpha_pbe", "beta_pbe", "alpha_hse",
           "beta_hse", "sigma_pbe", "sigma_hse", "sigma_gw"]
summ = az.summary(idata, var_names=GLOBALS, round_to=4)
log("--- global parameter posteriors ---")
for line in summ.to_string().splitlines():
    log("  " + line)
summ.to_csv(SUMMARY_CSV)

full = az.summary(idata, round_to=4)
max_rhat = float(full["r_hat"].max())
min_ess  = float(full["ess_bulk"].min())
log(f"--- convergence ---")
log(f"  max r_hat   = {max_rhat:.4f}  (must be < 1.01)")
log(f"  min ess_bulk= {min_ess:.0f}    (must be > 400)")
n_div = int(mcmc.get_extra_fields().get("diverging", jnp.array([0])).sum()) \
        if mcmc.get_extra_fields() else 0
log(f"  divergences = {n_div}")

if max_rhat >= 1.01 or min_ess <= 400:
    log("  *** CONVERGENCE FAILED. Do not use these posteriors. "
        "Try target_accept_prob=0.95, max_tree_depth=12, or switch runtime "
        "(T4 float64 is slow; a CPU high-RAM runtime is often faster here).")
else:
    log("  PASS")

# --- physics check: PBE must underestimate ---
a_pbe = float(summ.loc["alpha_pbe", "mean"]); b_pbe = float(summ.loc["beta_pbe", "mean"])
a_hse = float(summ.loc["alpha_hse", "mean"]); b_hse = float(summ.loc["beta_hse", "mean"])
log("--- physics sanity ---")
log(f"  PBE: gap ~ {a_pbe:+.3f} + {b_pbe:.3f} * E_true")
log(f"  HSE: gap ~ {a_hse:+.3f} + {b_hse:.3f} * E_true")
log(f"  implied PBE underestimation at E_true=2 eV: "
    f"{(1 - (a_pbe + 2*b_pbe)/2)*100:.0f}%")
log(f"  implied HSE underestimation at E_true=2 eV: "
    f"{(1 - (a_hse + 2*b_hse)/2)*100:.0f}%")
log("  Literature expects PBE to underestimate by roughly 30-50% and HSE by "
    "much less. If PBE comes out as an OVERestimate, the data plumbing is "
    "wrong, not the physics.")

# =============================================================================
# 5. Posterior predictive check
# =============================================================================
post = mcmc.get_samples()
pp = Predictive(model, post)(jax.random.PRNGKey(SEED + 1), y_pbe, y_hse, y_gw, m_gw)
for k, obs in (("obs_pbe", d["gap"].to_numpy()), ("obs_hse", d["gap_hse"].to_numpy())):
    pred = np.asarray(pp[k]).mean(axis=0)
    r = np.corrcoef(pred, obs)[0, 1]
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    log(f"  PPC {k}: pearson r={r:.4f}  RMSE={rmse:.3f} eV")

# =============================================================================
# 6. Export per-material posterior draws
# =============================================================================
E = np.asarray(post["E_true"])                 # (n_samples, N)
idx = np.linspace(0, E.shape[0] - 1, N_DRAWS_KEEP).astype(int)
E_keep = E[idx].T.astype(np.float32)           # (N, 1000)
np.save(POST_NPY, E_keep)
pd.DataFrame({"uid": d["uid"], "row": np.arange(N)}).to_csv(POST_IDS, index=False)
log(f"[step4] saved {POST_NPY}  shape={E_keep.shape}  dtype=float32")
log(f"[step4] saved {POST_IDS}  (row order matches the .npy)")

w = E_keep.std(axis=1) / E_keep.mean(axis=1)
log(f"  per-material relative posterior width: median={np.median(w)*100:.1f}%  "
    f"p90={np.percentile(w,90)*100:.1f}%")
log("  Materials WITHOUT a G0W0 anchor should show visibly wider posteriors. "
    "If they do not, the anchors are not informing the model and the "
    "identifiability assumption needs re-examining.")
wa = w[gw_mask]; wo = w[~gw_mask]
log(f"    with G0W0   (n={len(wa)}): median width {np.median(wa)*100:.1f}%")
log(f"    without     (n={len(wo)}): median width {np.median(wo)*100:.1f}%")

# record the final Step 7 sampling scales
scales = json.loads(SCALES_JSON.read_text())
scales["polarizability_FINAL"] = {
    "sigma_ln": 0.19,
    "basis": "robust sigma_ln 0.071 from 101 C2DB<->JARVIS polarizability "
             "pairs, combined in quadrature with the systematic median offset "
             "ln(1.194)=0.177 since neither code is known to be correct",
}
SCALES_JSON.write_text(json.dumps(scales, indent=2))
log(f"[step4] updated {SCALES_JSON}")
log("[step4] DONE\n")
