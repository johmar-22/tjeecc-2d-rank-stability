# =============================================================================
# TJEECC - CELL 11: gap-sensitivity census + PLAN STEP 6 validation gate
#
# Part A  Build the full device parameter table from the real data.
# Part B  CENSUS: for how many materials does gap uncertainty actually reach
#         the terminal? The gap sweep in cell10 showed on/off saturates above
#         ~1.4 eV, so most of the set may be gap-insensitive. This decides the
#         paper's headline framing and must be known BEFORE Step 7.
# Part C  STEP 6 VALIDATION GATE against published DFT-NEGF results.
#
# Run cell10 first (defines compute_fom, TECH, and the constants).
# =============================================================================

import json
import numpy as np, pandas as pd
from pymatgen.core import Composition

DEVICE_PQ  = SUBDIRS["processed"] / "c2db_device_ready.parquet"
POST_NPY   = SUBDIRS["processed"] / "gap_posterior.npy"
POST_IDS   = SUBDIRS["processed"] / "gap_posterior_uids.csv"
SCALES     = json.loads((SUBDIRS["processed"] / "uncertainty_scales.json").read_text())
PARAM_PQ   = SUBDIRS["processed"] / "device_parameters.parquet"
CENSUS_CSV = SUBDIRS["processed"] / "gap_sensitivity_census.csv"
VALID_CSV  = SUBDIRS["processed"] / "table8_validation.csv"

SIG_LN_MASS = 0.20      # ASSUMED, see cell06. Swept in Step 7.
SIG_LN_EPS  = 0.19      # measured, C2DB vs JARVIS polarizability

# =============================================================================
# PART A. Device parameter table
# =============================================================================
dev = pd.read_parquet(DEVICE_PQ)
ids = pd.read_csv(POST_IDS)
Epost = np.load(POST_NPY)                       # (N, 1000) calibrated gap draws
dev = dev.set_index("uid").loc[ids["uid"]].reset_index()
dev["E_mean"] = Epost.mean(axis=1)
dev["E_sd"]   = Epost.std(axis=1)

# in-plane permittivity from C2DB polarizability: eps = 1 + 4*pi*alpha/t
alpha_ip = 0.5 * (pd.to_numeric(dev["alphax_el"], errors="coerce")
                  + pd.to_numeric(dev["alphay_el"], errors="coerce"))
t_ang = pd.to_numeric(dev["thickness"], errors="coerce")
dev["eps_ch"] = 1.0 + 4.0 * np.pi * alpha_ip / t_ang
dev["t_ch_m"] = t_ang * 1e-10

need = ["E_mean", "eps_ch", "t_ch_m", "cbm_m_dos_file", "vbm_m_dos_file",
        "cbm_m_cond", "vbm_m_cond"]
par = dev.dropna(subset=need).reset_index(drop=True)
par = par[np.isfinite(par[need]).all(axis=1)]
log(f"[cell11] device-ready {len(dev):,} -> with full parameter set {len(par):,}")
log(f"  eps_ch: median={par['eps_ch'].median():.2f} "
    f"[{par['eps_ch'].quantile(.05):.2f}, {par['eps_ch'].quantile(.95):.2f}]")
log(f"  E_mean: median={par['E_mean'].median():.2f} eV "
    f"[{par['E_mean'].quantile(.05):.2f}, {par['E_mean'].quantile(.95):.2f}]")
# NOTE: the parquet is written AFTER the figure-of-merit columns are added,
# further down. Saving here (as an earlier version did) produced a file with
# no I_ON column, which cells 14 and 15 read and failed on.

def fom_at(Eg, me_d, mh_d, me_c, mh_c, eps, t, tech=None):
    return compute_fom(Eg=np.atleast_2d(Eg), m_dos_e=np.atleast_2d(me_d),
                       m_dos_h=np.atleast_2d(mh_d), m_cond_e=np.atleast_2d(me_c),
                       m_cond_h=np.atleast_2d(mh_c), eps_ch=np.atleast_2d(eps),
                       t_ch=np.atleast_2d(t), tech=tech or TECH)

cols = (par["E_mean"].to_numpy(), par["cbm_m_dos_file"].to_numpy(),
        par["vbm_m_dos_file"].to_numpy(), par["cbm_m_cond"].to_numpy(),
        par["vbm_m_cond"].to_numpy(), par["eps_ch"].to_numpy(),
        par["t_ch_m"].to_numpy())
central = fom_at(*cols)
for k in ("I_ON", "I_OFF", "on_off", "SS", "DIBL", "tau", "EDP"):
    par[k] = central[k].ravel()
log("--- central figures of merit ---")
log(f"  I_ON   median={np.median(par['I_ON']):.3e} A/m  (1 A/m = 1 uA/um)")
log(f"  on/off median={np.median(par['on_off']):.3e}")
log(f"  SS     median={np.median(par['SS']):.1f} mV/dec")
log(f"  tau    median={np.median(par['tau'])*1e12:.3f} ps")
par.to_parquet(PARAM_PQ)          # now includes I_ON, I_OFF, on_off, SS, tau, EDP
log(f"  parameter table (with FoMs) -> {PARAM_PQ.name}")

# =============================================================================
# PART B. CENSUS - which uncertainty actually reaches the terminal?
# =============================================================================
# One-at-a-time propagation: vary a single input over its own uncertainty,
# hold the rest at their central values, and measure the induced spread in
# log10(FoM). This is not a Sobol analysis (that comes in Step 8) but it
# answers the framing question directly and cheaply.
log("=" * 70)
log("CENSUS: which input moves which figure of merit, per material")

rng = np.random.default_rng(SEED)
NS = 128
n = len(par)

def spread(fom_key, draws):
    """log10 interquartile spread of a FoM under one perturbed input."""
    v = np.maximum(draws[fom_key], 1e-300)
    lg = np.log10(v)
    return np.percentile(lg, 84, axis=1) - np.percentile(lg, 16, axis=1)

# gap only: use the real posterior draws
# CRITICAL ALIGNMENT. Epost rows follow `dev`/`ids`, but `par` has had rows
# dropped and its index reset, so par.index no longer addresses Epost. Using
# it assigns each material ANOTHER material's gap distribution, which is what
# produced the spurious "54 gap-sensitive" set and the disagreement with
# Sobol. Always map through uid, as cells 12, 13 and 15 do.
_row_of = {u: i for i, u in enumerate(ids["uid"])}
_gi = np.array([_row_of[u] for u in par["uid"]])
assert len(_gi) == len(par)
gsel = rng.choice(Epost.shape[1], NS, replace=False)
Eg_draws = np.clip(Epost[_gi][:, gsel], 0.05, None)
# sanity: the drawn mean must match the stored central value per material
_chk = np.abs(Eg_draws.mean(axis=1) - par["E_mean"].to_numpy()) / par["E_mean"].to_numpy()
log(f"  gap-posterior alignment check: max relative deviation of draw mean "
    f"from E_mean = {float(np.nanmax(_chk)):.3f} (must be << 1)")
assert np.nanmax(_chk) < 0.25, "gap posteriors are misaligned with par"
tile = lambda a: np.repeat(a[:, None], NS, axis=1)
d_gap = compute_fom(Eg=Eg_draws,
                    m_dos_e=tile(cols[1]), m_dos_h=tile(cols[2]),
                    m_cond_e=tile(cols[3]), m_cond_h=tile(cols[4]),
                    eps_ch=tile(cols[5]), t_ch=tile(cols[6]))

# mass only (log-normal, ASSUMED sigma)
lm = rng.normal(0.0, SIG_LN_MASS, (n, NS))
d_mass = compute_fom(Eg=tile(cols[0]),
                     m_dos_e=tile(cols[1]) * np.exp(lm),
                     m_dos_h=tile(cols[2]) * np.exp(lm),
                     m_cond_e=tile(cols[3]) * np.exp(lm),
                     m_cond_h=tile(cols[4]) * np.exp(lm),
                     eps_ch=tile(cols[5]), t_ch=tile(cols[6]))

# permittivity only (measured sigma)
le = rng.normal(0.0, SIG_LN_EPS, (n, NS))
d_eps = compute_fom(Eg=tile(cols[0]), m_dos_e=tile(cols[1]), m_dos_h=tile(cols[2]),
                    m_cond_e=tile(cols[3]), m_cond_h=tile(cols[4]),
                    eps_ch=tile(cols[5]) * np.exp(le), t_ch=tile(cols[6]))

for fom in ("I_ON", "on_off"):
    par[f"spr_gap_{fom}"]  = spread(fom, d_gap)
    par[f"spr_mass_{fom}"] = spread(fom, d_mass)
    par[f"spr_eps_{fom}"]  = spread(fom, d_eps)

log("--- median +/-1sigma spread in log10(FoM) from each input alone ---")
log(f"  {'input':<14} {'I_ON':>10} {'on/off':>10}")
for lbl, key in (("gap (measured)", "gap"), ("mass (ASSUMED)", "mass"),
                 ("eps (measured)", "eps")):
    log(f"  {lbl:<14} {np.median(par[f'spr_{key}_I_ON']):>10.3f} "
        f"{np.median(par[f'spr_{key}_on_off']):>10.3f}")

SENS = 0.30      # >0.3 decades = a factor of 2 swing: materially sensitive
for fom in ("I_ON", "on_off"):
    ng = int((par[f"spr_gap_{fom}"] > SENS).sum())
    nm = int((par[f"spr_mass_{fom}"] > SENS).sum())
    log(f"--- {fom}: materials where the input swings the FoM by >2x ---")
    log(f"  gap-sensitive : {ng:>4} / {n} ({ng/n*100:.1f}%)")
    log(f"  mass-sensitive: {nm:>4} / {n} ({nm/n*100:.1f}%)")

gs = par["spr_gap_on_off"] > SENS
log(f"--- gap-sensitive subset (on/off) ---")
if gs.sum():
    log(f"  n={int(gs.sum())}  calibrated gap range "
        f"[{par.loc[gs,'E_mean'].min():.2f}, {par.loc[gs,'E_mean'].max():.2f}] eV "
        f"(median {par.loc[gs,'E_mean'].median():.2f})")
log(f"  gap-insensitive median E = {par.loc[~gs,'E_mean'].median():.2f} eV")
log("  INTERPRETATION. If the gap-sensitive fraction is small, the headline "
    "cannot be 'gap uncertainty reorders the shortlist'. It becomes: which "
    "figure of merit you rank by determines which uncertainty dominates, and "
    "for I_ON the dominant term is effective mass, whose sigma is ASSUMED. "
    "That makes the 0.10/0.20/0.40 sweep in Step 7 load-bearing, and it must "
    "be stated as the principal limitation.")
par.to_csv(CENSUS_CSV, index=False)

# =============================================================================
# PART C. STEP 6 VALIDATION GATE
# =============================================================================
# HARD GATE. Geometric-mean ratio within 2x, and >=4 of 5 within 3x.
#
# !! ACTION REQUIRED !!  The values below are placeholders in the right order
# of magnitude for ballistic monolayer n-FETs. Before the manuscript, REPLACE
# each with a number read from a specific paper and record the citation. Do
# not publish these as-is. The comparison is only meaningful at each paper's
# own L_g, V_DD and I_OFF specification, which is why those are per-row.
log("=" * 70)
log("STEP 6 VALIDATION against published DFT-NEGF (PLACEHOLDER VALUES)")

REF = [
    # formula, I_ON uA/um, L_g nm, V_DD V, I_OFF nA/um, citation key
    ("MoS2",  1250.0, 10.0, 0.65, 100.0, "REPLACE_MoS2"),
    ("WS2",   1600.0, 10.0, 0.65, 100.0, "REPLACE_WS2"),
    ("WSe2",  1100.0, 10.0, 0.65, 100.0, "REPLACE_WSe2"),
    ("MoTe2", 1400.0, 10.0, 0.65, 100.0, "REPLACE_MoTe2"),
    ("P",     2400.0, 10.0, 0.65, 100.0, "REPLACE_phosphorene"),
]
def ckey(f):
    rc = Composition(f).reduced_composition.get_el_amt_dict()
    return "-".join(f"{e}{int(round(n))}" for e, n in sorted(rc.items()))

rows = []
log(f"  {'mat':<7} {'I_ON pub':>9} {'I_ON model':>11} {'ratio':>7} "
    f"{'SS model':>9}  citation")
for f, ion_pub, lg, vdd, ioff, cite in REF:
    sub = par[par["comp_key"] == ckey(f)]
    if not len(sub):
        log(f"  {f:<7} absent from the parameter set"); continue
    r = sub.iloc[0]
    tech = dict(TECH); tech.update(L_g=lg * 1e-9, V_DD=vdd,
                                   I_OFF=ioff * 1e-9 / 1e-6)
    out = fom_at(r["E_mean"], r["cbm_m_dos_file"], r["vbm_m_dos_file"],
                 r["cbm_m_cond"], r["vbm_m_cond"], r["eps_ch"], r["t_ch_m"],
                 tech=tech)
    ion_mod = float(out["I_ON"].ravel()[0])          # A/m == uA/um
    ratio = ion_mod / ion_pub
    rows.append({"formula": f, "uid": r["uid"], "I_ON_published_uA_um": ion_pub,
                 "I_ON_model_uA_um": ion_mod, "ratio": ratio,
                 "SS_model_mV_dec": float(out["SS"].ravel()[0]),
                 "L_g_nm": lg, "V_DD": vdd, "I_OFF_nA_um": ioff,
                 "citation": cite})
    log(f"  {f:<7} {ion_pub:>9.0f} {ion_mod:>11.0f} {ratio:>7.2f} "
        f"{float(out['SS'].ravel()[0]):>9.1f}  {cite}")

val = pd.DataFrame(rows)
val.to_csv(VALID_CSV, index=False)
if len(val):
    gm = float(np.exp(np.mean(np.log(val["ratio"]))))
    within2 = int((val["ratio"].between(0.5, 2.0)).sum())
    within3 = int((val["ratio"].between(1/3, 3.0)).sum())
    log("--- validation gate ---")
    log(f"  geometric-mean ratio = {gm:.2f}   (gate: 0.5 to 2.0)")
    log(f"  within 2x: {within2}/{len(val)}    within 3x: {within3}/{len(val)}")
    passed = (0.5 <= gm <= 2.0) and within3 >= max(1, len(val) - 1)
    log(f"  {'PASS' if passed else '*** FAIL'}")
    if not passed:
        log("  Debug in this order, and do NOT tune a fudge factor:")
        log("   1. valley degeneracy g_v for the specific material "
            "(TMDs have g_v=2 at K/K'; using 1 makes you 2x low)")
        log("   2. DOS vs conductivity mass swapped somewhere")
        log("   3. I_OFF normalisation not applied, so you are comparing at "
            "different threshold voltages (the most common cause, and it "
            "produces order-of-magnitude errors)")
        log("   4. eps_ch from the polarizability conversion badly off")
    log(f"  saved {VALID_CSV}")
log("  REMINDER: replace the placeholder published values with cited numbers "
    "before this table becomes Table 8.")
log("[cell11] DONE\n")
