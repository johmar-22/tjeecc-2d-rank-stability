# =============================================================================
# TJEECC - CELL 5: repair the formula column, redo the validation preflight,
#                  and estimate cross-database uncertainty scales for m* and eps
#
# Fixes a Cell 2 bug: the formula was built by sorting species by atomic
# number, which yields "S2Mo" for MoS2. Only species whose Z-order happens to
# match conventional order (MoTe2, BN) came out right. Joins were on uid, so
# nothing downstream is corrupted; only labels were wrong.
#
# Also harvests JARVIS avg_elec_mass / avg_hole_mass / epsx,y to set the
# log-normal uncertainty scales that Step 4 needs for effective mass and
# permittivity, which have no multi-functional axis inside C2DB.
#
# Requires cell00_bootstrap.py.
# =============================================================================

import sqlite3, json
import numpy as np, pandas as pd
from pymatgen.core import Composition

ENRICHED_PQ = SUBDIRS["processed"] / "c2db_enriched.parquet"
DEVICE_PQ   = SUBDIRS["processed"] / "c2db_device_ready.parquet"
MATCH_PQ    = SUBDIRS["processed"] / "c2db_jarvis_matched.parquet"
SCALES_JSON = SUBDIRS["processed"] / "uncertainty_scales.json"
PREFLIGHT   = SUBDIRS["processed"] / "validation_material_check.csv"

# =============================================================================
# 1. Correct reduced formulae from the species table
# =============================================================================
PT = {1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",11:"Na",
      12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",
      21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",
      30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",
      39:"Y",40:"Zr",41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",
      48:"Cd",49:"In",50:"Sn",51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",
      57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",
      66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",72:"Hf",73:"Ta",74:"W",
      75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",81:"Tl",82:"Pb",83:"Bi",
      84:"Po",85:"At",86:"Rn"}

con = sqlite3.connect(SUBDIRS["raw"] / "c2db.db")
sp = pd.read_sql_query("SELECT id, Z, n FROM species", con)
con.close()
sp["sym"] = sp["Z"].map(PT)

# Composition.reduced_formula applies proper element ordering and reduction:
# {Mo:1, S:2} -> "MoS2"; {P:4} -> "P".  Also keep the unreduced cell formula,
# since C2DB uids encode formula units (e.g. "4P-1" is 4 P per cell).
comp = (sp.groupby("id")
          .apply(lambda g: Composition({s: float(n)
                                        for s, n in zip(g["sym"], g["n"])}),
                 include_groups=False)
          .rename("comp"))
# pymatgen orders reduced_formula by electronegativity, so MoTe2 renders as
# "Te2Mo". Never match on that string. comp_key is an order-independent
# canonical key built from the reduced composition, and it is what we join on.
def comp_key(c: Composition) -> str:
    rc = c.reduced_composition.get_el_amt_dict()
    return "-".join(f"{el}{int(round(n))}" for el, n in sorted(rc.items()))

fx = pd.DataFrame({
    "formula_reduced": comp.map(lambda c: c.reduced_formula),
    "formula_cell":    comp.map(lambda c: c.formula.replace(" ", "")),
    "natoms_cell":     comp.map(lambda c: int(sum(c.values()))),
    "comp_key":        comp.map(comp_key),
})

log("--- formula repair: before vs after ---")
enr = pd.read_parquet(ENRICHED_PQ)
enr = enr.drop(columns=[c for c in fx.columns if c in enr.columns])
enr = enr.merge(fx, left_on="id", right_index=True, how="left")
for u in ("1MoS2-1", "1WS2-1", "1WSe2-1", "1MoTe2-1", "1BN-1"):
    r = enr[enr["uid"] == u]
    if len(r):
        r = r.iloc[0]
        log(f"  {u:<12} old='{r['formula']}'  ->  new='{r['formula_reduced']}'")
enr.to_parquet(ENRICHED_PQ)

dev = pd.read_parquet(DEVICE_PQ)
dev = dev.drop(columns=[c for c in fx.columns if c in dev.columns])
dev = dev.merge(fx, left_on="id", right_index=True, how="left")
dev.to_parquet(DEVICE_PQ)
log(f"  repaired {ENRICHED_PQ.name} ({len(enr):,}) and {DEVICE_PQ.name} ({len(dev):,})")

# =============================================================================
# 2. Validation preflight, redone correctly
# =============================================================================
log("--- preflight (corrected): Step 6 validation channels ---")
REFS = ["MoS2", "WS2", "WSe2", "MoSe2", "MoTe2", "WTe2", "P", "InSe",
        "GaSe", "SnS2", "ZrS2", "HfS2", "BN", "SnSe", "Bi2Se3"]
rows = []
for f in REFS:
    key = comp_key(Composition(f))          # order-independent match
    a = enr[enr["comp_key"] == key]
    d = dev[dev["comp_key"] == key]
    rows.append({"formula": f, "comp_key": key,
                 "in_screened": len(a), "in_device_ready": len(d),
                 "uids": ",".join(d["uid"].astype(str).head(4)),
                 "cbm_aniso_min": float(a["cbm_anisotropy"].min()) if len(a) else np.nan,
                 "cbm_rel_min":   float(a["cbm_rel"].min()) if len(a) and "cbm_rel" in a else np.nan})
pre = pd.DataFrame(rows)
for r in pre.itertuples(index=False):
    tag = "OK  " if r.in_device_ready > 0 else ("CUT " if r.in_screened > 0 else "ABSENT")
    log(f"  {tag} {r.formula:<7} screened={r.in_screened:<3} device_ready={r.in_device_ready:<3} "
        f"min_aniso={r.cbm_aniso_min:.2f}  min_rel={r.cbm_rel_min:.3f}  {r.uids}")
pre.to_csv(PREFLIGHT, index=False)

log("  OK     = usable as a Step 6 validation reference")
log("  CUT    = present but removed by the parabolicity filter (a real, "
    "reportable Limitation)")
log("  ABSENT = not in C2DB under this composition at all")

# =============================================================================
# 3. Cross-database uncertainty scales for m* and permittivity
# =============================================================================
# C2DB gives three functionals for the GAP but only one value for effective
# mass and polarizability. The 131 C2DB<->JARVIS matches are the only handle
# we have on sigma_impl for those two, so extract it here rather than guessing.
log("--- cross-database uncertainty scales ---")
mt = pd.read_parquet(MATCH_PQ)
jd = json.loads((SUBDIRS["jarvis"] / "dft_2d.json").read_text())
jdf = pd.DataFrame(jd).set_index("jid")
for c in ("avg_elec_mass", "avg_hole_mass", "epsx", "epsy", "mbj_bandgap",
          "optb88vdw_bandgap", "hse_gap"):
    if c in jdf.columns:
        jdf[c] = pd.to_numeric(jdf[c], errors="coerce")

mt = mt[mt["jid"].notna()].copy()
for c in ("avg_elec_mass", "avg_hole_mass", "epsx", "epsy", "mbj_bandgap", "hse_gap"):
    if c in jdf.columns:
        mt[f"j_{c}"] = mt["jid"].map(jdf[c])

scales = {}

def logratio_scale(a, b, name):
    """sigma of ln(a/b): the multiplicative log-normal scale for Step 4."""
    d = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d["a"] > 0) & (d["b"] > 0)]
    if len(d) < 20:
        log(f"  {name:<28}: n={len(d)} too few, no estimate")
        return None
    lr = np.log(d["a"].to_numpy() / d["b"].to_numpy())
    s = {"n": int(len(d)), "median_ratio": float(np.exp(np.median(lr))),
         "sigma_ln": float(np.std(lr, ddof=1)),
         "mad_ln": float(np.median(np.abs(lr - np.median(lr))) * 1.4826)}
    log(f"  {name:<28}: n={s['n']:<4} median ratio={s['median_ratio']:.3f}  "
        f"sigma_ln={s['sigma_ln']:.3f}  robust sigma_ln={s['mad_ln']:.3f}")
    scales[name] = s
    return s

# JARVIS avg_elec_mass / avg_hole_mass vs C2DB DOS masses
if "j_avg_elec_mass" in mt:
    logratio_scale(mt["j_avg_elec_mass"], mt["cbm_m_dos_file"], "electron_mass_c2db_vs_jarvis")
if "j_avg_hole_mass" in mt:
    logratio_scale(mt["j_avg_hole_mass"].abs(), mt["vbm_m_dos_file"], "hole_mass_c2db_vs_jarvis")

# permittivity: C2DB polarizability -> eps_par = 1 + 4*pi*alpha/t
if {"alphax_el", "alphay_el", "thickness"}.issubset(mt.columns):
    alpha_ip = 0.5 * (mt["alphax_el"] + mt["alphay_el"])
    eps_c2db = 1.0 + 4.0 * np.pi * alpha_ip / mt["thickness"]
    mt["eps_c2db"] = eps_c2db
    if "j_epsx" in mt and "j_epsy" in mt:
        logratio_scale(0.5 * (mt["j_epsx"] + mt["j_epsy"]), eps_c2db,
                       "permittivity_c2db_vs_jarvis")

# gap axis, for the record
if "j_mbj_bandgap" in mt:
    logratio_scale(mt["j_mbj_bandgap"], mt["gap_hse"], "gap_tbmbj_vs_hse")

SCALES_JSON.write_text(json.dumps(scales, indent=2))
log(f"  saved {SCALES_JSON}")

# =============================================================================
# 4. Verdict on the secondary axis
# =============================================================================
n_gap = int(mt["j_mbj_bandgap"].notna().sum()) if "j_mbj_bandgap" in mt else 0
n_m   = int(mt["j_avg_elec_mass"].notna().sum()) if "j_avg_elec_mass" in mt else 0
log("--- verdict on the JARVIS axis ---")
log(f"  matched pairs                      : {len(mt):,}")
log(f"  with a TBmBJ gap (gap axis)        : {n_gap}")
log(f"  with avg_elec_mass (mass axis)     : {n_m}")
log("  RECOMMENDATION: the TBmBJ gap overlap is too thin to carry a "
    "distribution. Demote the cross-database GAP comparison to a one-sentence "
    "consistency check plus a supplementary figure, and rely on the "
    "within-C2DB PBE/HSE/G0W0 axis for the gap, which is the controlled "
    "comparison and the stronger result. Keep the JARVIS MASS and "
    "PERMITTIVITY comparison: it is the only empirical basis for sigma on "
    "those two parameters and it feeds Step 4 directly.")
log("[cell5] DONE\n")
