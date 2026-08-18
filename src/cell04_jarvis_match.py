# =============================================================================
# TJEECC - CELL 4 / PLAN STEP 3: JARVIS-DFT 2D pull and cross-database match
#
# Builds the secondary (cross-implementation) uncertainty axis:
#   C2DB PBE/HSE/GW   vs   JARVIS OptB88vdW/TBmBJ
#
# Structures are read from c2db.db (systems table), NOT from the extracted
# tree, so this cell is independent of the extracted C2DB_TREE on scratch.
#
# This is the longest wall-clock step in the project. No GPU is used.
# Requires cell00_bootstrap.py.
# =============================================================================

import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from tqdm.auto import tqdm

DEVICE_PQ   = SUBDIRS["processed"] / "c2db_device_ready.parquet"
JARVIS_JSON = SUBDIRS["jarvis"]    / "dft_2d.json"
MATCH_PQ    = SUBDIRS["processed"] / "c2db_jarvis_matched.parquet"
PREFLIGHT   = SUBDIRS["processed"] / "validation_material_check.csv"

C_TARGET = 30.0     # common vacuum-normalised c axis, angstrom

dev = pd.read_parquet(DEVICE_PQ)
log(f"[cell4] device-ready set: {len(dev):,} materials")

# =============================================================================
# 0. PREFLIGHT: do the Step 6 validation channels survive the parabolicity cut?
# =============================================================================
# If a reference channel was excluded as non-parabolic we cannot validate
# against its published DFT-NEGF numbers, and the validation list must change.
log("--- preflight: Step 6 validation channels ---")

REFS = {
    "MoS2":  "MoS2",   "WS2":   "WS2",   "WSe2": "WSe2",
    "MoSe2": "MoSe2",  "MoTe2": "MoTe2", "phosphorene": "P",
    "InSe":  "InSe",   "hBN":   "BN",
}

full = pd.read_parquet(SUBDIRS["processed"] / "c2db_enriched.parquet")
rows = []
for label, formula in REFS.items():
    # exact formula match, not substring: 'P' must not match 'BiNbP'
    cand_all = full[full["formula"].astype(str) == formula]
    cand_dev = dev[dev["formula"].astype(str) == formula]
    rows.append({
        "label": label, "formula": formula,
        "in_screened": len(cand_all), "in_device_ready": len(cand_dev),
        "uids_device_ready": ",".join(cand_dev["uid"].astype(str).head(5)),
        "min_anisotropy_cbm": float(cand_all["cbm_anisotropy"].min())
                              if len(cand_all) else np.nan,
    })
pre = pd.DataFrame(rows)
for r in pre.itertuples(index=False):
    flag = "OK  " if r.in_device_ready > 0 else "LOST"
    log(f"  {flag} {r.label:<12} formula={r.formula:<6} "
        f"screened={r.in_screened:<3} device_ready={r.in_device_ready:<3} "
        f"min_aniso_cbm={r.min_anisotropy_cbm:.2f}")
pre.to_csv(PREFLIGHT, index=False)

_lost = pre[pre["in_device_ready"] == 0]["label"].tolist()
if _lost:
    log(f"  *** {len(_lost)} validation channel(s) lost to the parabolicity "
        f"cut: {', '.join(_lost)}")
    log(f"      Step 6 must use replacements from the device-ready set, and "
        f"the anisotropy exclusion becomes a stated Limitation, not a "
        f"footnote. Highly anisotropic channels (phosphorene above all) are "
        f"outside the scope of a parabolic-band device model.")

# =============================================================================
# 1. JARVIS dft_2d
# =============================================================================
log("--- JARVIS dft_2d ---")
if JARVIS_JSON.exists():
    log(f"  cache hit: {JARVIS_JSON}")
    jd = json.loads(JARVIS_JSON.read_text())
else:
    from jarvis.db.figshare import data as jarvis_data
    log("  downloading dft_2d (first run only)...")
    jd = jarvis_data("dft_2d")
    JARVIS_JSON.write_text(json.dumps(jd))
    log(f"  cached -> {JARVIS_JSON}")

jdf = pd.DataFrame(jd)
log(f"  {len(jdf):,} JARVIS 2D entries")
log("  AVAILABLE COLUMNS (inspect before selecting):")
for i in range(0, len(jdf.columns), 4):
    log("    " + "  ".join(f"{c:<26}" for c in jdf.columns[i:i+4]))

# JARVIS uses the STRING 'na' as a missing-value sentinel, not NaN. Arithmetic
# on these columns silently fails or raises without coercion. This is the most
# common silent error when working with this dataset.
NUMCOLS = [c for c in ("optb88vdw_bandgap", "mbj_bandgap", "epsx", "epsy",
                       "epsz", "formation_energy_peratom", "ehull",
                       "magmom_outcar") if c in jdf.columns]
for c in NUMCOLS:
    before = jdf[c].astype(str).eq("na").sum()
    jdf[c] = pd.to_numeric(jdf[c], errors="coerce")
    log(f"  coerced {c:<26}: {before:>5} 'na' sentinels -> NaN, "
        f"{jdf[c].notna().sum():>5} usable")

# =============================================================================
# 2. Structures
# =============================================================================
from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from ase.db import connect as ase_connect

def normalise_slab(s: Structure, c_target: float = C_TARGET) -> Structure:
    """Put a 2D slab in a common cell: c axis fixed, slab centred.

    C2DB pads with roughly 15 A of vacuum, JARVIS with about 20 A. Without
    this the c lattice parameters differ by more than any sane `ltol` and
    StructureMatcher rejects every pair, including MoS2 against itself.
    """
    lat = s.lattice.matrix.copy()
    # .copy() is essential: lat[2] returns a VIEW, so assigning to lat[2]
    # below would silently mutate cvec and corrupt zdir. That produced a
    # different z scaling per database and zero matches.
    cvec = lat[2].copy()
    cnorm = np.linalg.norm(cvec)
    if cnorm == 0:
        return s
    zdir = cvec / cnorm                 # unit vector, computed BEFORE mutation
    lat[2] = zdir * c_target
    cart = s.cart_coords.copy()
    z = cart @ zdir
    cart = cart + zdir * (c_target / 2.0 - (z.min() + z.max()) / 2.0)
    return Structure(lat, s.species, cart, coords_are_cartesian=True)

log("--- building structures ---")
t0 = time.time()
adb = ase_connect(str(SUBDIRS["raw"] / "c2db.db"))
c2db_structs = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for r in tqdm(dev.itertuples(index=False), total=len(dev), desc="c2db"):
        try:
            atoms = adb.get(id=int(r.id)).toatoms()
            st = AseAtomsAdaptor.get_structure(atoms)
            c2db_structs[r.uid] = normalise_slab(st)
        except Exception as e:
            log(f"  c2db structure failed for {r.uid}: {e}")
log(f"  C2DB structures: {len(c2db_structs):,} in {time.time()-t0:.0f} s")

t0 = time.time()
from jarvis.core.atoms import Atoms as JAtoms
jarvis_structs, jkeep = {}, []
for row in tqdm(jd, desc="jarvis"):
    try:
        st = JAtoms.from_dict(row["atoms"]).pymatgen_converter()
        jarvis_structs[row["jid"]] = normalise_slab(st)
        jkeep.append(row["jid"])
    except Exception:
        pass
log(f"  JARVIS structures: {len(jarvis_structs):,} in {time.time()-t0:.0f} s")

# =============================================================================
# 3. Composition-bucketed matching
# =============================================================================
# Brute force is O(N*M) ~ 1.1e6 StructureMatcher calls and takes many hours.
# Reduced composition buckets cut it to a few thousand comparisons.
def redcomp(s: Structure) -> str:
    return Composition(s.composition).reduced_formula

log("--- bucketing by reduced composition ---")
jbucket = {}
for jid, st in jarvis_structs.items():
    jbucket.setdefault(redcomp(st), []).append(jid)
log(f"  {len(jbucket):,} distinct JARVIS compositions")

matcher = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=8,
                           primitive_cell=True, attempt_supercell=False,
                           scale=True)

log("--- matching ---")
t0 = time.time()
matches, ambiguous, unmatched, n_cmp = [], [], [], 0
for uid, cs in tqdm(c2db_structs.items(), desc="match"):
    cands = jbucket.get(redcomp(cs), [])
    hits = []
    for jid in cands:
        n_cmp += 1
        try:
            if matcher.fit(cs, jarvis_structs[jid]):
                hits.append(jid)
        except Exception:
            pass
    if not hits:
        unmatched.append(uid)
    else:
        if len(hits) > 1:
            ambiguous.append((uid, hits))
        matches.append({"uid": uid, "jid": hits[0], "n_hits": len(hits),
                        "all_jids": "|".join(hits)})
log(f"  {n_cmp:,} comparisons in {(time.time()-t0)/60:.1f} min")
log(f"  matched          : {len(matches):,}")
log(f"  polymorph ambiguous (>1 JARVIS hit): {len(ambiguous):,}  "
    f"<- report this rate in the paper")
log(f"  unmatched        : {len(unmatched):,}")

if not matches:
    raise RuntimeError(
        "Zero matches. Almost certainly the vacuum normalisation: verify that "
        "normalise_slab put both sets on the same c axis, and print the "
        "lattice abc of a C2DB MoS2 and a JARVIS MoS2 side by side."
    )

m = pd.DataFrame(matches)
jsel = jdf.set_index("jid")
for col in NUMCOLS + (["spg_number"] if "spg_number" in jdf.columns else []):
    m[f"jarvis_{col}"] = m["jid"].map(jsel[col])

out = dev.merge(m, on="uid", how="left")
out.to_parquet(MATCH_PQ)
log(f"  saved {MATCH_PQ}")

# =============================================================================
# 4. Hand check: the reference channels must match
# =============================================================================
log("--- hand check on reference channels ---")
for label, formula in REFS.items():
    sub = out[(out["formula"].astype(str) == formula) & out["jid"].notna()]
    if len(sub):
        r = sub.iloc[0]
        log(f"  OK   {label:<12} {r['uid']:<16} -> {r['jid']}  "
            f"C2DB PBE={r.get('gap', np.nan):.3f}  HSE={r.get('gap_hse', np.nan):.3f}  "
            f"JARVIS OptB88={r.get('jarvis_optb88vdw_bandgap', np.nan)}  "
            f"TBmBJ={r.get('jarvis_mbj_bandgap', np.nan)}")
    else:
        log(f"  none {label:<12} (absent from device-ready set or unmatched)")

log("  If MoS2 does not appear above, the matcher settings are wrong. "
    "Do not proceed to Step 4 until at least MoS2 and WS2 match.")

n_ok = int(out["jid"].notna().sum())
n_both = int((out["jid"].notna() & out["jarvis_mbj_bandgap"].notna()).sum())
log(f"--- cross-database axis size ---")
log(f"  matched with any JARVIS entry     : {n_ok:,}")
log(f"  matched AND has a TBmBJ gap       : {n_both:,}  <- usable for sigma_impl")
if n_both < 50:
    log("  WARNING: fewer than 50 usable cross-database pairs. The secondary "
        "axis becomes anecdotal. Report it as such, or drop it and rely on "
        "the within-C2DB functional axis alone, which is the stronger result "
        "anyway.")
log("[cell4] DONE\n")
