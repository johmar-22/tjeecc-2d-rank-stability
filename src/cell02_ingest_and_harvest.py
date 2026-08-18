# =============================================================================
# TJEECC - CELL 2: STEP 1 (ingest c2db.db) + STEP 2 (harvest per-material JSON)
#
# RUN THIS NOW, while the extracted C2DB_TREE on scratch still exists.
# It produces data/processed/c2db_enriched.parquet on Drive, after which the
# 140-minute extraction never needs repeating.
#
# Requires cell00_bootstrap.py. Uses: SUBDIRS, C2DB_TREE, log.
# =============================================================================

import sqlite3, json, os, re, time, math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DB_PATH       = SUBDIRS["raw"] / "c2db.db"
SCREENED_PQ   = SUBDIRS["interim"]   / "c2db_screened.parquet"
ENRICHED_PQ   = SUBDIRS["processed"] / "c2db_enriched.parquet"
WATERFALL_CSV = SUBDIRS["processed"] / "table7_screening_waterfall.csv"

# =============================================================================
# STEP 1 - ingest c2db.db into a tidy, screened dataframe
# =============================================================================
log("=" * 70)
log("STEP 1: ingesting c2db.db")
t0 = time.time()

con = sqlite3.connect(DB_PATH)

num = pd.read_sql_query("SELECT id, key, value FROM number_key_values", con)
txt = pd.read_sql_query("SELECT id, key, value FROM text_key_values", con)

# Pivot separately then concat: pivoting them together produces a MultiIndex mess.
num_w = num.pivot_table(index="id", columns="key", values="value", aggfunc="first")
txt_w = txt.pivot_table(index="id", columns="key", values="value", aggfunc="first")
df = pd.concat([num_w, txt_w], axis=1)
df.index.name = "id"
log(f"  pivoted: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Reduced formula from the species table (systems.numbers is an opaque blob).
sp = pd.read_sql_query("SELECT id, Z, n FROM species", con)
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
sp["sym"] = sp["Z"].map(PT)
formula = (sp.sort_values(["id", "Z"])
             .groupby("id")
             .apply(lambda g: "".join(f"{s}{int(n)}" if n > 1 else s
                                      for s, n in zip(g["sym"], g["n"])),
                    include_groups=False)
             .rename("formula"))
natoms = sp.groupby("id")["n"].sum().rename("natoms")
df = df.join(formula).join(natoms)
con.close()

# --- base material id --------------------------------------------------------
# uid format here is "<prototype><Formula>-<magstate>", e.g. "2AgCrAs2O6-2".
# The LEADING integer is part of the structure identity (prototype index) and
# must be kept; only the trailing "-<magstate>" is stripped. Verified against
# the tarball layout materials/<stoich>/<folder>/<magstate>/.
df["base_uid"] = df["uid"].str.rsplit("-", n=1).str[0]
df["magstate"] = df["uid"].str.rsplit("-", n=1).str[1]

log("  uid -> base_uid sample (EYEBALL THESE before trusting the dedup):")
for u, b in df[["uid", "base_uid"]].head(12).itertuples(index=False):
    log(f"    {u:<28} -> {b}")

# --- numeric hygiene ---------------------------------------------------------
for c in ("gap", "gap_hse", "gap_gw", "emass_cbm", "emass_vbm", "ehull",
          "hform", "thickness", "alphax_el", "alphay_el", "alphaz_el",
          "is_magnetic", "minhessianeig"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

n_inf_cbm = int(np.isinf(df["emass_cbm"]).sum())
n_inf_vbm = int(np.isinf(df["emass_vbm"]).sum())
log(f"  infinite emass_cbm: {n_inf_cbm}   emass_vbm: {n_inf_vbm}")

# --- screening waterfall -----------------------------------------------------
steps, mask = [], pd.Series(True, index=df.index)

def gate(name, cond):
    global mask
    mask = mask & cond.fillna(False)
    steps.append((name, int(mask.sum())))
    log(f"  {name:<52}: {int(mask.sum()):>6,}")

log("--- screening waterfall ---")
steps.append(("all rows", len(df)))
log(f"  {'all rows':<52}: {len(df):>6,}")
gate("dyn_stab == 'Yes'",                 df["dyn_stab"].eq("Yes"))
gate("gap > 0.3 eV",                      df["gap"] > 0.3)
gate("ehull < 0.2 eV/atom",               df["ehull"] < 0.2)
gate("emass_cbm & emass_vbm present",     df["emass_cbm"].notna() & df["emass_vbm"].notna())
gate("0.01 < m* < 10 (drops inf)",        df["emass_cbm"].between(0.01, 10, "neither")
                                        & df["emass_vbm"].between(0.01, 10, "neither"))
gate("gap_hse present",                   df["gap_hse"].notna())
# Mandatory: magnetic 3d-TM monolayers carry a Hubbard-U sensitivity our
# PBE/HSE/GW model does not represent (Pakdel et al., npj Comput Mater 2025).
gate("is_magnetic == 0",                  df["is_magnetic"].eq(0))

scr = df[mask].copy()

# dedup: lowest formation energy per base material
n_before = len(scr)
scr = scr.sort_values("hform").drop_duplicates("base_uid", keep="first")
log(f"  dedup on base_uid (keep min hform)          : {len(scr):>6,} "
    f"({n_before - len(scr)} duplicate magnetic states dropped)")
steps.append(("dedup on base_uid", len(scr)))

n_pol = int(scr["alphax_el"].notna().sum())
n_gw  = int(scr["gap_gw"].notna().sum())
log(f"  ... of which have polarizability (full device model): {n_pol:,}")
log(f"  ... of which have a G0W0 gap (calibration anchor)   : {n_gw:,}")
steps += [("+ polarizability", n_pol), ("+ G0W0 gap", n_gw)]

pd.DataFrame(steps, columns=["stage", "n"]).to_csv(WATERFALL_CSV, index=False)
scr.to_parquet(SCREENED_PQ)
log(f"  saved {SCREENED_PQ}   ({time.time()-t0:.0f} s)")

# =============================================================================
# STEP 2 - harvest emass.json and results-asr.stiffness.json from the tree
# =============================================================================
log("=" * 70)
log("STEP 2: harvesting per-material JSON")
t0 = time.time()

MAT_ROOT = C2DB_TREE / "materials"
if not MAT_ROOT.is_dir():
    raise FileNotFoundError(
        f"{MAT_ROOT} missing. The runtime was recycled and the extracted tree "
        f"is gone. Re-run cell01 (see the speed note in the plan) before this."
    )

def walk_material_dirs(root: Path):
    """Yield (stoich, folder, magstate, path) for materials/<s>/<f>/<m>/."""
    with os.scandir(root) as l1:
        for s in l1:
            if not s.is_dir():
                continue
            with os.scandir(s.path) as l2:
                for f in l2:
                    if not f.is_dir():
                        continue
                    with os.scandir(f.path) as l3:
                        for m in l3:
                            if m.is_dir():
                                yield s.name, f.name, m.name, Path(m.path)

def dig(d, *keys):
    """Tolerant getter across ASR schema versions."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d

rows, n_bad_json, n_missing_keys = [], 0, 0
dirs = list(walk_material_dirs(MAT_ROOT))
log(f"  found {len(dirs):,} material/magstate directories")

for stoich, folder, magstate, p in tqdm(dirs, desc="harvest"):
    rec = {"stoich": stoich, "folder": folder, "magstate": magstate,
           "uid": f"{folder}-{magstate}"}

    fe = p / "emass.json"
    if fe.exists():
        try:
            d = json.loads(fe.read_text())
            for edge in ("cbm", "vbm"):
                e = d.get(edge)
                if not isinstance(e, dict):
                    n_missing_keys += 1
                    continue
                mn, mx = e.get("min_emass"), e.get("max_emass")
                rec[f"{edge}_m_dos_file"]  = e.get("m_dos")
                rec[f"{edge}_min_emass"]   = mn
                rec[f"{edge}_max_emass"]   = mx
                rec[f"{edge}_warping"]     = e.get("warping")
                rec[f"{edge}_barrier"]     = e.get("barrier_found")
                if (mn and mx and np.isfinite(mn) and np.isfinite(mx)
                        and mn > 0 and mx > 0):
                    # 2D parabolic bands with principal masses m1, m2:
                    #   DOS mass          = geometric mean  sqrt(m1*m2)
                    #   conductivity mass = harmonic mean   2*m1*m2/(m1+m2)
                    # These coincide only for isotropic bands. n_s and C_Q use
                    # m_dos; the Natori current uses m_cond.
                    rec[f"{edge}_m_dos_check"] = math.sqrt(mn * mx)
                    rec[f"{edge}_m_cond"]      = 2 * mn * mx / (mn + mx)
                    rec[f"{edge}_anisotropy"]  = mx / mn
        except (json.JSONDecodeError, OSError):
            n_bad_json += 1

    fs = p / "results-asr.stiffness.json"
    if fs.exists():
        try:
            d = json.loads(fs.read_text())
            data = (dig(d, "kwargs", "data") or dig(d, "data") or d)
            if isinstance(data, dict):
                c11, c22 = data.get("c_11"), data.get("c_22")
                rec["c_11"] = c11
                rec["c_12"] = data.get("c_12")
                rec["c_22"] = c22
                rec["speed_of_sound_x"] = data.get("speed_of_sound_x")
                if c11 is not None and c22 is not None:
                    rec["C_2D"] = 0.5 * (c11 + c22)   # in-plane 2D stiffness, N/m
        except (json.JSONDecodeError, OSError):
            n_bad_json += 1

    rec["has_defpot"] = (p / "results-asr.deformationpotentials.json").exists()
    rows.append(rec)

h = pd.DataFrame(rows)
log(f"  harvested {len(h):,} records in {time.time()-t0:.0f} s")
log(f"  malformed JSON: {n_bad_json}   missing band-edge keys: {n_missing_keys}")
log(f"  with emass    : {h['cbm_m_cond'].notna().sum():,}")
log(f"  with stiffness: {h['C_2D'].notna().sum():,}" if "C_2D" in h else "  with stiffness: 0")
log(f"  with defpot   : {int(h['has_defpot'].sum()):,}  <- expected 0")

# --- self-consistency: sqrt(min*max) must reproduce the stored m_dos ---------
log("--- validation: m_dos self-consistency ---")
for edge in ("cbm", "vbm"):
    sub = h.dropna(subset=[f"{edge}_m_dos_file", f"{edge}_m_dos_check"])
    if len(sub) == 0:
        continue
    rel = ((sub[f"{edge}_m_dos_check"] - sub[f"{edge}_m_dos_file"]).abs()
           / sub[f"{edge}_m_dos_file"])
    frac = float((rel < 0.02).mean())
    log(f"  {edge}: {frac*100:.1f}% within 2%  (n={len(sub):,})")
    if frac < 0.95:
        log(f"  *** FAIL: {edge} self-consistency below 95%. The min/max mass "
            f"extraction is wrong. Do not proceed to Step 5.")
        log(f"      worst offenders:\n{sub.assign(rel=rel).nlargest(5,'rel')[[f'{edge}_m_dos_file',f'{edge}_m_dos_check','rel']]}")

# harmonic <= geometric always; a violation means min/max are swapped
for edge in ("cbm", "vbm"):
    s = h.dropna(subset=[f"{edge}_m_cond", f"{edge}_m_dos_check"])
    bad = int((s[f"{edge}_m_cond"] > s[f"{edge}_m_dos_check"] * 1.001).sum())
    log(f"  {edge}: m_cond > m_dos violations: {bad}  (must be 0)")

# --- join to the screened set ------------------------------------------------
enr = scr.reset_index().merge(h, on="uid", how="left", suffixes=("", "_tree"))
cov_e = float(enr["cbm_m_cond"].notna().mean())
cov_s = float(enr["C_2D"].notna().mean()) if "C_2D" in enr else 0.0
log("--- join coverage against the screened set ---")
log(f"  screened materials      : {len(enr):,}")
log(f"  with tree effective mass: {enr['cbm_m_cond'].notna().sum():,} ({cov_e*100:.1f}%)")
log(f"  with tree stiffness     : {int(enr['C_2D'].notna().sum()) if 'C_2D' in enr else 0:,} ({cov_s*100:.1f}%)")
if cov_e < 0.5:
    log("  *** Join coverage is low. Check the uid construction "
        "f'{folder}-{magstate}' against df['uid'] before proceeding.")

enr.to_parquet(ENRICHED_PQ)
log(f"  saved {ENRICHED_PQ}")
log("  The extracted tree is no longer needed. Everything downstream reads "
    "this Parquet.")
log("=" * 70)
log("CELL 2 DONE\n")
