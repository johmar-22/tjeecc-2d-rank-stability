#!/usr/bin/env python3
"""
Obtain and verify the raw input data for the TJEECC pipeline.

Two inputs are involved, and they are treated differently.

1. C2DB  (required only for pipeline stages 1-2)
   The Computational 2D Materials Database is produced by CAMD at the Technical
   University of Denmark. Its distribution page states, verbatim:

       "The full dataset is provided upon request."
       -- https://www.2dhub.org/c2db/c2db.html

   It is therefore NOT redistributed in this repository, and the author of this
   work is not at liberty to pass it on: obtain it from CAMD the same way, by
   request. This script prints how, and verifies that whatever you obtain is
   byte-identical to the snapshot the paper was computed from.

2. JARVIS-DFT 2D  (required for stage 4)
   Downloaded automatically by `jarvis-tools` on first use (~8 MB) and cached
   in cache/jarvis/. Nothing to do here. This script can pre-fetch it so a
   later run needs no network.

If you only want to reproduce the published numbers and figures, you need
NEITHER of these. Every derived table is committed in data/processed/, and
`python run_all.py --from 3` reproduces the whole analysis from them.

Usage
-----
  python download_data.py              print instructions, then verify anything present
  python download_data.py --verify     verify checksums only
  python download_data.py --jarvis     pre-fetch the JARVIS-2D cache
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RAW = Path(os.environ.get("TJEECC_ROOT", REPO)).expanduser().resolve() / "data" / "raw"

# The exact C2DB snapshot the paper was computed from.
EXPECTED = {
    "c2db.db": {
        "bytes": 74_481_664,
        "sha256": "a1a96ebc00c27dafecd7be8a1ebf7f2cbdedc8160626a674855d22ff466aef2d",
        "what": "ASE database of ~16,905 C2DB rows (properties table)",
    },
    "c2db.tar.gz": {
        "bytes": 449_561_496,
        "sha256": "e3b27bf0426ace94eb02e6ab31818c45cad5e619bbf3e1c3397fcdf2f35f9dc5",
        "what": "full C2DB material tree, 85,948 members / 54,457 files",
    },
}

INSTRUCTIONS = f"""
================================================================================
 Raw C2DB input
================================================================================
This repository does not redistribute C2DB, and cannot: the distribution page
states "The full dataset is provided upon request."
(https://www.2dhub.org/c2db/c2db.html). You must request it from CAMD at the
Technical University of Denmark yourself, exactly as the author of this work
did. This is a condition set by the database's authors, not a limitation of
this code.

  1. Browse the database and find the contact / request route at:
         https://www.2dhub.org/c2db/c2db.html
         https://c2db.fysik.dtu.dk/
     The DTU data record for the database is
         doi:10.11583/DTU.14616660

  2. Request the two artefacts below, and place them in:
         {RAW}

       c2db.db        {EXPECTED['c2db.db']['bytes']:>12,} bytes   {EXPECTED['c2db.db']['what']}
       c2db.tar.gz    {EXPECTED['c2db.tar.gz']['bytes']:>12,} bytes   {EXPECTED['c2db.tar.gz']['what']}

  3. Verify you received the same snapshot:
         python download_data.py --verify

  4. Then run:
         python run_all.py --from 1

YOU PROBABLY DO NOT NEED ANY OF THIS. Every derived table is committed in
data/processed/, so
         python run_all.py --from 3
reproduces every number and figure in the paper with no access to C2DB at all.
Stages 1-2 exist so the derivation is inspectable, not because reproducing the
results requires re-running them.

If CAMD has since published a newer snapshot the checksums below will not
match. That is expected and is not an error in this code - but the derived
numbers will then differ from the paper, and you should say so if you report
them. The committed tables in data/processed/ are the record of the snapshot
actually used.

Cite C2DB as:
  Haastrup S, Strange M, Pandey M, Deilmann T, Schmidt PS et al. 2D Materials
  2018; 5 (4): 042002.  doi:10.1088/2053-1583/aacfc1
  Gjerding MN, Taghizadeh A, Rasmussen A, Ali S, Bertoldo F et al. 2D
  Materials 2021; 8 (4): 044002.  doi:10.1088/2053-1583/ac1059
================================================================================
"""


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            pct = 100.0 * done / total if total else 100.0
            print(f"\r  hashing {path.name}: {pct:5.1f}%", end="", file=sys.stderr)
    print("\r" + " " * 40 + "\r", end="", file=sys.stderr)
    return h.hexdigest()


def verify() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Checking {RAW}\n")
    bad = 0
    for name, spec in EXPECTED.items():
        p = RAW / name
        if not p.exists():
            print(f"  ABSENT   {name}")
            bad += 1
            continue
        size = p.stat().st_size
        if size != spec["bytes"]:
            print(f"  SIZE     {name}: {size:,} bytes, expected "
                  f"{spec['bytes']:,} -> truncated download or different snapshot")
            bad += 1
            continue
        digest = sha256_of(p)
        if digest != spec["sha256"]:
            print(f"  SHA256   {name}: MISMATCH\n"
                  f"           got      {digest}\n"
                  f"           expected {spec['sha256']}\n"
                  f"           -> a different C2DB snapshot; results will not "
                  f"match the paper")
            bad += 1
        else:
            print(f"  OK       {name}  ({size:,} bytes, sha256 verified)")
    print()
    if bad:
        print(f"{bad} of {len(EXPECTED)} inputs are missing or do not match.\n"
              f"This blocks pipeline stages 1-2 only. Stages 3-18 run from the\n"
              f"committed tables in data/processed/:  python run_all.py --from 3")
    else:
        print("All raw inputs match the snapshot used in the paper.\n"
              "You can run the full pipeline:  python run_all.py --from 1")
    return 1 if bad else 0


def prefetch_jarvis() -> int:
    try:
        from jarvis.db.figshare import data as jarvis_data
    except ImportError:
        print("jarvis-tools is not installed. pip install -r requirements.txt",
              file=sys.stderr)
        return 2
    print("Fetching JARVIS-DFT 2D (dft_2d) ...")
    d = jarvis_data("dft_2d")
    print(f"  got {len(d)} entries. jarvis-tools has cached them for later runs.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="verify checksums of files already in data/raw/")
    ap.add_argument("--jarvis", action="store_true",
                    help="pre-fetch the JARVIS-2D dataset into the local cache")
    args = ap.parse_args()

    if args.jarvis:
        return prefetch_jarvis()
    if not args.verify:
        print(INSTRUCTIONS)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
