#!/usr/bin/env python3
"""
Run the TJEECC analysis pipeline end to end, or any contiguous slice of it.

The pipeline was developed as an ordered sequence of notebook cells that share
one namespace: cell00 defines ROOT, SUBDIRS, SCRATCH, C2DB_TREE, RNG, log and
MCMC_CHAIN_METHOD, and every later cell uses them. This runner reproduces that
by exec'ing each stage into a single shared globals dict, in order, exactly as
the notebook did. No Google Drive, no Colab, no network except the one-off
JARVIS-2D download in stage 4.

Usage
-----
  python run_all.py --list                 show the stage table and exit
  python run_all.py                        run every stage that your inputs allow
  python run_all.py --from 9               resume from stage 9
  python run_all.py --from 3 --to 6        run a contiguous slice
  python run_all.py --only 14 15 18        run selected stages
  python run_all.py --figures              regenerate all figures only

Stages 1 and 2 need the raw C2DB snapshot, which is NOT redistributed here
(see download_data.py). Everything from stage 3 onward runs from the derived
tables committed in data/processed/, so a reviewer can reproduce every number
and figure in the paper without obtaining C2DB.

Environment variables
---------------------
  TJEECC_ROOT         project root (default: this repository directory)
  TJEECC_SCRATCH      fast local disk for the extracted C2DB tree
  TJEECC_AUTOINSTALL  set to 1 to let cell00 pip-install missing packages
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"

# (stage number, filename stem, NOTEBOOK_NAME for the log, requirement tag)
#   raw   - needs the raw C2DB snapshot in data/raw/
#   net   - downloads JARVIS-2D on first run (~8 MB), then caches it
#   mcmc  - needs jax + numpyro
#   -     - runs from committed derived tables alone
STAGES = [
    (1,  "cell01_extract_tarball",                 "step01_ingest_c2db",       "raw"),
    (2,  "cell02_ingest_and_harvest",              "step01_ingest_c2db",       "raw"),
    (3,  "cell03_diagnose_parabolicity",           "step02_parabolicity",      "-"),
    (4,  "cell04_jarvis_match",                    "step03_jarvis_match",      "net"),
    (5,  "cell05_fix_formula_and_uncertainty_scales", "step03_jarvis_match",   "-"),
    (6,  "cell06_diagnose_jarvis_scales",          "step03_jarvis_match",      "-"),
    (7,  "cell07_step4_hierarchical_gap",          "step04_gap_hierarchical",  "mcmc"),
    (8,  "cell08_step4b_gap_calibration",          "step04b_gap_calibration",  "mcmc"),
    (9,  "cell09_anchor_representativeness",       "step04c_anchor_check",     "-"),
    (10, "cell10_step5_device_model",              "step05_device_model",      "-"),
    (11, "cell11_census_and_step6_validation",     "step06_census",            "-"),
    (12, "cell12_transport_regime_census",         "step07_transport_regime",  "-"),
    (13, "cell13_step8_complete",                  "step08_rank_stability",    "-"),
    (14, "cell14_step10_figures",                  "step10_figures",           "-"),
    (15, "cell15_sobol_split_and_refs",            "step10_figures",           "-"),
    (16, "cell16_step6_validation_real_refs",      "step06_validation",        "-"),
    (17, "cell17_diagnose_the_54",                 "step08_diagnostics",       "-"),
    (18, "cell18_figure1_pipeline",                "step10_figures",           "-"),
]

FIGURE_STAGES = [14, 15, 18]
BY_NUM = {n: (stem, nb, tag) for n, stem, nb, tag in STAGES}


def print_table() -> None:
    legend = {
        "raw":  "needs raw C2DB in data/raw/ (see download_data.py)",
        "net":  "downloads JARVIS-2D once (~8 MB), then cached",
        "mcmc": "needs jax + numpyro",
        "-":    "runs from committed derived tables",
    }
    print(f"{'#':>3}  {'stage':<42} {'needs':<5} note")
    print("-" * 100)
    for n, stem, _nb, tag in STAGES:
        print(f"{n:>3}  {stem:<42} {tag:<5} {legend[tag]}")
    print()
    print("Figures are produced by stages:", ", ".join(map(str, FIGURE_STAGES)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the TJEECC pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="show the stage table and exit")
    ap.add_argument("--from", dest="start", type=int, default=1, metavar="N")
    ap.add_argument("--to", dest="stop", type=int, default=18, metavar="M")
    ap.add_argument("--only", nargs="+", type=int, metavar="N")
    ap.add_argument("--figures", action="store_true",
                    help=f"shorthand for --only {' '.join(map(str, FIGURE_STAGES))}")
    ap.add_argument("--skip-raw", action="store_true",
                    help="skip stages 1-2 even if raw C2DB is present")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going after a failing stage (default: stop)")
    args = ap.parse_args()

    if args.list:
        print_table()
        return 0

    if args.figures:
        wanted = list(FIGURE_STAGES)
    elif args.only:
        wanted = sorted(args.only)
    else:
        wanted = [n for n in BY_NUM if args.start <= n <= args.stop]

    unknown = [n for n in wanted if n not in BY_NUM]
    if unknown:
        print(f"error: no such stage(s): {unknown}. Try --list.", file=sys.stderr)
        return 2

    # --- shared namespace, exactly as the notebook had -----------------------
    ns: dict = {"__name__": "__tjeecc__", "__file__": str(SRC / "cell00_bootstrap.py")}
    boot = SRC / "cell00_bootstrap.py"
    print(f"[run_all] bootstrap: {boot}")
    exec(compile(boot.read_text(encoding="utf-8"), str(boot), "exec"), ns)

    log = ns.get("log", print)
    raw_dir = ns["SUBDIRS"]["raw"]
    have_raw = (raw_dir / "c2db.db").exists() and (raw_dir / "c2db.tar.gz").exists()

    if args.skip_raw or not have_raw:
        dropped = [n for n in wanted if BY_NUM[n][2] == "raw"]
        if dropped:
            wanted = [n for n in wanted if n not in dropped]
            why = ("--skip-raw given" if args.skip_raw
                   else "raw C2DB not present in data/raw/")
            log(f"[run_all] skipping stage(s) {dropped}: {why}. "
                f"Later stages read the committed tables in data/processed/.")

    if not wanted:
        log("[run_all] nothing to do.")
        return 0

    log(f"[run_all] plan: stages {wanted}")
    failed: list[int] = []

    for n in wanted:
        stem, nb_name, tag = BY_NUM[n]
        path = SRC / f"{stem}.py"
        if not path.exists():
            log(f"[run_all] stage {n}: MISSING FILE {path}")
            failed.append(n)
            if not args.continue_on_error:
                break
            continue

        ns["NOTEBOOK_NAME"] = nb_name
        ns["__file__"] = str(path)
        log(f"[run_all] ===== stage {n}: {stem} (needs: {tag}) =====")
        t0 = time.time()
        try:
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
            log(f"[run_all] stage {n} finished in {time.time() - t0:.1f} s")
        except Exception:
            log(f"[run_all] stage {n} FAILED after {time.time() - t0:.1f} s")
            traceback.print_exc()
            failed.append(n)
            if not args.continue_on_error:
                log("[run_all] stopping. Re-run with --continue-on-error to "
                    "push past this, or --from N to resume.")
                break

    if failed:
        log(f"[run_all] FAILED stages: {failed}")
        return 1
    log("[run_all] all requested stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
