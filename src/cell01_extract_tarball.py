# =============================================================================
# TJEECC - CELL 1: STAGE AND EXTRACT c2db.tar.gz
#
# Run AFTER cell00_bootstrap.py. Requires: ROOT, SUBDIRS, C2DB_TREE, log.
#
# Strategy: stage the archive on fast local scratch, extract there, then walk
# the tree once and persist only a small JSON audit back to the project tree.
# Never extract onto a network filesystem (Google Drive FUSE, NFS, SMB): those
# layers collapse on ~86k small files and take hours instead of minutes.
# SCRATCH is set by cell00 and can be overridden with $TJEECC_SCRATCH.
#
# Idempotent. Re-running with an already-populated tree is a no-op.
# =============================================================================

import shutil, tarfile, json, time, os
from pathlib import Path
from tqdm.auto import tqdm

SRC_ARCHIVE = SUBDIRS["raw"] / "c2db.tar.gz"   # as obtained by download_data.py
SRC_LOCAL   = SCRATCH / "c2db.tar.gz"          # fast local staging copy
EXPECTED_BYTES = 449_561_496
AUDIT_JSON = SUBDIRS["processed"] / "c2db_tree_audit.json"

TARGET_FILES = (
    "emass.json",
    "results-asr.stiffness.json",
    "results-asr.deformationpotentials.json",
)

# --- 0. Preflight ------------------------------------------------------------
if not SRC_ARCHIVE.exists():
    raise FileNotFoundError(
        f"\n  {SRC_ARCHIVE} not found."
        f"\n  C2DB is not redistributed with this repository."
        f"\n  Run  python download_data.py  for how to obtain c2db.tar.gz"
        f"\n  ({EXPECTED_BYTES:,} bytes), then place it in data/raw/."
        f"\n"
        f"\n  This stage is only needed to regenerate the derived tables from"
        f"\n  scratch. Every result in the paper can be reproduced from the"
        f"\n  tables already committed in data/processed/ - see the README."
    )

_sz = SRC_ARCHIVE.stat().st_size
if _sz != EXPECTED_BYTES:
    log(f"WARNING: c2db.tar.gz is {_sz:,} bytes, expected {EXPECTED_BYTES:,}. "
        f"A partial download will fail mid-stream with a gzip CRC error. "
        f"Run 'python download_data.py --verify' to check the checksum.")

_free = shutil.disk_usage(str(SCRATCH)).free
if _free < 12 * 1024**3:
    raise RuntimeError(
        f"Only {_free/1024**3:.1f} GB free on {SCRATCH}. Need ~12 GB headroom "
        f"(0.45 GB archive + several GB of extracted JSON). "
        f"Free space there, or point $TJEECC_SCRATCH at a larger disk."
    )


# --- 1. Tree walk helper (scandir, not glob: ~40x faster on 80k files) -------
def audit_tree(root: Path) -> dict:
    """Single recursive pass. Returns total file count and per-name counts."""
    counts = {n: 0 for n in TARGET_FILES}
    total_files = total_dirs = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        total_dirs += 1
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        total_files += 1
                        if e.name in counts:
                            counts[e.name] += 1
        except OSError as exc:
            log(f"  scandir failed on {d}: {exc}")
    return {"files": total_files, "dirs": total_dirs, "targets": counts}


def tree_is_populated(root: Path) -> bool:
    if not root.is_dir():
        return False
    with os.scandir(root) as it:
        return any(True for _ in it)


# --- 2. Extract (skipped if already done) ------------------------------------
if tree_is_populated(C2DB_TREE):
    log(f"[cell1] {C2DB_TREE} already populated. Skipping copy and extraction.")
else:
    # 2a. Stage to local disk. Reading the archive straight off Drive FUSE
    # while decompressing is markedly slower than copy-then-extract.
    if SRC_LOCAL.exists() and SRC_LOCAL.stat().st_size == _sz:
        log(f"[cell1] Local copy already present: {SRC_LOCAL}")
    else:
        log(f"[cell1] Copying {_sz/1024**2:.0f} MB from Drive to local disk...")
        t0 = time.time()
        # tqdm.wrapattr instruments .read() so shutil does the actual copy.
        with tqdm.wrapattr(open(SRC_ARCHIVE, "rb"), "read", total=_sz,
                           unit="B", unit_scale=True, unit_divisor=1024,
                           desc="copy") as fsrc:
            with open(SRC_LOCAL, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
        log(f"[cell1] Copy done in {time.time()-t0:.0f} s "
            f"({_sz/1024**2/(time.time()-t0):.0f} MB/s)")

    # 2b. Streaming extraction.
    # mode 'r|gz' is a forward-only stream: faster than 'r:gz' because it never
    # builds the full member index. Consequence: getmembers() is unavailable,
    # so we count files by walking the tree afterwards rather than from the tar.
    C2DB_TREE.mkdir(parents=True, exist_ok=True)
    log(f"[cell1] Extracting to {C2DB_TREE} (streaming)...")
    t0 = time.time()
    n_members = 0
    rejected = []
    with tarfile.open(SRC_LOCAL, mode="r|gz", bufsize=16 * 1024 * 1024) as tf:
        bar = tqdm(unit=" members", desc="extract")
        for member in tf:
            try:
                # filter='data' (Python 3.12+) blocks absolute paths, '..'
                # traversal, device nodes, setuid bits and links pointing
                # outside the destination.
                tf.extract(member, path=C2DB_TREE, filter="data")
                n_members += 1
                bar.update(1)
            except (tarfile.FilterError, OSError) as exc:
                rejected.append((member.name, str(exc)))
        bar.close()
    dt = time.time() - t0
    log(f"[cell1] Extracted {n_members:,} members in {dt/60:.1f} min")
    if rejected:
        log(f"[cell1] {len(rejected)} members rejected by the security filter:")
        for name, why in rejected[:10]:
            log(f"    {name}: {why}")
        if len(rejected) > 10:
            log(f"    ... and {len(rejected)-10} more")

# --- 3. Coverage audit -------------------------------------------------------
log("[cell1] Auditing extracted tree...")
t0 = time.time()
audit = audit_tree(C2DB_TREE)
log(f"[cell1] Walk completed in {time.time()-t0:.0f} s")

log("--- extraction audit ---")
log(f"  directories                        : {audit['dirs']:,}")
log(f"  total files                        : {audit['files']:,}")
for name in TARGET_FILES:
    log(f"  {name:<35}: {audit['targets'][name]:,}")

# --- 4. Decision gate: deformation potentials --------------------------------
# Plan Step 5 needs the deformation potential E_1 for the quasi-ballistic
# mobility. If coverage is thin, that route is not viable and Step 8 falls
# back to a parametric mean free path.
n_dp = audit["targets"]["results-asr.deformationpotentials.json"]
DP_THRESHOLD = 500

log("--- decision gate ---")
if n_dp >= DP_THRESHOLD:
    DEFORMATION_POTENTIAL_ROUTE = True
    log(f"  PASS: {n_dp:,} deformation-potential files (>= {DP_THRESHOLD}). "
        f"Use deformation-potential mobility in Step 5.")
else:
    DEFORMATION_POTENTIAL_ROUTE = False
    log(f"  FAIL: only {n_dp:,} deformation-potential files (< {DP_THRESHOLD}). "
        f"Use the Step 8 fallback: fix the mean free path parametrically, "
        f"sweep it, and report ballistic-limit results as primary with "
        f"quasi-ballistic as a sensitivity. Record this number; it belongs "
        f"in the Methods section.")

# --- 5. Persist the audit ----------------------------------------------------
# These counts are paper numbers (Table 7) and the tree is deleted when the
# runtime recycles, so write them to Drive now.
audit_record = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "archive_bytes": _sz,
    "archive_bytes_expected": EXPECTED_BYTES,
    "directories": audit["dirs"],
    "total_files": audit["files"],
    "target_file_counts": audit["targets"],
    "deformation_potential_route": DEFORMATION_POTENTIAL_ROUTE,
    "dp_threshold": DP_THRESHOLD,
}
AUDIT_JSON.write_text(json.dumps(audit_record, indent=2), encoding="utf-8")
log(f"[cell1] Audit written to {AUDIT_JSON}")

# --- 6. Sanity check on tree shape -------------------------------------------
_materials = C2DB_TREE / "materials"
if _materials.is_dir():
    with os.scandir(_materials) as it:
        _strats = sum(1 for e in it if e.is_dir())
    log(f"[cell1] {C2DB_TREE}/materials/ contains {_strats:,} stoichiometry "
        f"directories. Step 2 walks this path.")
else:
    log(f"[cell1] WARNING: expected {_materials} to exist. Inspect the tree "
        f"layout before running Step 2, which assumes "
        f"materials/<stoich>/<uid>/<magstate>/.")

log("[cell1] DONE\n")

# Exported: C2DB_TREE, DEFORMATION_POTENTIAL_ROUTE, audit_record, AUDIT_JSON
