# Rank stability of two-dimensional transistor channel screening under first-principles uncertainty

Analysis code and derived data for the manuscript

Computational screening of 2D channel materials is normally reported as a ranked list of
single-valued figures of merit. This repository tests whether that **ordering** survives the
documented uncertainty in the first-principles inputs it is built from. Calibrated band-gap,
effective-mass and permittivity uncertainty are propagated through a ballistic ambipolar
double-gate device model for 754 monolayers, and the resulting rankings are assessed with
pairwise dominance probabilities, Kendall's tau, normalised rank entropy and total-order
Sobol indices.

---

## Reproduce the paper in about five minutes

Every derived table is committed here. You do **not** need the raw C2DB database, a GPU, or
any network access beyond a one-off 8 MB JARVIS download.

```bash
git clone https://github.com/<your-username>/tjeecc-2d-rank-stability.git
cd tjeecc-2d-rank-stability

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py --from 9          # rank stability, sensitivity, all figures
```

Outputs land in `figures/` and `data/processed/`. To regenerate only the figures:

```bash
python run_all.py --figures
```

To see what every stage does, what it needs, and in what order:

```bash
python run_all.py --list
```

---

## What is and is not in this repository

| Path | Contents | Committed? |
|---|---|---|
| `src/` | The 19 pipeline stages, in order | yes |
| `data/processed/` | 31 derived tables, posteriors and matrices (9.3 MB) | yes |
| `figures/` | The six manuscript figures as PDF | yes |
| `logs/` | Full log of the run that produced the published numbers | yes |
| `notebooks/` | The original exploratory notebook | yes |
| `data/raw/` | C2DB snapshot (520 MB) | **no** — provided by CAMD on request only |
| `cache/jarvis/` | JARVIS-DFT 2D dump (25 MB) | **no** — auto-downloaded |

### Why the raw data is not here

C2DB is produced by CAMD at the Technical University of Denmark, and its distribution page
states plainly:

> "The full dataset is provided upon request."
> — <https://www.2dhub.org/c2db/c2db.html>

The dataset is therefore released by its authors, to requesters, on their terms. If you need the raw database, request
it from CAMD. Instead:

```bash
python download_data.py            # how to obtain it, and from whom
python download_data.py --verify   # confirm you have the same snapshot as the paper
```

The exact snapshot used is pinned by SHA-256 in `download_data.py`:

```
c2db.db       74,481,664 bytes  sha256 a1a96ebc00c27daf…66aef2d
c2db.tar.gz  449,561,496 bytes  sha256 e3b27bf0426ace94…f35f9dc5
```

If DTU publishes a newer snapshot the checksums will not match. That is expected, not a bug —
but the numbers will then differ from the paper, and the committed tables in `data/processed/`
remain the record of what was actually used.

---

## The pipeline

| # | Stage | Needs | What it does |
|---|---|---|---|
| 1 | `cell01_extract_tarball` | raw | Stages and extracts the C2DB tree on local scratch; audits 54,457 files |
| 2 | `cell02_ingest_and_harvest` | raw | Ingests `c2db.db` (16,905 rows), harvests per-material JSON, applies the screening waterfall |
| 3 | `cell03_diagnose_parabolicity` | — | Tests the DOS-mass / principal-mass identity; excludes warped bands (1,493 → 1,085) |
| 4 | `cell04_jarvis_match` | net | Matches 1,103 JARVIS-2D entries to the device-ready set (131 matches) |
| 5–6 | `cell05`, `cell06` | — | Derives the permittivity uncertainty scale; rejects the JARVIS mass column with reasons |
| 7 | `cell07_step4_hierarchical_gap` | mcmc | The latent-variable gap model. **Documented to fail** (R̂ = 1.42, 529 divergences) because PBE and HSE06 are collinear |
| 8 | `cell08_step4b_gap_calibration` | mcmc | The model actually used: direct calibration on 129 G₀W₀ anchors, with 10-fold CV coverage |
| 9 | `cell09_anchor_representativeness` | — | Shows the anchors are not a random sample; validates interpolation only |
| 10 | `cell10_step5_device_model` | — | Ballistic ambipolar double-gate model; 7 unit tests including the Natori prefactor |
| 11 | `cell11_census_and_step6_validation` | — | Per-input uncertainty census over 754 materials |
| 12 | `cell12_transport_regime_census` | — | Ballistic → diffusive sweep and the mass-sigma sweep |
| 13 | `cell13_step8_complete` | — | Near-tie test, dominance matrix, rank entropy, robust shortlist |
| 14–15 | `cell14`, `cell15` | — | Figures 2–5; Sobol split by regime |
| 16 | `cell16_step6_validation_real_refs` | — | Validation against published NEGF results |
| 17 | `cell17_diagnose_the_54` | — | Diagnostic on the gap-sensitive subset |
| 18 | `cell18_figure1_pipeline` | — | Figure 1, the pipeline schematic |

`raw` = needs the C2DB snapshot · `net` = one-off 8 MB download · `mcmc` = needs jax + numpyro

Stage 7 is deliberately retained even though it fails. The failure is the reason the paper
calibrates directly rather than treating PBE and HSE06 as independent measurements of a latent
gap, and a reviewer should be able to see it rather than take the claim on trust.

---

## Environment

The pipeline is pure CPU except for the optional GPU path in the gap calibration. It does not
require Google Colab, Google Drive, or any cloud service.

```bash
pip install -r requirements.txt        # pinned versions used for the paper
# or
conda env create -f environment.yml && conda activate tjeecc
```

`jax` is not pinned: install the build matching your hardware first
(`pip install "jax[cpu]"` is fine), and `numpyro` will use it. Only stages 7–8 need it.

Paths are resolved with no hard-coded locations:

| Variable | Default | Purpose |
|---|---|---|
| `TJEECC_ROOT` | the repository directory | project root for `data/`, `figures/`, `logs/` |
| `TJEECC_SCRATCH` | system temp dir | fast local disk for the extracted C2DB tree |
| `TJEECC_AUTOINSTALL` | unset | set to `1` to let `src/cell00_bootstrap.py` pip-install missing packages |

Put the extracted tree on a **local** disk. It contains ~86,000 small files; extracting it onto
a network filesystem (Drive FUSE, NFS, SMB) takes hours instead of minutes.

### Determinism

`SEED = 20260815` is fixed in `src/cell00_bootstrap.py` and threaded through NumPy, Python's
`random`, and the NumPyro sampler. `jax_enable_x64` is switched on: the Fermi–Dirac integrals
and the gap posterior both need float64, and in float32 the sampler produces divergences that
look like model misspecification but are pure precision artefacts.

---

## Key derived files

| File | What it is |
|---|---|
| `gap_posterior.npy` | (1085 × 1000) calibrated band-gap predictive draws |
| `gap_posterior_uids.csv` | Row order for the above |
| `c2db_device_ready.parquet` | The 1,085 screened, parabolic, non-magnetic monolayers |
| `device_parameters.parquet` | The 754 with a complete parameter set, plus central figures of merit |
| `fig4_dominance_matrix.csv` | 30 × 30 pairwise dominance probabilities |
| `step8_rank_entropy.csv` | Normalised rank entropy per material |
| `fig5_sobol_by_regime.csv` | Total-order Sobol indices, split above/below the ambipolar limit |
| `transport_regime_rank_stability.csv` | Kendall's tau and top-k retention vs ballisticity |
| `table10_robust_shortlist.csv` | Mean-ranked vs lower-confidence-bound shortlists |
| `uncertainty_scales.json` | The σ values used, and which were measured vs assumed |

`uncertainty_scales.json` is worth opening first. It records explicitly which uncertainty
scales were derived from data and which were assumed — the effective-mass dispersion
`σ_ln m = 0.20` is assumed, and it is the parameter the paper's conclusions are most
conditional on.

---

## Licence and citation

* Code: MIT — see `LICENSE`
* Derived data, figures, logs: CC BY 4.0 — see `LICENSE-DATA`
* Upstream C2DB and JARVIS-DFT carry their own terms and are not redistributed here

Citation metadata is in `CITATION.cff`. If you use this work, please cite the manuscript and
also the upstream databases:

> Haastrup S, Strange M, Pandey M, Deilmann T, Schmidt PS et al. The Computational 2D Materials
> Database. *2D Materials* 2018; 5 (4): 042002. doi:10.1088/2053-1583/aacfc1
>
> Gjerding MN, Taghizadeh A, Rasmussen A, Ali S, Bertoldo F et al. Recent progress of the
> Computational 2D Materials Database (C2DB). *2D Materials* 2021; 8 (4): 044002.
> doi:10.1088/2053-1583/ac1059
>
> Choudhary K, Garrity KF, Reid ACE, DeCost B, Biacchi AJ et al. The joint automated repository
> for various integrated simulations (JARVIS). *npj Computational Materials* 2020; 6: 173.
> doi:10.1038/s41524-020-00440-1

---

## Known limitations

Stated here as well as in the paper, because they bound what this code can support.

1. **The dominant uncertainty is assumed, not measured.** No second first-principles source
   reports band-curvature masses in a comparable convention, so `σ_ln m = 0.20` is a stated
   assumption. It is swept over 0.10 / 0.20 / 0.40, which moves Kendall's tau from 0.887 to
   0.823 to 0.698. Every rank-stability conclusion is conditional on it.
2. **Absolute currents are not predictive.** The device model is effective-mass,
   ideal-contact and ballistic. Against full-band DFT-NEGF with metal contacts it runs a
   factor of 3.49 high. The offset is multiplicative and does not affect the relative
   orderings this work is about.
3. **The parabolicity filter removes 27 % of the screened set.** Strongly warped bands are
   outside the scope.
4. **G₀W₀ anchors are biased towards thin monolayers.** The coverage result validates
   interpolation; extrapolation is supported only by a seven-material spot check.
5. **No synthesisability or toxicity screen.** The top-ranked candidates are exotic
   heavy-element compounds and should not be read as recommendations.

## PAPER UNDER REVIEW
