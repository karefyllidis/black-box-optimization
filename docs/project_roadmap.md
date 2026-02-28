# Project roadmap: planned structure

Components listed here are in use or planned. Add folders back when you need them.

## Current project structure (simplified)

```
black-box-optimization/
├── initial_data/                 # Raw challenge data (DO NOT MODIFY)
│   ├── function_1/ … function_8/
│
├── src/
│   ├── optimizers/
│   │   └── bayesian/             # acquisition_functions.py (UCB, EI, PI, Thompson, Entropy Search)
│   └── utils/
│       ├── load_challenge_data.py # load_function_data(N), assert_not_under_initial_data (blocks writes under initial_data only)
│       ├── plot_utilities.py     # style_axis, add_colorbar, style_legend; DEFAULT_FONT_SIZE_*, DEFAULT_EXPORT_*
│       └── sampling_utils.py    # sample_candidates() wrapper (F1 uses this; F2/F3+ use skopt.sampler directly)
│
├── data/
│   ├── problems/                 # Local appended data: only observations.csv per function (no .npy under data/)
│   ├── submissions/              # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
│   (data/results/)               # Exported plots (observations+contour, 3D surface, GP kernels, all acquisition points)
│
├── notebooks/
│   ├── function_1_Radiation-Detection.ipynb  # F1 (2D): full options — 3 kernels, all acquisitions, baselines
│   ├── function_2_Mystery-ML-Model.ipynb     # F2 (2D): d=2 reference — 3 kernels, ensemble, configurable bounds
│   ├── function_3_Drug-Discovery.ipynb       # F3 (3D): d≥3 reference — 2D pairwise, GP slices, ensemble
│   ├── function_4 … function_8              # Adapt from F2 (d=2) or F3 (d≥3) template
│
├── run_all.py                   # Submission summary (portal strings); --execute-notebooks runs all 8 notebooks
├── scripts/                     # Optional; if present, run_all.py runs *.py here before summary
├── configs/
│   └── problems/                 # (removed for now; see docs_private/private_notes.md)
│
├── tests/
│   ├── test_optimizers/
│   └── test_utils/
│
├── docs/
│   ├── project_roadmap.md        # (this file)
│   ├── Capstone_Project_FAQs.md
│   └── …
│
├── docs_private/                 # Private notes (contents gitignored except below)
│   ├── notebooks/
│   │   └── function_0_devel.ipynb   # 1D tutorial (tracked): GP kernels, skopt acquisition, ensemble EI+PI+UCB, true max
│   ├── phase_a_training/            # Stage 1 (archived; no longer relevant)
│   ├── ENSEMBLE_ACQUISITION_GUIDE.md
│   ├── TODO.md
│   └── ...                        # Rest gitignored via docs_private/*
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Removed for now (add back when needed):**
- `configs/algorithms/`, `configs/experiments/` — algorithm/experiment configs
- Scripts in `scripts/` — run_all.py runs any `scripts/*.py`; folder may be empty
- `tests/test_objectives/` — we have no src/objective
- `notebooks/weekly_review/` — weekly notes
- `src/objective/`, `src/experiments/` — see private notes (e.g. in docs_private/)

## Notebook workflow (F2/F3 template — in use)

1. **Setup and load data** — Imports (GP, skopt acquisition/sampler), repo root, load from local CSV or `initial_data`, flags.
2. **Parameters** — Kernel choice (`GP_KERNEL = "auto"` or manual), `OPTIMIZE_KERNEL`, kernel bounds (constant scale, length scale, white noise), acquisition coefficients (`XI_EI_PI`, `KAPPA_UCB`), candidate sampling, ensemble vs solo mode.
3. **Visualize** — Observations scatter, IDW contour, convergence plot. d=2: 2D contour + 3D surface. d≥3: 2D pairwise projections + IDW.
4. **GP surrogate** — Fit 3 kernels (RBF, Matérn, RBF+WhiteKernel) with configurable bounds; select best by LML. 3×2 grid (mean + std). d≥3: 2D slices at median.
5. **Acquisition** — EI/PI/UCB computed for all kernels via `skopt.acquisition`; ensemble logic (agree → EI argmax, disagree → centroid) or solo. Baselines: exploit + explore (no high-distance in F2/F3+).
6. **Select & illustrate** — Final plot: d=2: 1×2 (mean + std); d≥3: 3×2 GP slices with acquisition markers.
7. **Export** — Append new observation (§6) and/or save next_x (§7).

**F1** retains the original full-options layout (all acquisition functions, high-distance baseline, Thompson/Entropy).

For step-by-step adaptation checklists, see `docs_private/FUNCTION_NOTEBOOK_ADAPTATION_GUIDE.md`.

**run_all.py** — Run from project root. Prints full portal strings for functions 1–8 and file paths. Use `--execute-notebooks` to run all 8 notebooks (generates submissions); `--skip-scripts` to skip running `scripts/*.py`.

Write safety: `assert_not_under_initial_data(path, project_root)` only forbids writes under `project_root/initial_data/`; `data/results/`, `data/submissions/`, `data/problems/` are allowed.

## Planned components (add as you go)

### `src/optimizers/bayesian/`
- acquisition_functions.py (in use): UCB, EI, PI, Thompson Sampling, Entropy Search. Alternative to skopt; notebooks F1–F3 and function_0_devel use **skopt** (gaussian_ei, gaussian_pi, gaussian_lcb) for acquisition. EI remains the default next-query criterion.
- Add: GP surrogate, base_optimizer.py when you run BO in code.

### `src/utils/`
- load_challenge_data.py (in use). plot_utilities.py (in use): style_axis, add_colorbar, style_legend, DEFAULT_FONT_SIZE_*, DEFAULT_EXPORT_*.
- sampling_utils.py (in use by F1): `sample_candidates()` wrapper. F2/F3+ use `skopt.sampler.Sobol` / `Lhs` directly for space-filling candidate pools.
- Add: logging.py, visualization.py, metrics.py as needed.

### `configs/problems/`
- Removed for now (no code loaded it). Add problem YAMLs + loader later if we want a single source for dim, bounds, maximize; see docs_private/private_notes.md.

### `tests/`
- test_optimizers/, test_utils/: add tests when you add code.

### `docs/` and `docs_private/`
- project_roadmap.md, Capstone_Project_FAQs.md. Add learning_log.md, algorithms_summary.md as needed.
- docs_private/: ENSEMBLE_ACQUISITION_GUIDE.md, FUNCTION_NOTEBOOK_ADAPTATION_GUIDE.md, TODO.md. function_0_devel.ipynb is tracked (gitignore exception).
