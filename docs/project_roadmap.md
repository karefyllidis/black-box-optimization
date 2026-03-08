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
│       ├── plot_utilities.py     # style_axis, add_colorbar, style_legend, prepare_surface_for_plot, style_axis_3d; plot_2d_bo_state, plot_2d_function, plot_convergence, plot_gp_1d, plot_acquisition_1d, plot_bo_iteration_1d, plot_parallel_coordinates; DEFAULT_FONT_SIZE_*, DEFAULT_EXPORT_*
│       ├── warping.py            # apply_output_warping(y, mode=None|"log"|"boxcox"); inverse_output_warping — HEBO-inspired y transform for GP
│       └── sampling_utils.py    # sample_candidates() wrapper (F1 uses this; F2/F3+ use skopt.sampler directly)
│
├── data/
│   ├── problems/                 # Local appended data: only observations.csv per function (no .npy under data/)
│   ├── submissions/              # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
│   (data/results/)               # Exported plots (observations+contour, 3D surface, GP kernels, all acquisition points)
│
├── notebooks/
│   ├── function_1_Radiation-Detection.ipynb      # F1 (2D): full options — 3 kernels, all acquisitions, baselines
│   ├── function_2_Mystery-ML-Model.ipynb         # F2 (2D): d=2 template — 3 kernels, ensemble, configurable bounds
│   ├── function_3_Drug-Discovery.ipynb           # F3 (3D): d≥3 baseline — pairwise projections, GP slices, ensemble
│   ├── function_4_Warehouse-Logistics.ipynb      # F4 (4D): 6 pairwise plots, per-row colorbars, coarser viz / finer acq
│   ├── function_5_Chemical-Process-Yield.ipynb   # F5 (4D): 6 pairwise plots, same workflow as F4
│   ├── function_6_Recipe-Optimization.ipynb      # F6 (5D): 10 pairwise plots, per-row colorbars
│   ├── function_7_Hyperparameter-Tuning.ipynb    # F7 (6D): 15 pairwise plots, per-row colorbars
│   └── function_8_High-dimensional-ML-Model.ipynb # F8 (8D): 28 pairwise plots, per-row colorbars
│
├── run_all.py                   # Submission summary (portal strings); --execute-notebooks runs all 8 notebooks
├── scripts/                     # append_week{1..5}_results.py — portal feedback → observations.csv
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
│   ├── TECHNICAL_FOUNDATIONS.md  # Justification, key papers, library choices
│   └── …
│
├── docs_private/                 # Private notes (gitignored; structure not listed in open repo)
├── requirements.txt
├── .gitignore
└── README.md
```

**Removed for now (add back when needed):**
- `configs/algorithms/`, `configs/experiments/` — algorithm/experiment configs
- Scripts in `scripts/` — run_all.py runs any `scripts/*.py` (e.g. append_week1..5_results.py); folder may be empty
- `tests/test_objectives/` — we have no src/objective
- `notebooks/weekly_review/` — weekly notes
- `src/objective/`, `src/experiments/` — see private notes (e.g. in docs_private/)

## Notebook workflow (F2/F4 template — all notebooks adapted)

1. **Setup and load data** — Imports (GP, skopt acquisition/sampler), repo root, load from local CSV or `initial_data`, flags.
2. **Parameters** — Kernel choice (`GP_KERNEL = None` → LML auto-select, or manual), `OPTIMIZE_KERNEL`, kernel bounds (constant scale, length scale, white noise `(1e-12, 1e1)`), acquisition coefficients (`XI_EI_PI`, `KAPPA_UCB`), candidate sampling (`n_cand` as power of 2), ensemble vs solo mode (`SOLO_STRATEGY`), `MIN_DIST_THRESHOLD` (min L2 distance from any observation; masks acquisition and drives proximity check/fallback), `BOUNDARY_MARGIN` (optional; mask candidates near edges [0,margin] or [1−margin,1]; 0.05 for low-d F1–F3, 0 for F4–F8).
3. **Visualize** — Observations scatter, IDW contour, convergence plot. d=2: 2D contour + 3D surface. d≥3: 2D pairwise projections + IDW with per-row colorbars; uses coarser `n_grid_viz` for fast rendering.
4. **GP surrogate** — Fit 3 kernels (RBF, Matérn, RBF+WhiteKernel) with configurable bounds; select best by LML. 3×2 grid (mean + std). d≥3: 2D slices at median of held-out dimensions.
5. **Acquisition** — EI/PI/UCB computed for all kernels via `skopt.acquisition` on `n_cand` Sobol/LHS candidates. This cell computes `next_x_high_dist` (farthest candidate from observations) and uses it as fallback when the acquisition argmax lies in the masked set. Candidates within `MIN_DIST_THRESHOLD` of any observation are masked (EI/PI → −∞, LCB → +∞); when `BOUNDARY_MARGIN` > 0, candidates with any coordinate in [0, margin] or [1−margin, 1] are also masked (low-d only; F4–F8 use 0). If `BOUNDARY_MARGIN` is undefined (e.g. parameters cell not run), the cell sets it to 0. Ensemble logic (agree → EI argmax, disagree → centroid) or solo. Baselines: exploit + explore (no high-distance in F2–F8).
6. **Select & illustrate** — Final plot: d=2: 1×2 (mean + std); d≥3: pairwise GP slices with acquisition markers; `tight_layout(rect=[0,0,1,0.96])` + `suptitle(..., y=0.98)` avoids title overlap.
7. **Export** — Append new observation (§6) and/or save next_x (§7).

**F1** retains the original full-options layout (all acquisition functions, high-distance baseline, Thompson/Entropy). All F3–F8 notebooks are fully adapted with dimension-specific pair counts, per-row colorbars, and optimised rendering.

For step-by-step adaptation checklists, see `docs_private/40_notes_and_references/function_notebook_adaptation_guide.md`.

**run_all.py** — Run from project root. Runs any `scripts/*.py` (e.g. append_week1..5_results.py), then prints full portal strings for functions 1–8 and file paths. Use `--execute-notebooks` to run all 8 notebooks (generates submissions); `--skip-scripts` to skip running scripts.

Write safety: `assert_not_under_initial_data(path, project_root)` only forbids writes under `project_root/initial_data/`; `data/results/`, `data/submissions/`, `data/problems/` are allowed.

## Planned components (add as you go)

### `src/optimizers/bayesian/`
- acquisition_functions.py (in use): UCB, EI, PI, Thompson Sampling, Entropy Search. Alternative to skopt; all notebooks (F1–F8) and function_0_devel use **skopt** (gaussian_ei, gaussian_pi, gaussian_lcb) for acquisition. Default next-query criterion configurable via `SOLO_STRATEGY`.
- Add: GP surrogate, base_optimizer.py when you run BO in code.

### `src/utils/`
- load_challenge_data.py (in use). plot_utilities.py (in use): style_axis, add_colorbar, style_legend, prepare_surface_for_plot, style_axis_3d; plot_2d_bo_state, plot_2d_function, plot_convergence, plot_gp_1d, plot_acquisition_1d, plot_bo_iteration_1d, plot_parallel_coordinates; DEFAULT_FONT_SIZE_*, DEFAULT_EXPORT_*.
- warping.py (in use): `apply_output_warping(y, mode=None|"log"|"boxcox")`, `inverse_output_warping` — HEBO-inspired output warping; all 8 notebooks use it when `OUTPUT_WARPING` is set; F1, F5, F7 default to `"log"`.
- sampling_utils.py (in use by F1): `sample_candidates()` wrapper. F2/F3+ use `skopt.sampler.Sobol` / `Lhs` directly for space-filling candidate pools.
- Add: logging.py, visualization.py, metrics.py as needed.

### `configs/problems/`
- Removed for now (no code loaded it). Add problem YAMLs + loader later if we want a single source for dim, bounds, maximize; see docs_private/private_notes.md.

### `tests/`
- test_optimizers/, test_utils/: add tests when you add code.

### `docs/` and `docs_private/`
- project_roadmap.md (this file), Capstone_Project_FAQs.md. Add learning_log.md, algorithms_summary.md as needed.
- docs_private/: private notes, project log, TODO, guides; one notebook is tracked (gitignore exception). Contents gitignored; structure not listed in open repo.
