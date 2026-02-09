# Project roadmap: planned structure

Components listed here are in use or planned. Add folders back when you need them.

## Current project structure (simplified)

```
black-box-optimization/
├── initial_data/                 # Raw challenge data (DO NOT MODIFY)
│   ├── function_1/ … function_8/
│
├── phase_a_training/             # Stage 1 (reference only)
│
├── src/
│   ├── optimizers/
│   │   └── bayesian/             # acquisition_functions.py (UCB, EI, PI, Thompson, Entropy Search)
│   └── utils/
│       ├── load_challenge_data.py # load_function_data(N), assert_not_under_initial_data (blocks writes under initial_data only)
│       └── plot_utilities.py     # style_axis, add_colorbar, style_legend; DEFAULT_FONT_SIZE_*, DEFAULT_EXPORT_*
│
├── data/
│   ├── problems/                 # Local appended data: only observations.csv per function (no .npy under data/)
│   ├── submissions/              # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
│   (data/results/)               # Exported plots (observations+contour, 3D surface, GP kernels, all acquisition points)
│
├── notebooks/
│   ├── function_1_Radiation-Detection.ipynb … function_8_High-dimensional-ML-Model.ipynb  # One per function (1–8); same workflow, dimension-specific pairwise plots
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
├── docs_private/                 # Private notes (gitignored); you moved notes here
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

## Function 1 notebook workflow (in use)

1. **Setup and load data** — Imports (GP, acquisition_functions, plot_utilities), repo root, load from local or `initial_data`, flags (IF_EXPORT_PLOT, IF_EXPORT_QUERIES, IF_APPEND_DATA).
2. **Visualize** — Grid, `min_dist` (distance to nearest observation), IDW y; 2D scatter + contour, 3D surface.
3. **Suggest next point (Bayesian)** — GP (RBF, Matérn ν=1.5, RBF+WhiteKernel); acquisition (EI, UCB, PI, Thompson, Entropy) × RBF/Matérn; sanity checks (low σ, (0,0) suggestions); baseline: exploit, explore, **high distance** (argmax of `min_dist`).
4. **Illustrate** — One contour of `min_dist` with all acquisition suggestions + Naive exploit, Random explore, High distance.
5. **Select next query** — Default: `next_x = next_x_high_dist`. Alternatives: `next_x_explore`, `x_best_EI_RBF`, `next_x_exploit`.
6. **Append new feedback** — After portal returns (x,y), append to `data/problems/function_1/` when IF_APPEND_DATA=True.
7. **Save suggestion** — When IF_EXPORT_QUERIES=True, write `next_x` to `data/submissions/function_1/` (npy + portal-format txt).

**run_all.py** — Run from project root. Prints full portal strings for functions 1–8 and file paths. Use `--execute-notebooks` to run all 8 notebooks (generates submissions); `--skip-scripts` to skip running `scripts/*.py`.

Write safety: `assert_not_under_initial_data(path, project_root)` only forbids writes under `project_root/initial_data/`; `data/results/`, `data/submissions/`, `data/problems/` are allowed.

## Planned components (add as you go)

### `src/optimizers/bayesian/`
- acquisition_functions.py (in use): UCB, EI, PI, Thompson Sampling, Entropy Search.
- Add: GP surrogate, base_optimizer.py when you run BO in code.

### `src/utils/`
- load_challenge_data.py (in use). plot_utilities.py (in use): style_axis, add_colorbar, style_legend, DEFAULT_FONT_SIZE_*, DEFAULT_EXPORT_*.
- sampling_utils.py (in use): sample_candidates(n, dim, method='random'|'lhs'|'sobol'|'grid') for uniform/space-filling candidate pools in [0,1]^d (LHS, Sobol, or regular lattice/grid); use in notebooks for better coverage than plain random.
- Add: logging.py, visualization.py, metrics.py as needed.

### `configs/problems/`
- Removed for now (no code loaded it). Add problem YAMLs + loader later if we want a single source for dim, bounds, maximize; see docs_private/private_notes.md.

### `tests/`
- test_optimizers/, test_utils/: add tests when you add code.

### `docs/`
- project_roadmap.md, Capstone_Project_FAQs.md. Add learning_log.md, algorithms_summary.md, etc. as needed.
