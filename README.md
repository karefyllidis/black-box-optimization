# Black-Box Optimization (BBO) Challenge

A structured project for the Black-Box Optimization challenge, based on the NeurIPS 2020 BBO competition format. The goal is to explore unknown functions and identify their maxima through a documented, iterative strategy.

**Author:** Nikolas Karefyllidis, PhD

---

## Section 1: Project overview

### What is the BBO capstone?

We optimize **8 unknown** objective functions with a limited number of expensive evaluations: we get initial (x, y) pairs and, after each submission, the single y returned for the x we chose. The purpose is to design and document a **strategy** for the next query each round—balancing exploration (uncertain regions) and exploitation (promising ones).

### Goal, relevance, and career link

- **Goal:** Find input x that **maximizes** y for each function. Success is judged on process quality (surrogate, acquisition, trade-off), not only on finding the global maximum.
- **Relevance:** Real-world ML often has expensive black-box objectives (hyperparameter tuning, A/B tests, drug discovery); BBO teaches how to make each query count and defend technical choices.
- **Career:** This project demonstrates decision-making with limited data, clear technical communication, and a reusable optimization pipeline—aligning with goals to deliver evidence-based, portfolio-ready work.

### Background: Bayesian optimization

**Bayesian optimization (BO)** is a sample-efficient, sequential method for black-box, expensive objectives under a limited budget. No formula for f(x)—we only evaluate at chosen points. BO uses a **probabilistic surrogate** (here, a Gaussian Process) that gives a **predictive mean** μ(x) (exploitation) and **uncertainty** σ(x) (exploration); an **acquisition function** (e.g. EI, UCB) combines them to pick the next point. Loop: fit surrogate → maximize acquisition → evaluate f(x) → add (x, y) → repeat.

**GPs** define a distribution over functions and provide μ(x) and σ(x) after observing data; uncertainty is lower near observations and higher elsewhere, which acquisition functions use to balance exploration and exploitation. GPs are data-efficient and scale to moderate dimensions (d ≤ 20), matching our 2D–8D functions.

---

## Section 2: Inputs and outputs

### Inputs (format, dimensionality, constraints)

- **Initial data:** `initial_inputs.npy` (shape n×d), `initial_outputs.npy` (shape n). Dimension d = 2–8 depending on function (F1–F2: 2D; F3: 3D; F4–F5: 4D; F6: 5D; F7: 6D; F8: 8D).
- **Domain:** All inputs in [0, 1]^d (portal: [0, 0.999999]).
- **Submission format:** One vector per function, as a string with **six decimal places**, hyphen-separated, no spaces, e.g. `0.498317-0.625531` (2D), `0.123456-0.234567-0.345678` (3D).

### Outputs

- **Portal returns:** The submitted x and the corresponding **y** (one real number per query). We append (x, y) for the next round. **Higher y is better** (maximization); F3 and F6 can be negative (e.g. −1 better than −2).

### Example

| Function | Dimensions | Example input (portal format)     | Example output |
|----------|------------|------------------------------------|----------------|
| 1        | 2          | `0.002223-0.994219`               | 0              |
| 3        | 3          | `0.999764-0.052131-0.975598`      | -0.412…        |
| 8        | 8          | `0.050323-0.062907-…-0.836079`    | 9.898…         |

---

## Section 3: Challenge objectives

**Goal:** **Maximise** y for each of the 8 functions (higher is better; negative y allowed for F3, F6).

**Constraints:** (1) **Query limit** — one submission per function per week. (2) **Unknown structure** — no equation or full surface; only initial data and our own feedback, so we rely on a surrogate and acquisition. (3) **Response delay** — plan next round from updated dataset (initial + all feedback). (4) **Dimensions** — 2D to 8D; visualization and sampling scale with d.

---

## Section 4: Technical approach (living record)

*Updated as the approach evolves.*

### Methods

- **Surrogate:** GP regression (RBF/Matérn; Function 1 also compares RBF+WhiteKernel). GP gives mean and uncertainty; acquisition uses both to choose the next query.
- **Acquisition** (`src/optimizers/bayesian/acquisition_functions.py`): **EI** primary; also PI, UCB, Thompson Sampling, Entropy Search. Maximized over a **candidate set** from `sample_candidates(..., method='sobol'|'lhs'|'grid'|'random')` for space-filling coverage in [0,1]^d.
- **Baselines:** “Exploit” (current best point), “Explore” (random candidate), and **“High distance”** (point farthest from existing observations on the same candidate grid). Default: **EI**.
- **Other methods:** No **linear/logistic regression**—surface is nonlinear and multimodal. **SVMs** could classify high vs low regions (soft-margin or kernel); GP kept as main surrogate for uncertainty (needed for EI). Possible combo: SVM for regions, GP+EI for exact query.

### Rounds 1–3 (evolution)

- **Round 1:** Exploration-heavy; ~10 points, high uncertainty. EI/UCB over Sobol/LHS candidates. Default: EI (sometimes UCB with higher κ).
- **Round 2:** Same method; 11 points after Week 1. Posterior improved; suggestion changed without changing the algorithm. Exploration bonus unchanged (5D–8D still uncertain).
- **Round 3:** ~12–14 points. More trust in EI over fixed candidates; heuristics (Sobol/grid), default RBF, no GP tuning. EI single criterion; GP uncertainty drives exploration.

### Exploration–exploitation

**Trade-off:** Exploration (under-sampled/uncertain regions) vs exploitation (near known good values). **EI** balances them: high where μ(x) > current best (exploit) or σ(x) large (explore). No round-by-round κ tuning. Alternatives: UCB (μ + κσ), PI, Thompson Sampling, high-distance baseline; per-function and documented in notebooks.

**Thoughtful aspects:** Documented kernel/acquisition/default in each notebook; fixed seeds and flags for reproducibility; section and notebooks updated each round.

---

## Key deliverables

Fully documented algorithm and model; portfolio-ready artifact (CV/profile). Submission materials: `submission-template/` (data sheet, model card, README).

---

## Challenge overview (reference)

8 black-box functions (no equations or full surface); one query per function per week; warm start: initial (x, y) pairs in `initial_data/function_N/` (`initial_inputs.npy`, `initial_outputs.npy`). Load via `load_function_data(N)` (read-only).

### The 8 functions (brief)

| # | Dim | Analogy | Notes |
|---|-----|---------|--------|
| 1 | 2D | Radiation detection | Sparse signal; proximity yields non-zero reading. |
| 2 | 2D | Mystery ML model | Noisy; many local peaks; balance exploration vs exploitation. |
| 3 | 3D | Drug discovery | Maximize transformed output; \(y\) can be negative—higher is better (e.g. −1 > −2). |
| 4 | 4D | Warehouse logistics | Many local optima; output vs expensive baseline. |
| 5 | 4D | Chemical process yield | Typically unimodal; single peak. |
| 6 | 5D | Recipe optimization | Maximize transformed score; \(y\) can be negative—higher is better (e.g. −1 > −2). |
| 7 | 6D | Hyperparameter tuning | e.g. learning rate, regularization, hidden layers; maximize accuracy/F1. |
| 8 | 8D | High-dimensional ML model | Learning rate, batch size, layers, dropout, etc.; single validation accuracy in [0,1]. |

## Weekly workflow

Review dataset → run optimizer → submit 8 x values to portal → receive y → append (x, y) and reflect (strategy, exploration vs exploitation) → repeat.

## Project structure (in use)

```
black-box-optimization/
├── initial_data/                 # Raw challenge data (DO NOT MODIFY)
│   ├── function_1/ … function_8/   # initial_inputs.npy, initial_outputs.npy each
│
├── phase_a_training/             # Stage 1 (reference only)
│
├── src/
│   ├── optimizers/
│   │   └── bayesian/              # acquisition_functions.py (UCB, EI, PI, Thompson Sampling, Entropy Search)
│   └── utils/
│       ├── load_challenge_data.py # load_function_data(N), assert_not_under_initial_data — read-only guard
│       ├── plot_utilities.py      # style_axis, add_colorbar, style_legend; DEFAULT_FONT_SIZE_*, export DPI/format
│       └── sampling_utils.py      # sample_candidates(n, dim, method='random'|'lhs'|'sobol'|'grid') — uniform/space-filling candidates in [0,1]^d
│
├── data/
│   ├── problems/                  # Appended data: only observations.csv (no .npy under data/; initial_data/ is read-only .npy)
│   └── submissions/               # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
├── data/results/                  # Exported plots (when IF_EXPORT_PLOT = True; see Write safety below)
│
├── notebooks/
│   ├── function_1_Radiation-Detection.ipynb   # Function 1 (2D): full options — 3 GP kernels, all acquisitions; use as reference
│   ├── function_2_Mystery-ML-Model.ipynb       # Function 2 (2D): simplified (RBF, EI+PI+UCB)
│   ├── function_3_Drug-Discovery.ipynb        # Function 3 (3D): minimization→maximization, 2D pairwise + 3D scatter
│   ├── function_4_Warehouse-Logistics.ipynb  # Function 4 (4D): 6 pairwise 2D plots, GP slices, acquisition
│   ├── function_5_Chemical-Process-Yield.ipynb   # Function 5 (4D): same workflow as function_4
│   ├── function_6_Recipe-Optimization.ipynb      # Function 6 (5D): 10 pairwise plots
│   ├── function_7_Hyperparameter-Tuning.ipynb    # Function 7 (6D): 15 pairwise plots
│   └── function_8_High-dimensional-ML-Model.ipynb # Function 8 (8D): 28 pairwise plots
│
├── run_all.py                  # Print submission summary; optional: --execute-notebooks, --skip-scripts
├── configs/
│   └── problems/                  # (optional) problem configs; see docs_private/private_notes.md
│
├── tests/
│   ├── test_optimizers/
│   └── test_utils/
│
├── docs/
│   ├── project_roadmap.md        # Current structure and planned components
│   └── Capstone_Project_FAQs.md
│
├── submission-template/          # Data sheet, model card, README for portfolio
├── requirements.txt
└── README.md
```

**Notebooks:** One notebook per function (1–8). All notebooks use **F1: EI (Expected Improvement)** as the primary acquisition for the next query; section **5. Select next query** sets `next_x = x_best_EI_RBF` by default. **Function 1** has the most options (three GP kernels, all acquisition functions, baselines); use it as reference if you need more than RBF+EI. Functions 2–8 use the same Bayesian optimisation workflow with dimension-specific pairwise plots and GP slices (3D→3 pairs, 4D→6, 5D→10, 6D→15, 8D→28).

Further planned components (GP surrogate, extra notebooks, etc.) are in `docs/project_roadmap.md`.

### Write safety (avoid overwriting)

- **Never written to (read-only):** `initial_data/` — challenge data. The loader and notebooks only read from here. `assert_not_under_initial_data()` (in `src/utils/load_challenge_data.py`) blocks any write path under `initial_data/`; paths under `data/problems/`, `data/submissions/`, and `data/results/` are allowed.
- **Written only when you enable a flag** (in `notebooks/function_1_Radiation-Detection.ipynb`):
  - **Plots** (only if `IF_EXPORT_PLOT = True`): `data/results/function_1_observations_and_distance_contour.png`, `data/results/function_1_3d_surface_distance_colour.png`, etc. Directory and format/DPI come from `PLOT_EXPORT_DIR` (default `repo_root / "data" / "results"`), `DEFAULT_EXPORT_FORMAT`, `DEFAULT_EXPORT_DPI` (see `src/utils/plot_utilities.py`).
  - **Submissions** (only if `IF_EXPORT_QUERIES = True`): `data/submissions/function_1/next_input.npy`, `next_input_portal.txt` (portal format: 6 decimals, hyphens, no spaces).
  - **Appended data** (only if `IF_APPEND_DATA = True`): `data/problems/function_1/observations.csv` — your local copy (initial + appended points). Under `data/` we operate only with CSV; no `.npy` in `data/problems/`. Run the append cell after you receive new \((x,y)\) from the portal.
- **Default:** All flags are `False`; running the notebook then writes no files. Turn only the flags you need to `True` for that run.

## Allowed techniques

- Random Search (including non-uniform distributions).
- Grid Search (limited by dimensionality).
- **Bayesian Optimization** (GP surrogate + acquisition). This repo includes acquisition functions in `src/optimizers/bayesian/acquisition_functions.py`: **EI (primary, F1)**, UCB, PI, Thompson Sampling, Entropy Search (simplified proxy). References are in the module docstring. For maximising acquisition over the input space, candidate points can be generated with **uniform coverage** via `src/utils/sampling_utils.py`: `sample_candidates(n, dim, method='grid'|'lhs'|'sobol'|'random')` — e.g. `'grid'` for a regular lattice, `'lhs'` or `'sobol'` for space-filling.
- Manual reasoning (e.g. plotting and guessing in 2D).
- Custom surrogates (e.g. Random Forests, Gradient Boosted Trees instead of GPs).

You are not required to build a submission optimizer from scratch or to find the global maximum for every function; you are required to document and justify your process.

## Getting started

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Place raw challenge data in `initial_data/` (one folder per function with `initial_inputs.npy` and `initial_outputs.npy`). Do not edit the raw files.

3. **Function 1 notebook** (`notebooks/function_1_Radiation-Detection.ipynb`):  
   - **1. Setup and load data** — Imports, repo root, load from local or `initial_data`, flags.  
   - **2. Visualize** — Grid, distance to nearest observation, 2D contour + 3D surface.  
   - **3. Suggest next point (Bayesian)** — GP surrogates (RBF, Matérn, RBF+WhiteKernel); acquisition (EI, UCB, PI, Thompson, Entropy) with RBF/Matérn, maximised over **uniform-coverage candidates** (`sample_candidates(..., method='grid')` by default; set `CANDIDATE_SAMPLING_METHOD` to `'lhs'`, `'sobol'`, or `'random'` to compare); sanity checks for (0,0) and low σ; baseline (exploit, explore, **high distance** = point farthest from observations on the same candidate set).  
   - **4. Illustrate** — Single plot: all acquisition suggestions + Naive exploit, Random explore, High distance on distance contour.  
   - **5. Select next query** — Default: `next_x = x_best_EI_RBF` (**F1: EI**). Alternatives: `next_x_high_dist`, `x_best_UCB_RBF`, `next_x_exploit`, `next_x_explore`.  
   - **6. Append new feedback** — After portal returns \((x,y)\), run with `IF_APPEND_DATA = True` to append to `data/problems/function_1/`.  
   - **7. Save suggestion** — With `IF_EXPORT_QUERIES = True`, write `next_x` to `data/submissions/function_1/` (npy + portal-format txt).  
   After you receive the new \(y\), run section 6 (Append) then re-run the notebook for the next round.

4. Acquisition functions live in `src/optimizers/bayesian/acquisition_functions.py`; import via `from src.optimizers.bayesian.acquisition_functions import expected_improvement, upper_confidence_bound, ...`. Plot styling: `src/utils/plot_utilities.py` (`style_axis`, `add_colorbar`, `DEFAULT_FONT_SIZE_AXIS`, etc.). For more structure and planned components, see `docs/project_roadmap.md`. Complete the submission using the templates in `submission-template/`.

5. **Submission summary** — From the project root, run:
   ```bash
   python run_all.py
   ```
   This prints a **submission summary**: full portal strings (copy-paste per function) and where files live. Options:
   - `python run_all.py --execute-notebooks` — run all 8 function notebooks (writes `data/submissions/function_N/`; needs `nbconvert`).
   - `python run_all.py --skip-scripts` — skip running any scripts in `scripts/` (if present); only show the summary.

## Documentation

- **README.md** (this file): Project overview, inputs/outputs, technical approach, structure, getting started.
- **docs/project_roadmap.md**: Current project structure, function notebook workflow, planned components.
- **docs/Capstone_Project_FAQs.md**: Capstone FAQs (data, submission, method); includes a short note on this repo’s use of EI as F1.

## References

- NeurIPS 2020 BBO Challenge (Huawei: GPs + heteroscedasticity/non-stationarity; Nvidia: ensembles; JetBrains: GP + SVM + nearest neighbour).
- Sample repos: [Bayesian Optimisation (soham96)](https://github.com/soham96/hyperparameter_optimisation), [Bayesian Optimization with XGBoost (solegalli)](https://github.com/solegalli/hyperparameter-optimization), [Bayesian Hyperparameter Optimization of GBM (WillKoehrsen)](https://github.com/WillKoehrsen/hyperparameter-optimization).
- Capstone Project FAQs: `docs/Capstone_Project_FAQs.md`.
