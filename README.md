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

- **Surrogate:** GP regression with three kernels — RBF, Matérn (ν=1.5), RBF+WhiteKernel. All three are fitted; the kernel with the highest log-marginal-likelihood is selected automatically (`GP_KERNEL = None` → LML auto-select), or can be forced manually. Kernel hyperparameter bounds (constant scale, length scale, white noise) are configurable per notebook; optimization can be toggled via `OPTIMIZE_KERNEL`. White noise bounds default to `(1e-12, 1e1)` for near-noiseless functions.
- **Acquisition:** Notebooks use **scikit-optimize (skopt)** `gaussian_ei`, `gaussian_pi`, `gaussian_lcb`. Acquisition is maximised over a **candidate set** generated via `skopt.sampler.Sobol` or `Lhs` (configurable with `CANDIDATE_SAMPLING_METHOD`) for space-filling coverage in [0,1]^d. Candidate counts (`n_cand`) are always powers of 2 for Sobol balance properties (e.g. `2**18`). **Ensemble acquisition** (EI+PI+UCB; agree → EI argmax, disagree → centroid) is available in all notebooks (F1–F8); see `docs_private/ENSEMBLE_ACQUISITION_GUIDE.md`. An alternative acquisition implementation lives in `src/optimizers/bayesian/acquisition_functions.py` (EI, PI, UCB, Thompson Sampling, Entropy Search).
- **Baselines:** “Exploit” (perturb current best) and “Explore” (random candidate). Default acquisition configurable via `SOLO_STRATEGY` (EI, PI, or UCB). (F1 retains a “High distance” baseline; F2–F8 use a proximity warning instead.)
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
| 3 | 3D | Drug discovery | Maximize raw \(y\) (can be negative); 2D pairwise projections, GP slices at median. |
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
├── src/
│   ├── optimizers/
│   │   └── bayesian/              # acquisition_functions.py (UCB, EI, PI, Thompson Sampling, Entropy Search)
│   └── utils/
│       ├── load_challenge_data.py # load_function_data(N), assert_not_under_initial_data — read-only guard
│       ├── plot_utilities.py      # style_axis, add_colorbar, style_legend; DEFAULT_FONT_SIZE_*, export DPI/format
│       └── sampling_utils.py      # sample_candidates() wrapper (F1 uses this; F2/F3+ use skopt.sampler directly)
│
├── data/
│   ├── problems/                  # Appended data: only observations.csv (no .npy under data/; initial_data/ is read-only .npy)
│   └── submissions/               # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
├── data/results/                  # Exported plots (when IF_EXPORT_PLOT = True; see Write safety below)
│
├── notebooks/
│   ├── function_1_Radiation-Detection.ipynb      # F1 (2D): full options — 3 GP kernels, all acquisitions, baselines
│   ├── function_2_Mystery-ML-Model.ipynb         # F2 (2D): d=2 template — 3 kernels, ensemble, configurable bounds
│   ├── function_3_Drug-Discovery.ipynb           # F3 (3D): d≥3 template — pairwise projections, GP slices, ensemble
│   ├── function_4_Warehouse-Logistics.ipynb      # F4 (4D): 6 pairwise plots, GP slices, per-row colorbars
│   ├── function_5_Chemical-Process-Yield.ipynb   # F5 (4D): 6 pairwise plots, same workflow as F4
│   ├── function_6_Recipe-Optimization.ipynb      # F6 (5D): 10 pairwise plots, per-row colorbars
│   ├── function_7_Hyperparameter-Tuning.ipynb    # F7 (6D): 15 pairwise plots, per-row colorbars
│   └── function_8_High-dimensional-ML-Model.ipynb # F8 (8D): 28 pairwise plots, per-row colorbars
│
├── run_all.py                  # Print submission summary; optional: --execute-notebooks, --skip-scripts
├── configs/
│   └── problems/                  # (optional) problem configs; see docs_private/project_log.md
│
├── tests/
│   ├── test_optimizers/
│   └── test_utils/
│
├── docs/
│   ├── project_roadmap.md        # Current structure and planned components
│   └── Capstone_Project_FAQs.md
│
├── scripts/                     # append_week{1..4}_results.py — append portal feedback to observations.csv
│
├── docs_private/                 # Private notes (mostly gitignored)
│   ├── project_log.md            # Weekly evolution, assumptions, reflections
│   └── TODO.md
│
├── submission-template/          # Data sheet, model card, README for portfolio
├── requirements.txt
└── README.md
```

**Notebooks:** One notebook per function (1–8), all fully adapted and operational. **Function 2** is the canonical d=2 template; **Function 4** is the d≥3 template (extended from F3). All notebooks use three GP kernels (RBF, Matérn, RBF+WhiteKernel) with automatic best-kernel selection (LML), configurable kernel bounds, and ensemble/solo acquisition modes. **Function 1** retains the original full-options layout. F3–F8 use coarser visualisation grids (`n_grid_viz`) for fast plotting and finer Sobol candidate sets (`n_cand`, always a power of 2) for acquisition. d≥3 notebooks feature 2D pairwise projections with per-row colorbars and GP slices at median of held-out dimensions. **function_0_devel** (`docs_private/notebooks/`) is a 1D tutorial. See `docs_private/FUNCTION_NOTEBOOK_ADAPTATION_GUIDE.md` for the full adaptation guide.

Further details on planned components are in `docs/project_roadmap.md`.

## Allowed techniques

- Random Search (including non-uniform distributions).
- Grid Search (limited by dimensionality).
- **Bayesian Optimization** (GP surrogate + acquisition). Notebooks use **skopt** (`gaussian_ei`, `gaussian_pi`, `gaussian_lcb`) with candidates from `skopt.sampler.Sobol` or `Lhs`; alternative: `src/optimizers/bayesian/acquisition_functions.py` (EI, UCB, PI, Thompson Sampling, Entropy Search).
- Manual reasoning (e.g. plotting and guessing in 2D).
- Custom surrogates (e.g. Random Forests, Gradient Boosted Trees instead of GPs).

You are not required to build a submission optimizer from scratch or to find the global maximum for every function; you are required to document and justify your process.

## Getting started

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Place raw challenge data in `initial_data/` (one folder per function with `initial_inputs.npy` and `initial_outputs.npy`). Do not edit the raw files.

3. **Notebook workflow** — Each notebook follows the same structure:
   - **1. Setup and load data** — Imports, repo root, load from local CSV or `initial_data`, flags.
   - **2. Parameters** — Kernel choice (`GP_KERNEL = "auto"` or manual), `OPTIMIZE_KERNEL`, kernel bounds, acquisition coefficients (`XI_EI_PI`, `KAPPA_UCB`), candidate sampling, ensemble vs solo mode.
   - **3. Visualize** — Observations, distances, GP surrogate surfaces (2D contour for d=2; 2D pairwise slices for d≥3).
   - **4. Acquisition** — EI/PI/UCB computed for all three kernels; best kernel selected by LML. Ensemble logic picks the next query.
   - **5. Select next query** — Default: EI argmax from the best kernel. Alternatives: PI, UCB, exploit, explore.
   - **6. Append new feedback** — After portal returns \((x,y)\), run with `IF_APPEND_DATA = True`.
   - **7. Save suggestion** — With `IF_EXPORT_QUERIES = True`, write `next_x` to `data/submissions/function_N/`.

   **Templates:** **Function 2** is the d=2 template; **Function 4** is the d≥3 template (all F3–F8 are fully adapted). **Function 1** retains the original full-options layout. See `docs_private/FUNCTION_NOTEBOOK_ADAPTATION_GUIDE.md` for the full adaptation guide and checklists.

4. **Acquisition & utilities:** Notebooks use `skopt.acquisition` (`gaussian_ei`, `gaussian_pi`, `gaussian_lcb`) and `skopt.sampler` (Sobol/LHS). Alternative acquisition: `src/optimizers/bayesian/acquisition_functions.py`. Plot styling: `src/utils/plot_utilities.py`. **Tutorial:** `docs_private/notebooks/function_0_devel.ipynb`. Complete the submission using templates in `submission-template/`.

5. **Submission summary** — From the project root, run:
   ```bash
   python run_all.py
   ```
   By default this runs any scripts in `scripts/` (e.g. `append_week1_results.py` through `append_week4_results.py` to append portal feedback to `data/problems/function_N/observations.csv`), then prints a **submission summary**: full portal strings (copy-paste per function) and where files live. Options:
   - `python run_all.py --execute-notebooks` — run all 8 function notebooks (writes `data/submissions/function_N/`; needs `nbconvert`).
   - `python run_all.py --skip-scripts` — skip running any scripts in `scripts/` (if present); only show the summary.

## Documentation

| File | Purpose |
|------|---------|
| **README.md** (this file) | Project overview, inputs/outputs, technical approach, structure, getting started |
| **docs/project_roadmap.md** | Current structure, notebook workflow, planned components |
| **docs/Capstone_Project_FAQs.md** | Capstone FAQs: data, submission, method |
| **docs_private/FUNCTION_NOTEBOOK_ADAPTATION_GUIDE.md** | Complete adaptation guide: F2 (d=2) / F4 (d≥3) templates, checklists, dimension reference, styling patterns |
| **docs_private/ENSEMBLE_ACQUISITION_GUIDE.md** | Ensemble EI+PI+UCB: agree/disagree logic, skopt usage |
| **docs_private/project_log.md** | Weekly evolution, assumptions, results, reflections, peer ideas |
| **docs_private/TODO.md** | Near-term tasks and status |
| **docs_private/notebooks/function_0_devel.ipynb** | 1D tutorial (tracked); GP kernels, skopt, ensemble |

*Note:* `docs_private/` is mostly gitignored; `function_0_devel.ipynb` is an exception (tracked).

## References

- NeurIPS 2020 BBO Challenge (Huawei: GPs + heteroscedasticity/non-stationarity; Nvidia: ensembles; JetBrains: GP + SVM + nearest neighbour).
- Sample repos: [Bayesian Optimisation (soham96)](https://github.com/soham96/hyperparameter_optimisation), [Bayesian Optimization with XGBoost (solegalli)](https://github.com/solegalli/hyperparameter-optimization), [Bayesian Hyperparameter Optimization of GBM (WillKoehrsen)](https://github.com/WillKoehrsen/hyperparameter-optimization).
- Capstone Project FAQs: `docs/Capstone_Project_FAQs.md`.
