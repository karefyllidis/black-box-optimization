# Black-Box Optimization (BBO) Challenge — Stage 2 Capstone

A structured project for the Black-Box Optimization challenge (Stage 2), based on the NeurIPS 2020 BBO competition format. The goal is to explore unknown functions and identify their maxima through a documented, iterative strategy.

**Author:** Nikolas Karefyllidis, PhD

---

## Section 1: Project overview

### What is the BBO capstone and its purpose?

The BBO (Black-Box Optimization) capstone simulates a setting where we must optimize an **unknown** objective function using only a limited number of expensive evaluations. We are not given the formula or a full picture of the landscape—only initial \((x, y)\) pairs and, after each submission, the single new \(y\) returned by the system for the \(x\) we chose. The purpose is to design and document a **strategy** for choosing the next query at each round, balancing exploration of uncertain regions with exploitation of promising ones.

### Overall goal and relevance

- **Goal:** For each of **8 black-box functions**, find an input vector \(x\) that **maximizes** the output value \(y\). Success is judged on the quality and justification of the process (e.g. choice of surrogate, acquisition, and exploration–exploitation trade-off), not only on hitting a global maximum.
- **Relevance:** In real-world ML and operations, we often face expensive black-box objectives—hyperparameter tuning, A/B tests, drug discovery, process control—where each evaluation is costly or slow. BBO teaches how to make each query count and how to communicate and defend technical choices.

### How this supports my career

This project demonstrates the ability to structure an optimization pipeline, implement and compare acquisition functions, and document reasoning in a way that is reusable and portfolio-ready. It supports my current and future work by showing evidence-based decision-making, clear technical communication, and familiarity with Bayesian optimization and related tools (GPs, acquisition functions, space-filling sampling).

---

## Section 2: Inputs and outputs

### What the model receives

- **Initial data (per function):** A set of \((x, y)\) pairs in NumPy format:
  - `initial_inputs.npy`: shape `(n, d)` — \(n\) input points (e.g. 10–40 depending on function), each of dimension \(d\) (2–8).
  - `initial_outputs.npy`: shape `(n,)` — the corresponding objective values.
- **Domain:** All inputs lie in \([0, 1]^d\) (or effectively \([0, 0.999999]\) for portal submission).
- **Query format for submission:** A single vector \(x\) per function, submitted as a string with **exactly six decimal places**, hyphen-separated, no spaces, e.g.  
  `0.498317-0.625531` (2D) or `0.123456-0.234567-0.345678` (3D).

### What the model returns (and what the portal returns)

- **We submit:** One input \(x\) per function (8 strings per week).
- **Portal returns:** The **same** inputs we submitted and the **corresponding** \(y\) value for each. We then append these \((x, y)\) pairs to our local dataset for the next round.
- **Output (objective):** A single real number per query. For all functions, **higher is better** (maximization). For F3 and F6, \(y\) can be negative; in that case e.g. \(-1\) is better than \(-2\).

### Example

| Function | Dimensions | Example input (portal format)     | Example output |
|----------|------------|------------------------------------|----------------|
| 1        | 2          | `0.002223-0.994219`               | 0              |
| 3        | 3          | `0.999764-0.052131-0.975598`      | -0.412…        |
| 8        | 8          | `0.050323-0.062907-…-0.836079`    | 9.898…         |

---

## Section 3: Challenge objectives

### Maximization

- **All 8 functions are maximization problems.** We always prefer a **higher** numerical value of \(y\). If outputs are negative (e.g. F3, F6), \(-1\) is better than \(-2\). Real-world analogies (e.g. “minimize side effects”) are already encoded in the transformed objective we receive.

### Constraints and limitations

- **Limited queries:** One submission per function per week; we must make each query count.
- **Unknown structure:** We do not see the equation or full surface; we only have initial data and the feedback from our own submissions.
- **Response delay:** Results arrive after submission; we plan the next round using the updated dataset (initial + all previous feedback).
- **Dimensions:** Functions 1–2 are 2D; 3 is 3D; 4–5 are 4D; 6 is 5D; 7 is 6D; 8 is 8D. Visualization and candidate sampling scale with dimension (e.g. pairwise plots, space-filling designs).

---

## Section 4: Technical approach (living record)

*This section is updated as the approach evolves across submission rounds.*

### Methods and modelling

- **Surrogate:** Gaussian Process (GP) regression (e.g. `sklearn.gaussian_process.GaussianProcessRegressor`) with RBF and/or Matérn kernels. Function 1 notebook also compares RBF, Matérn, and RBF+WhiteKernel.
- **Acquisition functions** (in `src/optimizers/bayesian/acquisition_functions.py`): Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence Bound (UCB), Thompson Sampling, and a simplified Entropy Search proxy. All are used in a **maximize-acquisition** step over a candidate set.
- **Candidate set:** Instead of optimizing acquisition on a tiny random sample, candidates are generated for **uniform or space-filling coverage** in \([0,1]^d\) via `src/utils/sampling_utils.sample_candidates(n, dim, method='sobol'|'lhs'|'grid'|'random')` (Sobol, LHS, grid, or random). This improves exploration and avoids missing good regions.
- **Baselines:** “Exploit” (current best point), “Explore” (random candidate), and **“High distance”** (point farthest from existing observations on the same candidate grid)—useful for sparse, peak-like functions (e.g. Function 1).

### Exploration vs exploitation

- **EI / PI:** Favour points that are likely to improve over the current best; more exploitation when the GP is confident.
- **UCB:** Explicit trade-off via \(\kappa\) (e.g. \(\mu + \kappa\sigma\)); higher \(\kappa\) encourages exploration.
- **Thompson Sampling:** Random draw from the GP posterior; natural exploration through randomness.
- **High-distance baseline:** Purely exploratory; maximises distance to nearest observation to cover under-sampled regions. Used as default for Function 1 in early rounds when the peak location is still uncertain.
- Strategy is **per-function**: e.g. more exploration for noisy or multi-peak functions (2, 4), more exploitation once a clear basin is found; high-distance or EI for sparse 2D (1).

### What makes the approach thoughtful

- **Documented choices:** Kernel choice, acquisition choice, and default “next point” (e.g. high-distance vs EI best) are stated and justified in the notebook.
- **Reproducibility:** Fixed seeds, explicit candidate method (e.g. Sobol), and flags for export/append so that runs are repeatable.
- **Living record:** Section 4 and the notebooks are updated after each round (e.g. “Week 2: switched default to EI for F1 after feedback”; “F5: UCB with lower \(\kappa\) to exploit”).

---

## Key deliverables

1. A fully documented algorithm and model.
2. A portfolio-ready artifact suitable for strengthening your CV and career profile.

Submission materials live in `submission-template/` (data sheet, model card, README).

---

## Challenge overview (reference)

- 8 black-box functions: You do not see their equations or full visualizations.
- Simulation: Each function represents a high-stakes task (e.g. tuning a radiation detector, controlling a robot) where data is expensive or slow to obtain.
- Constraint: A limited number of queries per week; strategy matters.
- Warm start: 10 \((x, y)\) pairs per function are provided as initial data.
- Data format: NumPy `.npy` files. Each function has a folder `initial_data/function_N/` with `initial_inputs.npy` (shape n×d) and `initial_outputs.npy` (shape n). Load via `src.utils.load_challenge_data.load_function_data(N)` (read-only).

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

1. Review: Download and analyze your current dataset (all previous queries and results).
2. Choose: Run your optimizer (or manual process) to select the single best next input \(x\) for each of the 8 functions.
3. Submit: Upload these \(x\) values to the capstone project portal.
4. Receive: The system returns the corresponding \(y\) values.
5. Reflect and update: Add the new points to your set and reflect—did the strategy work? Too explorative or too exploitative? How to adjust acquisition or model for the next round?

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
│   ├── funtion_2_Mystery-ML-Model.ipynb       # Function 2 (2D): simplified (RBF, EI+PI+UCB)
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

**Notebooks:** One notebook per function (1–8). **Function 1** has the most options (three GP kernels, all acquisition functions, baselines); use it as reference if you need more than RBF+EI/PI/UCB. Function 2 is simplified (RBF, EI+PI+UCB). Functions 3–8 use the same Bayesian optimisation workflow with dimension-specific pairwise plots and GP slices (3D→3 pairs, 4D→6, 5D→10, 6D→15, 8D→28).

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
- **Bayesian Optimization** (GP surrogate + acquisition). This repo includes acquisition functions in `src/optimizers/bayesian/acquisition_functions.py`: UCB, Expected Improvement (EI), Probability of Improvement (PI), Thompson Sampling, Entropy Search (simplified proxy). References are in the module docstring. For maximising acquisition over the input space, candidate points can be generated with **uniform coverage** via `src/utils/sampling_utils.py`: `sample_candidates(n, dim, method='grid'|'lhs'|'sobol'|'random')` — e.g. `'grid'` for a regular lattice, `'lhs'` or `'sobol'` for space-filling.
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
   - **5. Select next query** — Default: `next_x = next_x_high_dist` (high distance). Alternatives: `next_x_explore`, `x_best_EI_RBF`, `next_x_exploit`.  
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

## References

- NeurIPS 2020 BBO Challenge (Huawei: GPs + heteroscedasticity/non-stationarity; Nvidia: ensembles; JetBrains: GP + SVM + nearest neighbour).
- Sample repos: [Bayesian Optimisation (soham96)](https://github.com/soham96/hyperparameter_optimisation), [Bayesian Optimization with XGBoost (solegalli)](https://github.com/solegalli/hyperparameter-optimization), [Bayesian Hyperparameter Optimization of GBM (WillKoehrsen)](https://github.com/WillKoehrsen/hyperparameter-optimization).
- Capstone Project FAQs: `docs/Capstone_Project_FAQs.md`.
