# Black-Box Optimization (BBO) Challenge — Stage 2 Capstone

A structured project for the Black-Box Optimization challenge (Stage 2), based on the NeurIPS 2020 BBO competition format. The goal is to explore unknown functions and identify their maxima through a documented, iterative strategy.

Author: Nikolas Karefyllidis, PhD

## Context and objectives

- Stage 2 focus: Work only with the shared challenge dataset provided for this stage (not Stage 1 data).
- Goal: For each of 8 black-box functions, find the input vector \(x\) that maximizes the output value \(y\).
- Approach: Prioritize exploration, reflection, and iterative strategy over seeking a single perfect answer immediately.
- Success: Evaluated on your process—a documented, evidence-based approach to selecting the next points. A documented failure (e.g. switching from UCB to PI after getting stuck) is valuable; a lucky guess is not.

## Key deliverables

1. A fully documented algorithm and model.
2. A portfolio-ready artifact suitable for strengthening your CV and career profile.

Submission materials live in `submission-template/` (data sheet, model card, README).

## Challenge overview

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
| 3 | 3D | Drug discovery | Minimize side effects; \(y\) is negative of side effects. |
| 4 | 4D | Warehouse logistics | Many local optima; output vs expensive baseline. |
| 5 | 4D | Chemical process yield | Typically unimodal; single peak. |
| 6 | 5D | Recipe optimization | Combined score (flavour, consistency, calories, waste, cost); bad factors negative. |
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
│       └── load_challenge_data.py # load_function_data(N) — read-only
│
├── data/
│   ├── problems/                  # Local appended data (function_1/inputs.npy, outputs.npy)
│   ├── submissions/               # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
│   └── results/                   # Exported plots
│
├── notebooks/
│   └── function_1_explore.ipynb   # Load, plot, suggest next x, save for submission
│
├── configs/
│   └── problems/
│       └── function_1.yaml        # 2D Radiation Detection (dim, bounds)
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

Further planned components (GP surrogate, extra notebooks, etc.) are in `docs/project_roadmap.md`.

### Write safety (avoid overwriting)

- **Never written to (read-only):** `initial_data/` — challenge data. The loader and notebooks only read from here; no code in this repo writes to `initial_data/`.
- **Written only when you enable a flag** (in `notebooks/function_1_explore.ipynb`):
  - `data/results/function_1_observations_and_distance_contour.png` and `function_1_3d_surface_distance_colour.png` — only if `IF_EXPORT_PLOT = True`
  - `data/submissions/function_1/next_input.npy` and `next_input_portal.txt` — only if `IF_EXPORT_QUERIES = True`
  - `data/problems/function_1/inputs.npy` and `outputs.npy` — only if `IF_APPEND_DATA = True` (append cell). This is your local copy (initial + appended points).
- **Default:** All flags are `False`; running the notebook then writes no files. Turn only the flags you need to `True` for that run.

## Allowed techniques

- Random Search (including non-uniform distributions).
- Grid Search (limited by dimensionality).
- **Bayesian Optimization** (GP surrogate + acquisition). This repo includes acquisition functions in `src/optimizers/bayesian/acquisition_functions.py`: UCB, Expected Improvement (EI), Probability of Improvement (PI), Thompson Sampling, Entropy Search (simplified proxy). References are in the module docstring.
- Manual reasoning (e.g. plotting and guessing in 2D).
- Custom surrogates (e.g. Random Forests, Gradient Boosted Trees instead of GPs).

You are not required to build a submission optimizer from scratch or to find the global maximum for every function; you are required to document and justify your process.

## Getting started

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Place raw challenge data in `initial_data/` (one folder per function with `initial_inputs.npy` and `initial_outputs.npy`). Do not edit the raw files.

3. Starting with function 1: open `notebooks/function_1_explore.ipynb`, run all cells to load the 10 initial points (via `load_function_data(1)`), visualize them, suggest a next \(x\), and save it to `data/submissions/function_1/` (including `next_input_portal.txt` for copy-paste to the portal). After you receive the new \(y\), use the append cell to add it to your working dataset and re-run for the next round.

4. Acquisition functions (UCB, EI, PI, etc.) live in `src/optimizers/bayesian/acquisition_functions.py`; import via `from src.optimizers.bayesian import upper_confidence_bound, expected_improvement, ...`. For more structure and planned components, see `docs/project_roadmap.md`. Complete the submission using the templates in `submission-template/`.

## References

- NeurIPS 2020 BBO Challenge (Huawei: GPs + heteroscedasticity/non-stationarity; Nvidia: ensembles; JetBrains: GP + SVM + nearest neighbour).
- Sample repos: [Bayesian Optimisation (soham96)](https://github.com/soham96/hyperparameter_optimisation), [Bayesian Optimization with XGBoost (solegalli)](https://github.com/solegalli/hyperparameter-optimization), [Bayesian Hyperparameter Optimization of GBM (WillKoehrsen)](https://github.com/WillKoehrsen/hyperparameter-optimization).
- Capstone Project FAQs: `docs/Capstone_Project_FAQs.md`.
