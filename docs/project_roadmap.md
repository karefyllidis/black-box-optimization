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
│   │   └── bayesian/             # Acquisition functions (UCB, EI, PI, Thompson Sampling, Entropy Search)
│   └── utils/
│       └── load_challenge_data.py # (in use)
│
├── data/
│   ├── problems/                 # Local appended data (function_1/inputs.npy, outputs.npy)
│   ├── submissions/              # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
│   └── results/                  # Exported plots
│
├── notebooks/
│   └── function_1_explore.ipynb  # (in use)
│
├── configs/
│   └── problems/                 # function_1.yaml (in use)
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
- `scripts/` — run_experiment.py, benchmark_all.py
- `tests/test_objectives/` — we have no src/objective
- `notebooks/weekly_review/` — weekly notes
- `src/objective/`, `src/experiments/` — see private notes (e.g. in docs_private/)

## Planned components (add as you go)

### `src/optimizers/bayesian/`
- acquisition_functions.py (in use): UCB, EI, PI, Thompson Sampling, Entropy Search.
- Add: GP surrogate, base_optimizer.py when you run BO in code.

### `src/utils/`
- load_challenge_data.py (in use).
- Add: logging.py, visualization.py, metrics.py as needed.

### `configs/problems/`
- function_1.yaml (in use). Add function_2 … function_8 when you work on them.

### `tests/`
- test_optimizers/, test_utils/: add tests when you add code.

### `docs/`
- project_roadmap.md, Capstone_Project_FAQs.md. Add learning_log.md, algorithms_summary.md, etc. as needed.
