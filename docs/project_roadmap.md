# Project roadmap: planned structure

Components listed here are not yet in use. Add them under the paths below as you implement features. When something is in active use, you can move its description back into the README.

## Full project structure (target)

```
black-box-optimization/
├── initial_data/                 # Raw challenge data (DO NOT MODIFY)
│   ├── function_1/ … function_8/
│
├── phase_a_training/             # Stage 1 (reference only)
│
├── src/
│   ├── optimizers/
│   │   ├── gradient_free/         # Random Search, Nelder-Mead, Pattern Search
│   │   ├── evolutionary/         # GA, Differential Evolution, CMA-ES
│   │   ├── bayesian/              # Gaussian Process, acquisition functions
│   │   └── base_optimizer.py      # Abstract base class
│   ├── objective/
│   │   ├── black_box.py           # Black-box wrapper
│   │   ├── test_functions/        # sphere, Rastrigin, Rosenbrock, Ackley
│   │   └── evaluator.py
│   ├── utils/
│   │   ├── load_challenge_data.py # (in use)
│   │   ├── logging.py
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   └── comparison.py
│   └── experiments/
│       ├── runner.py
│       └── benchmark.py
│
├── data/
│   ├── problems/
│   ├── submissions/
│   └── results/ (training/, experiments/)
│
├── notebooks/
│   ├── weekly_review/
│   ├── function_1_explore.ipynb  # (in use)
│   ├── algorithm_comparison.ipynb
│   ├── convergence_analysis.ipynb
│   └── final_benchmarks.ipynb
│
├── configs/
│   ├── algorithms/                # Algorithm hyperparameters
│   ├── problems/                 # (function_1.yaml in use)
│   └── experiments/              # Experiment setups
│
├── tests/
│   ├── test_optimizers/
│   ├── test_objectives/
│   └── test_utils/
│
├── scripts/
│   ├── run_experiment.py
│   ├── benchmark_all.py
│   └── generate_report.py
│
├── docs/
│   ├── project_roadmap.md        # (this file)
│   ├── learning_log.md
│   ├── algorithms_summary.md
│   ├── key_concepts.md
│   └── project_notes.md
│
├── submission-template/
├── requirements.txt
├── .gitignore
└── README.md
```

## Planned components (incorporate gradually)

### `src/optimizers/`
- gradient_free/: Random Search, Nelder-Mead, Pattern Search.
- evolutionary/: Genetic Algorithm, Differential Evolution, CMA-ES.
- bayesian/: Gaussian Process surrogate, acquisition functions (UCB, EI).
- base_optimizer.py: Abstract base class for a common interface.

### `src/objective/`
- black_box.py: Wrapper for the black-box interface.
- test_functions/: Benchmark problems (sphere, Rastrigin, Rosenbrock, Ackley) for local testing.
- evaluator.py: Function evaluation management.

### `src/utils/` (extras)
- logging.py, visualization.py, metrics.py, comparison.py — add as needed.

### `src/experiments/`
- runner.py: Experiment execution framework.
- benchmark.py: Benchmarking utilities.

### `configs/algorithms/`
- JSON configs for algorithm hyperparameters (e.g. GP kernel, acquisition params).

### `configs/experiments/`
- JSON configs for experiment setups (which functions, how many rounds, etc.).

### `scripts/`
- run_experiment.py: Run a single experiment or round.
- benchmark_all.py: Run benchmarks across functions/algorithms.
- generate_report.py: Generate analysis or submission reports.

### `tests/`
- test_optimizers/, test_objectives/, test_utils/: Unit and integration tests as you add code.

### `notebooks/`
- weekly_review/: Weekly strategy notes and summaries.
- algorithm_comparison.ipynb, convergence_analysis.ipynb, final_benchmarks.ipynb: Add when you compare strategies or analyze convergence.

### `docs/`
- learning_log.md: Track learning progress.
- algorithms_summary.md: Reference for algorithms you implement.
- key_concepts.md: Important concepts and formulas.
- project_notes.md: General project notes.
