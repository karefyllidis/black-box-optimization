# Black-Box Optimization Project

A comprehensive project structure for learning and implementing black-box optimization algorithms.

Nikolas Karefyllidis, PhD

## Project Structure (dummy subfolders and .py for now)

```
black-box-optimization/
├── phase_a_training/            # Training materials and practice work (NOT git tracked)
│   ├── docs/                    # Course materials and documentation
│   └── notebooks/               # Training notebooks and exercises
│
├── src/                        # Your main codebase (production-quality)
│   ├── optimizers/             # Note: All .py files and subfolders are placeholders (dummy for now)
│   │   ├── gradient_free/
│   │   │   ├── random_search.py
│   │   │   ├── nelder_mead.py
│   │   │   └── pattern_search.py
│   │   ├── evolutionary/
│   │   │   ├── genetic_algorithm.py
│   │   │   ├── differential_evolution.py
│   │   │   └── cma_es.py
│   │   ├── bayesian/
│   │   │   ├── gaussian_process.py
│   │   │   └── acquisition.py
│   │   └── base_optimizer.py   # Abstract base class
│   │
│   ├── objective/
│   │   ├── black_box.py        # Black box wrapper
│   │   ├── test_functions/     # Benchmark problems
│   │   │   ├── sphere.py
│   │   │   ├── rastrigin.py
│   │   │   ├── rosenbrock.py
│   │   │   └── ackley.py
│   │   └── evaluator.py
│   │
│   ├── utils/
│   │   ├── logging.py
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   └── comparison.py
│   │
│   └── experiments/
│       ├── runner.py
│       └── benchmark.py
│
├── data/
│   ├── problems/               # Problem instances
│   └── results/                # Experimental results
│       ├── training/           # Results from practice
│       └── experiments/        # Formal experiment results
│
├── notebooks/                  # Analysis and visualization
│   ├── weekly_review/          # Weekly learning summaries
│   │   ├── week03_review.ipynb
│   │   └── ...
│   ├── algorithm_comparison.ipynb
│   ├── convergence_analysis.ipynb
│   └── final_benchmarks.ipynb
│
├── configs/
│   ├── algorithms/             # Algorithm configs
│   ├── problems/               # Problem specifications
│   └── experiments/            # Experiment setups
│
├── tests/
│   ├── test_optimizers/
│   ├── test_objectives/
│   └── test_utils/
│
├── scripts/
│   ├── run_weekly_exercise.py
│   ├── run_experiment.py
│   ├── benchmark_all.py
│   └── generate_report.py
│
├── docs/
│   ├── learning_log.md         # Track your progress
│   ├── algorithms_summary.md   # Reference for algorithms learned
│   ├── key_concepts.md         # Important concepts/formulas
│   └── project_notes.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Directory Descriptions

### `phase_a_training/`
Training materials and practice work (NOT git tracked):
- **docs/**: Course materials, video transcripts, and documentation
- **notebooks/**: Training notebooks and exercises

### `src/`
Production-quality source code organized by functionality:

- **optimizers/**: Implementation of various optimization algorithms
  - **gradient_free/**: Direct search methods (Random Search, Nelder-Mead, Pattern Search)
  - **evolutionary/**: Population-based methods (GA, DE, CMA-ES)
  - **bayesian/**: Bayesian optimization components (GP, acquisition functions)
  - **base_optimizer.py**: Abstract base class defining the optimizer interface

- **objective/**: Objective function handling
  - **black_box.py**: Wrapper for black-box functions
  - **test_functions/**: Benchmark optimization problems
  - **evaluator.py**: Function evaluation management

- **utils/**: Utility modules
  - **logging.py**: Logging configuration
  - **visualization.py**: Plotting and visualization tools
  - **metrics.py**: Performance metrics calculation
  - **comparison.py**: Algorithm comparison utilities

- **experiments/**: Experiment management
  - **runner.py**: Experiment execution framework
  - **benchmark.py**: Benchmarking utilities

### `data/`
Data storage:
- **problems/**: Problem instance files
- **results/**: Experimental results
  - **training/**: Results from practice exercises
  - **experiments/**: Formal experiment results

### `notebooks/`
Jupyter notebooks for analysis:
- **weekly_review/**: Weekly learning summaries and reflections
- **algorithm_comparison.ipynb**: Compare different algorithms
- **convergence_analysis.ipynb**: Analyze convergence behavior
- **final_benchmarks.ipynb**: Final benchmark results

### `configs/`
Configuration files:
- **algorithms/**: Algorithm hyperparameter configurations
- **problems/**: Problem specifications and settings
- **experiments/**: Experiment setup configurations

### `tests/`
Unit and integration tests:
- **test_optimizers/**: Tests for optimization algorithms
- **test_objectives/**: Tests for objective functions
- **test_utils/**: Tests for utility functions

### `scripts/`
Executable scripts:
- **run_weekly_exercise.py**: Run weekly exercises
- **run_experiment.py**: Execute experiments
- **benchmark_all.py**: Run comprehensive benchmarks
- **generate_report.py**: Generate analysis reports

### `docs/`
Documentation:
- **learning_log.md**: Track learning progress
- **algorithms_summary.md**: Reference guide for algorithms
- **key_concepts.md**: Important concepts and formulas
- **project_notes.md**: General project notes

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Explore the training materials in `phase_a_training/`

3. Implement algorithms in `src/optimizers/`

4. Run experiments using scripts in `scripts/`

5. Analyze results in `notebooks/`

## Notes

- Keep production code in `src/` separate from learning exercises
- Use `phase_a_training/` for practice and experimentation
- Document your learning progress in `docs/learning_log.md`
- Store experimental results in `data/results/`

