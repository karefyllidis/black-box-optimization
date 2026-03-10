# Technical foundations

Short reference for the main justification, key literature, and library choices behind this BBO capstone. See README §4 (Technical approach) and References for full detail.

---

## Main justification

- **Approach:** Bayesian optimisation (BO) with a **Gaussian process (GP) surrogate** and an **acquisition function** (Expected Improvement, EI; or UCB, PI).
- **Prior research:** BO is the standard sample-efficient framework for expensive black-box objectives with limited evaluations; the loop (fit surrogate → maximise acquisition → evaluate → update) is well understood and has theoretical support (e.g. regret bounds).
- **Established benchmark:** The **NeurIPS 2020 BBO Challenge** used the same suggest–observe API; many competitive entries used GP surrogates and EI/UCB-style acquisition (skopt, TuRBO, team submissions). Our design aligns with this benchmark.
- **Why GPs:** They provide a predictive mean and **uncertainty** σ(x) natively (required by EI/UCB), are data-efficient for small n (10–20+ points per function), and scale to our dimensions (2D–8D).
- **Output warping:** When y spans many orders of magnitude or is heavily skewed, the GP fit and uncertainty can be poor. We optionally transform y before fitting (HEBO-inspired): **log** or **Box–Cox** of a shifted y (so all values &gt; 0). The GP and acquisition use the warped y; the suggested next x is unchanged in input space. Implemented in `src/utils/warping.py` (`apply_output_warping`). Mode `None` or string `"none"` (case-insensitive) performs no transform. **When to use:** F1 (sparse tiny y), F5 (y ~0.1–1e3), F7 (y ~0.003–1.4) use `OUTPUT_WARPING = "log"` by default; F2, F3, F4, F6, F8 use `None`. Use `"log"` or `"boxcox"` when y is multi-scale or right-skewed; leave `None` when y is well-behaved.

---

## Key papers and ideas

| Source | Idea / technique | How it strengthens this project |
|--------|------------------|---------------------------------|
| **Rasmussen & Williams**, *Gaussian Processes for Machine Learning* | GP regression as a distribution over functions; kernel choice (RBF, Matérn); hyperparameters via log-marginal likelihood (LML) | Justifies the surrogate (calibrated uncertainty, kernel selection); kernel choice (RBF, Matérn, RBF+WhiteKernel with LML) is traceable to established practice. |
| **Jones et al.** (Expected Improvement) | EI as an acquisition function balancing exploration and exploitation using μ and σ | Gives a principled, non–ad hoc criterion for the next query. |
| **NeurIPS 2020 BBO Challenge** (organisers’ report, starter kit) | Suggest–observe API; space-filling candidates (Sobol); avoidance of re-querying the same points | Aligns implementation with a comparable benchmark and challenge-tested methods; supports duplicate-avoidance (e.g. MIN_DIST_THRESHOLD) and Sobol candidates. |
| **HEBO-style output warping** (e.g. log/Box–Cox of y) | Transform response before GP fit to stabilize variance when y is skewed or multi-scale | Justifies optional `OUTPUT_WARPING` in all notebooks; default `"log"` for F1, F5, F7; `None` for others. |

---

## Third-party libraries (role and justification)

| Library | Role | Why chosen over alternatives |
|---------|------|------------------------------|
| **scikit-learn** (`GaussianProcessRegressor`) | Core GP surrogate (fit, predict mean and std) | Stable API, built-in LML-based kernel optimisation, good behaviour for small n. **GPyTorch** would scale better to large n but adds complexity and is unnecessary for our evaluation budget. |
| **scikit-optimize (skopt)** (`gaussian_ei`, `gaussian_pi`, `gaussian_lcb`; `Sobol`, `Lhs`) | Compute EI/PI/UCB over a candidate set; generate space-filling candidates | Widely used in BO tutorials and NeurIPS 2020 starter kit; Sobol gives low-discrepancy coverage. We also implement EI/UCB/PI in `src/optimizers/bayesian/` for transparency; notebooks use skopt for consistency. |
| **NumPy, SciPy, Matplotlib** | Numerical computation and visualisation | Standard, stable stack. **PyTorch/TensorFlow** not chosen: surrogate is a GP, not a neural network; for our data size, a GP is more data-efficient and provides uncertainty without extra machinery. |

---

## Where this is documented in the repo

- **README.md** — §1–4: overview, inputs/outputs, technical approach; References: NeurIPS 2020 BBO, starter kit, sample repos.
- **docs/project_roadmap.md** — Structure, notebook workflow, planned components.
- **docs/Capstone_Project_FAQs.md** — Capstone-specific FAQs.
- **Notebooks** — Parameters cell per function: kernel choice, `OUTPUT_WARPING`, acquisition coefficients, sampling method. See README and project_roadmap for default scaling (F1, F5, F7: `"log"`; others: `None`).
- **submission-template/** — Data sheet and model card for portfolio deliverable (summarise approach and point to README and references).
- **docs_private/similar_projects/** — Notes from BBO starter kit; references (e.g. HEBO) for optional follow-up.

---

## Additional sources (for ongoing refinement)

- **Research:** NeurIPS 2020 BBO team write-ups (e.g. Huawei HEBO, Nvidia ensembles, JetBrains GP+SVM); TuRBO and trust-region BO for high dimensions.
- **Benchmarks:** Bayesmark (from BBO starter kit); HPOBench or similar BO benchmarks.
- **Software:** hyperopt, nevergrad (starter kit); GPyTorch/BoTorch if budget or dimension grows.
