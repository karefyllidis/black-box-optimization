# Technical foundations

Short reference for the main justification, key literature, and library choices behind this BBO capstone. See README §4 (Technical approach) and References for full detail.

---

## Main justification

- **Approach:** Bayesian optimisation (BO) with a **Gaussian process (GP) surrogate** and an **acquisition function** (Expected Improvement, EI; or UCB, PI).
- **Prior research:** BO is the standard sample-efficient framework for expensive black-box objectives with limited evaluations; the loop (fit surrogate → maximise acquisition → evaluate → update) is well understood and has theoretical support (e.g. regret bounds).
- **Established benchmark:** The **NeurIPS 2020 BBO Challenge** used the same suggest–observe API; many competitive entries used GP surrogates and EI/UCB-style acquisition (skopt, TuRBO, team submissions). Our design aligns with this benchmark.
- **Why GPs:** They provide a predictive mean and **uncertainty** σ(x) natively (required by EI/UCB), are data-efficient for small n (10–20+ points per function), and scale to our dimensions (2D–8D).

---

## Key papers and ideas

| Source | Idea / technique | How it strengthens this project |
|--------|------------------|---------------------------------|
| **Rasmussen & Williams**, *Gaussian Processes for Machine Learning* | GP regression as a distribution over functions; kernel choice (RBF, Matérn); hyperparameters via log-marginal likelihood (LML) | Justifies the surrogate (calibrated uncertainty, kernel selection); kernel choice (RBF, Matérn, RBF+WhiteKernel with LML) is traceable to established practice. |
| **Jones et al.** (Expected Improvement) | EI as an acquisition function balancing exploration and exploitation using μ and σ | Gives a principled, non–ad hoc criterion for the next query. |
| **NeurIPS 2020 BBO Challenge** (organisers’ report, starter kit) | Suggest–observe API; space-filling candidates (Sobol); avoidance of re-querying the same points | Aligns implementation with a comparable benchmark and challenge-tested methods; supports duplicate-avoidance (e.g. MIN_DIST_THRESHOLD) and Sobol candidates. |

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
- **Notebooks** — Parameters cell per function: kernel choice, acquisition coefficients, sampling method.
- **submission-template/** — Data sheet and model card for portfolio deliverable (summarise approach and point to README and references).
- **docs_private/similar_projects/** — Notes from BBO starter kit; references (e.g. HEBO) for optional follow-up.

---

## Additional sources (for ongoing refinement)

- **Research:** NeurIPS 2020 BBO team write-ups (e.g. Huawei HEBO, Nvidia ensembles, JetBrains GP+SVM); TuRBO and trust-region BO for high dimensions.
- **Benchmarks:** Bayesmark (from BBO starter kit); HPOBench or similar BO benchmarks.
- **Software:** hyperopt, nevergrad (starter kit); GPyTorch/BoTorch if budget or dimension grows.
