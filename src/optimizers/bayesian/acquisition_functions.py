"""
Acquisition functions for Bayesian optimization.

References
----------
.. [1] Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian process
       optimization in the bandit setting: no regret and experimental design.
       ICML. (GP-UCB: α(x) = μ(x) + κ σ(x).)
.. [2] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization
       of expensive black-box functions. Journal of Global Optimization, 13(4), 455–492.
       (Expected Improvement; EI = E[max(f(x) − f(x⁺), 0)] in closed form via Gaussian.)
.. [3] Kushner, H. J. (1964). A new method of locating the maximum point of an arbitrary
       multipeak curve in the presence of noise. J. Basic Eng., 86(1), 97–106.
       (Probability of Improvement; PI = P(f(x) > f(x⁺)) = Φ(Z).)
.. [4] Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds
       another in view of the evidence of two samples. Biometrika, 25(3/4), 285–294.
       (Thompson sampling: sample from posterior and maximize the sample.)
.. [5] Hennig, P., & Schuler, C. J. (2012). Entropy search for information-efficient
       global optimization. JMLR, 13(1), 1809–1837.
       (Full ES: minimize entropy of location of optimum; no simple closed form.)
"""
import numpy as np
from scipy.stats import norm


def upper_confidence_bound(mu, sigma, kappa=2.0):
    """
    Upper Confidence Bound (UCB) acquisition function.

    α(x) = μ(x) + κ σ(x). Reference: [1]_ (GP-UCB).

    Parameters
    ----------
    mu : array-like
        Predicted mean.
    sigma : array-like
        Predicted standard deviation.
    kappa : float, default=2.0
        Exploration parameter (higher = more exploration).

    Returns
    -------
    array-like
        UCB value at each point.

    Notes
    -----
    When the GP mean μ is nearly constant across candidates (e.g. sparse/flat
    objective, most observations ~0), UCB ≈ κ σ so argmax(UCB) = argmax(σ).
    In that regime the chosen point does not depend on κ; kappa only matters
    when μ varies enough to trade off with σ.
    """
    return mu + kappa * sigma


def expected_improvement(mu, sigma, y_best, xi=0.01):
    """
    Expected Improvement (EI) acquisition function.

    EI = E[max(f(x) − f(x⁺), 0)] in closed form (Gaussian). Reference: [2]_.

    Parameters
    ----------
    mu : array-like
        Predicted mean.
    sigma : array-like
        Predicted standard deviation.
    y_best : float
        Best observed value so far.
    xi : float, default=0.01
        Exploration parameter.

    Returns
    -------
    array-like
        Expected improvement at each point.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    with np.errstate(divide="warn"):
        improvement = mu - y_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """
    Probability of Improvement (PI) acquisition function.

    PI = P(f(x) > y_best) = Φ((μ − y_best − ξ) / σ). Reference: [3]_.

    Parameters
    ----------
    mu : array-like
        Predicted mean.
    sigma : array-like
        Predicted standard deviation.
    y_best : float
        Best observed value so far.
    xi : float, default=0.01
        Exploration parameter.

    Returns
    -------
    array-like
        Probability of improvement at each point.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    with np.errstate(divide="warn"):
        Z = (mu - y_best - xi) / sigma
        pi = norm.cdf(Z)
        # When sigma=0: PI = 1 if mu > y_best + xi else 0
        zero_sigma = sigma == 0.0
        pi[zero_sigma] = np.where(mu[zero_sigma] > y_best + xi, 1.0, 0.0)
    return pi


def thompson_sampling_sample(mu, sigma, rng=None):
    """
    Thompson Sampling: draw one sample from the posterior at each point.

    Sample from N(μ, σ²) then maximize the sample. Reference: [4]_.
    Returns a single sample (one random surface); call repeatedly for different samples.

    Parameters
    ----------
    mu : array-like
        Predicted mean.
    sigma : array-like
        Predicted standard deviation.
    rng : np.random.Generator or None, default=None
        Random number generator. If None, uses np.random.default_rng().

    Returns
    -------
    array-like
        One sample from N(mu, sigma²) at each point.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    if rng is None:
        rng = np.random.default_rng()
    return mu + sigma * rng.standard_normal(size=mu.shape)


def entropy_search(mu, sigma, y_best, xi=0.01):
    """
    Entropy Search (ES) acquisition function.

    Simplified proxy: −σ(x) to favor reducing uncertainty. Full ES [5]_ minimizes
    entropy of the distribution of the global minimizer (no simple closed form).

    Parameters
    ----------
    mu : array-like
        Predicted mean.
    sigma : array-like
        Predicted standard deviation.
    y_best : float
        Best observed value so far (unused in this proxy).
    xi : float, default=0.01
        Exploration parameter (unused in this proxy).

    Returns
    -------
    array-like
        Acquisition value (higher = more reduction in uncertainty).
    """
    _ = y_best, xi  # unused in simplified proxy
    sigma = np.asarray(sigma)
    return -sigma
