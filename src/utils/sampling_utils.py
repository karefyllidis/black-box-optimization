"""
Uniform / space-filling sampling for candidate points in black-box optimisation.

Use these to build candidate pools (e.g. for maximising acquisition functions) with better
coverage than plain random uniform. Methods:
- random: i.i.d. uniform.
- lhs: Latin Hypercube — one point per bin per dimension.
- sobol: Sobol (quasi-Monte Carlo) — low-discrepancy.
- grid: regular lattice (cell centres) — deterministic, uniform coverage; best when n ≈ k^d.
"""

from __future__ import annotations

import numpy as np


def sample_candidates(
    n: int,
    dim: int,
    method: str = "random",
    seed: int | None = 42,
) -> np.ndarray:
    """
    Sample n points in [0, 1]^dim for use as candidate points (e.g. for acquisition maximisation).

    Parameters
    ----------
    n : int
        Number of points.
    dim : int
        Dimension (number of inputs).
    method : str
        - "random": i.i.d. uniform (default; reproducible with seed).
        - "lhs": Latin Hypercube Sampling — one point per bin per dimension; good space-filling.
        - "sobol": Sobol sequence (quasi-Monte Carlo) — low-discrepancy; often better than random.
        - "grid": regular lattice (cell centres) — deterministic, uniform coverage; n ≈ k^d.
    seed : int or None
        Random seed for reproducibility. Used for "random" and "lhs"; Sobol uses its own state;
        "grid" is deterministic (seed ignored).

    Returns
    -------
    np.ndarray
        Shape (n, dim), values in [0, 1].
    """
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 1, (n, dim)).astype(np.float64)

    if method == "lhs":
        from scipy.stats import qmc
        rng = np.random.default_rng(seed)
        sampler = qmc.LatinHypercube(d=dim, seed=rng)
        # LHS gives [0,1); scale to (0,1] or keep [0,1) and clip if needed
        x = sampler.random(n=n)
        return np.clip(x, 1e-10, 1.0 - 1e-10).astype(np.float64)  # avoid exact 0/1 for stability

    if method == "sobol":
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, seed=seed)
        # Sobol typically needs 2^m points; we take next power of 2 >= n then take first n
        n_sobol = max(n, 2)
        x = sampler.random(n=n_sobol)[:n]
        return np.clip(x, 1e-10, 1.0 - 1e-10).astype(np.float64)

    if method == "grid":
        # Regular lattice: k points per axis, cell centres in [0, 1]; total points k^dim.
        # Choose k so that k^dim >= n, then take first n in row-major order.
        k = max(2, int(np.ceil(n ** (1.0 / dim))))
        # Cell centres: (0.5, 1.5, ... )/k scaled to (0,1) -> (i+0.5)/k for i in 0..k-1
        # so edges 0 and 1 are half a step away; use linspace for symmetric [0,1] coverage
        axes = [np.linspace(0.0, 1.0, num=k, dtype=np.float64) for _ in range(dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        # Stack to (k^dim, dim), then take first n
        lattice = np.stack([m.ravel() for m in mesh], axis=1)
        n_actual = min(n, lattice.shape[0])
        out = lattice[:n_actual].copy()
        # If we need exactly n and have fewer grid points, pad with centre point (deterministic)
        if out.shape[0] < n:
            pad = np.full((n - out.shape[0], dim), 0.5, dtype=np.float64)
            out = np.vstack([out, pad])
        return np.clip(out, 1e-10, 1.0 - 1e-10).astype(np.float64)

    raise ValueError(
        f"method must be 'random', 'lhs', 'sobol', or 'grid'; got {method!r}"
    )


def sample_candidates_legacy(
    n: int,
    dim: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Legacy interface: same as sample_candidates(n, dim, method='random', seed=seed).
    Use sample_candidates(..., method='lhs') or method='sobol' for uniform coverage.
    """
    return sample_candidates(n, dim, method="random", seed=seed)
