"""
Output warping for GP surrogates (HEBO-inspired).
Transform y before fitting the GP so the surrogate sees a better-behaved response;
acquisition and next_x are computed in warped space.
"""

import numpy as np


def apply_output_warping(y, mode=None, eps=1e-10):
    """
    Optionally warp y for GP fitting (e.g. Box-Cox or log).

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed outputs (raw).
    mode : str or None
        None: no warping. "boxcox": Box-Cox transform. "log": log transform.
        For Box-Cox and log, y is shifted so all values are positive (y - min(y) + eps).
    eps : float
        Small constant added so shifted y is strictly positive.

    Returns
    -------
    y_warped : np.ndarray, shape (n,)
        Warped y (or copy of y if mode is None or invalid).
    warp_params : tuple or None
        Parameters for inverse transform: ("boxcox", lam, y_min, eps) or ("log", y_min, eps), or None.
    message : str
        Short description for logging/print.
    """
    y_orig = np.asarray(y, dtype=np.float64).ravel()
    warp_params = None
    message = "No output warping (mode is None). GP fits raw y."

    if not mode:
        return y_orig.copy(), warp_params, message

    y_min = np.min(y_orig)
    y_shift = y_orig - y_min + eps

    if mode == "boxcox":
        from scipy.stats import boxcox
        y_warped, lam = boxcox(y_shift)
        warp_params = ("boxcox", float(lam), float(y_min), eps)
        message = f"Output warping: Box-Cox applied (λ={lam:.4g}). y now in warped space; GP and acquisition use warped y."
        return np.asarray(y_warped, dtype=np.float64), warp_params, message

    if mode == "log":
        y_warped = np.log(y_shift).astype(np.float64)
        warp_params = ("log", float(y_min), eps)
        message = "Output warping: log applied. y now in warped space; GP and acquisition use warped y."
        return y_warped, warp_params, message

    message = "OUTPUT_WARPING set but not 'boxcox' or 'log'; skipping."
    return y_orig.copy(), warp_params, message


def inverse_output_warping(y_warped, warp_params):
    """
    Inverse of apply_output_warping: map from warped space back to original y scale.

    Parameters
    ----------
    y_warped : array-like
        Values in warped space.
    warp_params : tuple
        As returned by apply_output_warping: ("boxcox", lam, y_min, eps) or ("log", y_min, eps).

    Returns
    -------
    y_orig : np.ndarray
        Values in original scale.
    """
    if warp_params is None:
        return np.asarray(y_warped, dtype=np.float64)
    kind = warp_params[0]
    y_w = np.asarray(y_warped, dtype=np.float64)
    if kind == "boxcox":
        from scipy.special import inv_boxcox
        _, lam, y_min, eps = warp_params
        return inv_boxcox(y_w, lam) + y_min - eps
    if kind == "log":
        _, y_min, eps = warp_params
        return np.exp(y_w) + y_min - eps
    return y_w
