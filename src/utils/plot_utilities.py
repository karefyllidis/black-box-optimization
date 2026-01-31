"""
Plot styling utilities: font sizes, colorbars, axis labels and titles.
Includes ready-made plots for 1D/2D GP, acquisition, BO state, convergence, and parallel coordinates.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.contour import QuadContourSet
from matplotlib.collections import PathCollection
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Default parameters (plots)
# ---------------------------------------------------------------------------
# Font sizes: axis labels, colorbar labels/ticks, legend vs subplot/figure titles
DEFAULT_FONT_SIZE_AXIS = 8
DEFAULT_FONT_SIZE_TITLES = 10
# Export: resolution and format when saving figures (directory is set by caller, e.g. repo_root / "data" / "results")
DEFAULT_EXPORT_DPI = 150
DEFAULT_EXPORT_FORMAT = "png"


def add_colorbar(
    mappable: QuadContourSet | ScalarMappable | PathCollection,
    ax: Axes | None = None,
    *,
    label: str | None = None,
    font_size_axis: int | None = None,
    shrink: float = 1.0,
    **kwargs: Any,
) -> Colorbar:
    """
    Add a colorbar to an axis and set label/tick font size.

    Parameters
    ----------
    mappable : contourf, scatter, or ScalarMappable
        The mappable to attach the colorbar to.
    ax : Axes, optional
        Axis to attach the colorbar to. If None, uses current figure.
    label : str, optional
        Colorbar label text.
    font_size_axis : int, optional
        Font size for colorbar label and tick labels. Default: DEFAULT_FONT_SIZE_AXIS.
    shrink : float, optional
        Fraction of axis size for the colorbar (e.g. 0.8). Default 1.0.
    **kwargs
        Passed to plt.colorbar() (e.g. shrink=0.8 for 3D).

    Returns
    -------
    Colorbar
        The colorbar instance (e.g. to call set_label / tick_params if needed).
    """
    font_size_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    pad = kwargs.pop("pad", 0.04)  # fraction of axes width between plot and colorbar (matplotlib default ~0.02)
    cb = plt.colorbar(mappable, ax=ax, shrink=shrink, pad=pad, **kwargs)
    if label is not None:
        cb.set_label(label, fontsize=font_size_axis)
    cb.ax.tick_params(labelsize=font_size_axis)
    return cb


def style_axis(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> None:
    """
    Set axis labels and title with consistent font sizes.

    Parameters
    ----------
    ax : Axes
        The axis to style.
    xlabel, ylabel, title : str, optional
        Text for x-axis, y-axis, and title.
    font_size_axis : int, optional
        Font size for xlabel and ylabel. Default: DEFAULT_FONT_SIZE_AXIS.
    font_size_titles : int, optional
        Font size for title. Default: DEFAULT_FONT_SIZE_TITLES.
    """
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs_axis)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs_axis)
    if title is not None:
        ax.set_title(title, fontsize=fs_titles)


def style_axis_3d(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    title: str | None = None,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> None:
    """
    Set 3D axis labels and title with consistent font sizes.

    Parameters
    ----------
    ax : Axes (with projection='3d')
        The 3D axis to style.
    xlabel, ylabel, zlabel, title : str, optional
        Text for axes and title.
    font_size_axis : int, optional
        Font size for axis labels. Default: DEFAULT_FONT_SIZE_AXIS.
    font_size_titles : int, optional
        Font size for title. Default: DEFAULT_FONT_SIZE_TITLES.
    """
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs_axis)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs_axis)
    if zlabel is not None:
        ax.set_zlabel(zlabel, fontsize=fs_axis)
    if title is not None:
        ax.set_title(title, fontsize=fs_titles)


def style_legend(
    ax: Axes,
    *,
    font_size_axis: int | None = None,
    **kwargs: Any,
) -> Any:
    """
    Add or update legend with consistent font size.

    Parameters
    ----------
    ax : Axes
        The axis (legend is taken from ax or created).
    font_size_axis : int, optional
        Font size for legend. Default: DEFAULT_FONT_SIZE_AXIS.
    **kwargs
        Passed to ax.legend() (e.g. loc='upper right').

    Returns
    -------
    Legend
        The legend instance.
    """
    fs = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    return ax.legend(fontsize=fs, **kwargs)


# ---------------------------------------------------------------------------
# Ready-made plot functions (1D GP, acquisition, BO iteration, 2D, convergence, parallel coords)
# ---------------------------------------------------------------------------


def plot_gp_1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    title: str = "Gaussian Process",
    *,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> tuple[Any, Axes]:
    """
    Plot 1D Gaussian Process with uncertainty bands.

    Parameters
    ----------
    X_train : array-like
        Training points.
    y_train : array-like
        Training observations.
    X_test : array-like
        Test points for prediction.
    mu : array-like
        Predicted mean.
    sigma : array-like
        Predicted standard deviation.
    title : str
        Plot title.
    font_size_axis, font_size_titles : int, optional
        Override default axis/title font sizes.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES

    ax.plot(X_test, mu, "b-", label="Mean prediction", linewidth=2)
    ax.fill_between(
        X_test.ravel(),
        mu - 1.96 * sigma,
        mu + 1.96 * sigma,
        alpha=0.3,
        label="95% confidence",
    )
    ax.scatter(
        X_train, y_train, c="red", s=50, zorder=10, edgecolors="black", label="Observations"
    )
    style_axis(ax, xlabel="x", ylabel="f(x)", title=title, font_size_axis=fs_axis, font_size_titles=fs_titles)
    style_legend(ax, loc="best", font_size_axis=fs_axis)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_acquisition_1d(
    X_test: np.ndarray,
    acquisition_values: np.ndarray,
    X_next: float,
    title: str = "Acquisition Function",
    *,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> tuple[Any, Axes]:
    """
    Plot 1D acquisition function and next sampling point.

    Parameters
    ----------
    X_test : array-like
        Test points.
    acquisition_values : array-like
        Acquisition function values.
    X_next : float
        Next point to sample.
    title : str
        Plot title.
    font_size_axis, font_size_titles : int, optional
        Override default axis/title font sizes.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES

    ax.plot(X_test, acquisition_values, "g-", linewidth=2, label="Acquisition")
    ax.axvline(
        X_next, color="red", linestyle="--", linewidth=2, label=f"Next sample: x={X_next:.3f}"
    )
    style_axis(
        ax,
        xlabel="x",
        ylabel="Acquisition Value",
        title=title,
        font_size_axis=fs_axis,
        font_size_titles=fs_titles,
    )
    style_legend(ax, loc="best", font_size_axis=fs_axis)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_bo_iteration_1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    acquisition_values: np.ndarray,
    X_next: float,
    true_func: Callable[..., np.ndarray] | None = None,
    iteration: int = 0,
    acq_name: str = "Acquisition",
    *,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> tuple[Any, tuple[Axes, Axes]]:
    """
    Combined plot: GP surrogate and acquisition function for one BO iteration.

    Parameters
    ----------
    X_train, y_train : array-like
        Current training points and observations.
    X_test : array-like
        Test points.
    mu, sigma : array-like
        GP mean and standard deviation.
    acquisition_values : array-like
        Acquisition function values.
    X_next : float
        Next point to sample.
    true_func : callable, optional
        True function to overlay (e.g. lambda x: true_func(x)).
    iteration : int
        Iteration number.
    acq_name : str
        Name of acquisition function.
    font_size_axis, font_size_titles : int, optional
        Override default axis/title font sizes.

    Returns
    -------
    fig, (ax1, ax2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES

    # GP panel
    ax1.plot(X_test, mu, "b-", label="GP Mean", linewidth=2)
    ax1.fill_between(
        X_test.ravel(), mu - 1.96 * sigma, mu + 1.96 * sigma, alpha=0.3, label="95% CI"
    )
    if true_func is not None:
        y_true = true_func(X_test)
        ax1.plot(X_test, y_true, "k--", alpha=0.4, label="True function")
    ax1.scatter(
        X_train, y_train, c="red", s=50, zorder=10, edgecolors="black", label="Observations"
    )
    ax1.axvline(X_next, color="red", linestyle=":", alpha=0.5)
    style_axis(
        ax1,
        xlabel="x",
        ylabel="f(x)",
        title=f"Iteration {iteration}: GP Surrogate",
        font_size_axis=fs_axis,
        font_size_titles=fs_titles,
    )
    style_legend(ax1, loc="best", font_size_axis=fs_axis)
    ax1.grid(True, alpha=0.3)

    # Acquisition panel
    ax2.plot(X_test, acquisition_values, "g-", linewidth=2)
    ax2.axvline(
        X_next, color="red", linestyle="--", linewidth=2, label=f"Next: x={X_next:.3f}"
    )
    style_axis(
        ax2,
        xlabel="x",
        ylabel="Acquisition Value",
        title=acq_name,
        font_size_axis=fs_axis,
        font_size_titles=fs_titles,
    )
    style_legend(ax2, loc="best", font_size_axis=fs_axis)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_2d_function(
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    title: str = "2D Function",
    *,
    levels: int = 20,
    cmap: str = "viridis",
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> tuple[Any, Axes]:
    """
    Plot a 2D function as a contour plot.

    Parameters
    ----------
    X1, X2 : 2D arrays
        Meshgrid coordinates.
    Z : 2D array
        Function values.
    title : str
        Plot title.
    levels : int
        Number of contour levels.
    cmap : str
        Colormap name.
    font_size_axis, font_size_titles : int, optional
        Override default axis/title font sizes.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    contour = ax.contourf(X1, X2, Z, levels=levels, cmap=cmap)
    add_colorbar(
        contour, ax=ax, font_size_axis=font_size_axis or DEFAULT_FONT_SIZE_AXIS
    )
    style_axis(
        ax,
        xlabel="x1",
        ylabel="x2",
        title=title,
        font_size_axis=font_size_axis,
        font_size_titles=font_size_titles,
    )
    plt.tight_layout()
    return fig, ax


def plot_2d_bo_state(
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    title: str = "Bayesian Optimization Progress",
    *,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> tuple[Any, Axes]:
    """
    Plot 2D function with sampled points overlay and best point highlighted.

    Parameters
    ----------
    X1, X2 : 2D arrays
        Meshgrid coordinates.
    Z : 2D array
        Function values.
    X_samples : array-like, shape (n_samples, 2)
        Sampled points.
    y_samples : array-like
        Function values at samples.
    title : str
        Plot title.
    font_size_axis, font_size_titles : int, optional
        Override default axis/title font sizes.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES

    contour = ax.contourf(X1, X2, Z, levels=20, cmap="viridis", alpha=0.8)
    add_colorbar(contour, ax=ax, label="f(x1, x2)", font_size_axis=fs_axis)

    scatter = ax.scatter(
        X_samples[:, 0],
        X_samples[:, 1],
        c=y_samples,
        s=100,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=2,
        zorder=10,
    )
    best_idx = np.argmax(y_samples)
    ax.scatter(
        X_samples[best_idx, 0],
        X_samples[best_idx, 1],
        marker="*",
        s=500,
        c="gold",
        edgecolors="black",
        linewidths=2,
        zorder=11,
        label="Best found",
    )
    style_axis(
        ax,
        xlabel="x1",
        ylabel="x2",
        title=title,
        font_size_axis=fs_axis,
        font_size_titles=fs_titles,
    )
    style_legend(ax, loc="best", font_size_axis=fs_axis)
    plt.tight_layout()
    return fig, ax


def plot_convergence(
    iterations: np.ndarray,
    best_values: np.ndarray,
    true_optimum: float | None = None,
    *,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> tuple[Any, Axes]:
    """
    Plot convergence of best found value over iterations.

    Parameters
    ----------
    iterations : array-like
        Iteration numbers.
    best_values : array-like
        Best value found at each iteration.
    true_optimum : float, optional
        True optimal value (horizontal line).
    font_size_axis, font_size_titles : int, optional
        Override default axis/title font sizes.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES

    ax.plot(
        iterations, best_values, "b-o", linewidth=2, markersize=6, label="Best value found"
    )
    if true_optimum is not None:
        ax.axhline(
            true_optimum,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True optimum: {true_optimum:.4f}",
        )
    style_axis(
        ax,
        xlabel="Iteration",
        ylabel="Best f(x) found",
        title="Optimization Convergence",
        font_size_axis=fs_axis,
        font_size_titles=fs_titles,
    )
    style_legend(ax, loc="best", font_size_axis=fs_axis)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_parallel_coordinates(
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    n_best: int = 5,
    *,
    font_size_axis: int | None = None,
    font_size_titles: int | None = None,
) -> tuple[Any, Axes]:
    """
    Plot parallel coordinates for high-dimensional optimization results.

    Parameters
    ----------
    X_samples : array-like, shape (n_samples, n_dims)
        Sampled points.
    y_samples : array-like
        Function values.
    n_best : int
        Number of best samples to highlight.
    font_size_axis, font_size_titles : int, optional
        Override default axis/title font sizes.

    Returns
    -------
    fig, ax
    """
    n_dims = X_samples.shape[1]
    fig, ax = plt.subplots(figsize=(10, 4))
    fs_axis = font_size_axis if font_size_axis is not None else DEFAULT_FONT_SIZE_AXIS
    fs_titles = font_size_titles if font_size_titles is not None else DEFAULT_FONT_SIZE_TITLES

    best_indices = np.argsort(y_samples)[-n_best:]
    for i in range(len(X_samples)):
        if i not in best_indices:
            ax.plot(range(n_dims), X_samples[i], "gray", alpha=0.2, linewidth=1)
    colors = cm.viridis(np.linspace(0.3, 1, n_best))
    for idx, color in zip(best_indices, colors):
        ax.plot(
            range(n_dims),
            X_samples[idx],
            color=color,
            linewidth=2,
            alpha=0.8,
            label=f"f(x)={y_samples[idx]:.3f}",
        )
    style_axis(
        ax,
        xlabel="Dimension",
        ylabel="Value",
        title=f"Parallel Coordinates - Top {n_best} Solutions",
        font_size_axis=fs_axis,
        font_size_titles=fs_titles,
    )
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f"x{i}" for i in range(n_dims)], fontsize=fs_axis)
    style_legend(ax, loc="best", font_size_axis=fs_axis)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig, ax
