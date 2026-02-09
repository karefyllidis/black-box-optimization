"""
Load challenge data from initial_data (read-only).
Do not modify or write to initial_data; copy elsewhere if you need to extend the dataset.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

# Original challenge data: never overwrite. All writes must go elsewhere (e.g. data/problems/, data/results/).
INITIAL_DATA_READ_ONLY = "initial_data"


def get_initial_data_root(project_root: Path | None = None) -> Path:
    """Return path to initial_data directory. Default: one level above src/."""
    if project_root is not None:
        return Path(project_root) / "initial_data"
    # Assume this file lives in src/utils/
    return Path(__file__).resolve().parent.parent.parent / "initial_data"


def load_function_data(
    function_id: int,
    data_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load initial (inputs, outputs) for one challenge function. Read-only.

    Parameters
    ----------
    function_id : int
        1..8
    data_root : Path, optional
        Path to initial_data. If None, uses repo initial_data/.

    Returns
    -------
    X : np.ndarray
        Shape (n, d) — n initial input points (e.g. 10), d dimensions.
    y : np.ndarray
        Shape (n,) — corresponding objective values (maximize).
    """
    root = data_root if data_root is not None else get_initial_data_root()
    folder = root / f"function_{function_id}"
    if not folder.exists():
        raise FileNotFoundError(f"Challenge data not found: {folder}")
    X = np.load(folder / "initial_inputs.npy")
    y = np.load(folder / "initial_outputs.npy")
    if y.ndim > 1:
        y = y.squeeze()
    return X, y


def load_problem_data_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (X, y) from a problem observations CSV (header: x1,x2,...,xd,y).
    Returns (X, y) with X shape (n, d) and y shape (n,).
    """
    import csv
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    n, ncols = len(rows), len(header)
    if ncols < 2:
        raise ValueError(f"CSV must have at least x1 and y columns: {csv_path}")
    # Last column is y
    X = np.array([[float(r[i]) for i in range(ncols - 1)] for r in rows], dtype=np.float64)
    y = np.array([float(r[ncols - 1]) for r in rows], dtype=np.float64)
    return X, y


def save_problem_data_csv(csv_path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """
    Save (X, y) to a problem observations CSV (header: x1,x2,...,xd,y).
    X shape (n, d), y shape (n,).
    """
    import csv
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    assert len(y) == len(X), "X and y length mismatch"
    d = X.shape[1]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"x{i+1}" for i in range(d)] + ["y"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(y)):
            writer.writerow(list(X[i]) + [float(y[i])])


def assert_not_under_initial_data(path: Path, project_root: Path | None = None) -> None:
    """
    Raise if path is under initial_data (original challenge data is read-only).
    Call this before any write so we never overwrite original data.
    """
    path = Path(path).resolve()
    # Forbid writes only under initial_data, not under project root
    root = (Path(project_root) / "initial_data").resolve() if project_root is not None else get_initial_data_root().resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return  # path is not under initial_data — OK
    raise PermissionError(
        f"Refusing to write under {root} (original data is read-only). "
        "Use data/problems/, data/submissions/, or data/results/ instead."
    )
