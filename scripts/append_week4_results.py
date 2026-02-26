#!/usr/bin/env python3
"""
Append Week 4 portal results to local datasets (data/problems/function_N/).
Under data/ we use only CSV: observations.csv. No .npy in data/problems/.
Run from project root. Idempotent: skips if Week 4 point already in dataset.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT))
from src.utils.load_challenge_data import (
    load_function_data,
    load_problem_data_csv,
    save_problem_data_csv,
)

# Week 4 results from portal (input list, output scalar per function)
WEEK4 = {
    1: (np.array([0.994846, 0.995225]), 1.067788738253734e-187),
    2: (np.array([0.002223, 0.994219]), 0.08040688189057967),
    3: (np.array([0.495784, 0.208416, 0.589975]), -0.07234615305353402),
    4: (np.array([0.112039, 0.397856, 0.969470, 0.865507]), -26.987941085683477),
    5: (np.array([0.370941, 0.901940, 0.806694, 0.984858]), 1744.9667002452438),
    6: (np.array([0.230417, 0.001474, 0.729345, 0.966845, 0.224293]), -0.9655555705829371),
    7: (np.array([0.003720, 0.241506, 0.509723, 0.379960, 0.448485, 0.980648]), 0.8647811336323187),
    8: (np.array([0.186590, 0.188232, 0.184249, 0.057002, 0.587798, 0.840348, 0.017498, 0.049005]), 9.7307400534525),
}

ATOL = 1e-9

CSV_NAME = "observations.csv"


def _already_appended(X_saved: np.ndarray, x_new: np.ndarray) -> bool:
    if X_saved is None or len(X_saved) == 0:
        return False
    return np.any(np.all(np.isclose(X_saved, x_new.ravel(), atol=ATOL), axis=1))


def _load_current(out_dir: Path, fid: int):
    """Load current data from CSV only (under data/ we use only CSV). Returns (X, y) or None."""
    csv_path = out_dir / CSV_NAME
    if csv_path.exists():
        return load_problem_data_csv(csv_path)
    return None


def main():
    problems_dir = ROOT / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(1, 9):
        x_new, y_new = WEEK4[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = _load_current(out_dir, fid)
        if current is not None:
            X_cur, y_cur = current
            if _already_appended(X_cur, x_new):
                print(f"Function {fid}: already appended (Week 4 in dataset), skip. Points: {len(y_cur)}")
                continue
            assert x_new.shape[1] == X_cur.shape[1], f"Function {fid}: dimension mismatch"
            X_updated = np.vstack([X_cur, x_new])
            y_updated = np.append(y_cur, y_new)
        else:
            X_init, y_init = load_function_data(fid)
            assert x_new.shape[1] == X_init.shape[1], f"Function {fid}: dimension mismatch"
            X_updated = np.vstack([X_init, x_new])
            y_updated = np.append(y_init, y_new)

        out_dir.mkdir(parents=True, exist_ok=True)
        save_problem_data_csv(csv_path, X_updated, y_updated)
        n = len(y_updated)
        print(f"Function {fid}: {n} points -> {csv_path.name}")
    print("Done. data/problems/function_1..8 updated (CSV). Re-run notebooks for Week 5.")


if __name__ == "__main__":
    main()
