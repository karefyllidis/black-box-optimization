#!/usr/bin/env python3
"""
Append Week 5 portal results to local datasets (data/problems/function_N/).
Under data/ we use only CSV: observations.csv. No .npy in data/problems/.
Run from project root. Idempotent: skips if Week 5 point already in dataset.
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

# Week 5 results from portal (input list, output scalar per function)
WEEK5 = {
    1: (np.array([0.874328, 0.796958]), -3.384836234095372e-62),
    2: (np.array([0.678849, 0.870308]), 0.5421199215934085),
    3: (np.array([0.694620, 0.572056, 0.631664]), -0.08145286754352399),
    4: (np.array([0.324369, 0.377106, 0.478210, 0.488917]), -2.6229519964306713),
    5: (np.array([0.538420, 0.997040, 0.990356, 0.999476]), 4574.257628289265),
    6: (np.array([0.390913, 0.340103, 0.597945, 0.631205, 0.020811]), -0.361775394296891),
    7: (np.array([0.007864, 0.284203, 0.078178, 0.068973, 0.254110, 0.889294]), 0.8413543326796774),
    8: (np.array([0.023603, 0.520401, 0.001388, 0.160869, 0.832212, 0.054203, 0.006707, 0.195293]), 9.5110319062241),
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
        x_new, y_new = WEEK5[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = _load_current(out_dir, fid)
        if current is not None:
            X_cur, y_cur = current
            if _already_appended(X_cur, x_new):
                print(f"Function {fid}: already appended (Week 5 in dataset), skip. Points: {len(y_cur)}")
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
    print("Done. data/problems/function_1..8 updated (CSV). Re-run notebooks for Week 6.")


if __name__ == "__main__":
    main()
