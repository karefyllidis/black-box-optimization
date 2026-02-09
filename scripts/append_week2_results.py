#!/usr/bin/env python3
"""
Append Week 2 portal results to local datasets (data/problems/function_N/).
Under data/ we use only CSV: observations.csv. No .npy in data/problems/.
Run from project root. Idempotent: skips if Week 2 point already in dataset.
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

# Week 2 results from portal (input list, output scalar per function)
WEEK2 = {
    1: (np.array([0.002223, 0.994219]), 0.0),
    2: (np.array([0.755740, 0.832334]), 0.39274592699035027),
    3: (np.array([0.999764, 0.052131, 0.975598]), -0.4122344671527349),
    4: (np.array([0.374317, 0.373678, 0.284633, 0.377770]), -1.5987632558201272),
    5: (np.array([0.392458, 0.754265, 0.918426, 0.950929]), 1396.7301709182432),
    6: (np.array([0.129036, 0.430316, 0.274705, 0.887752, 0.040586]), -0.9522758924426504),
    7: (np.array([0.200565, 0.215359, 0.291867, 0.172757, 0.379366, 0.910834]), 1.517066575612848),
    8: (np.array([0.050323, 0.062907, 0.187347, 0.032471, 0.743353, 0.723330, 0.136056, 0.836079]), 9.8985683697244),
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
        x_new, y_new = WEEK2[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = _load_current(out_dir, fid)
        if current is not None:
            X_cur, y_cur = current
            if _already_appended(X_cur, x_new):
                print(f"Function {fid}: already appended (Week 2 in dataset), skip. Points: {len(y_cur)}")
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
    print("Done. data/problems/function_1..8 updated (CSV). Re-run notebooks for Week 3.")


if __name__ == "__main__":
    main()
