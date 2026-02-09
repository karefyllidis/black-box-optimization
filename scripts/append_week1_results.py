#!/usr/bin/env python3
"""
Append Week 1 portal results to local datasets (data/problems/function_N/).
Run from project root. Uses initial_data for existing points, then appends the new (x, y).
Idempotent: if the Week 1 point is already in the saved dataset, skips appending (safe to run multiple times).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT))
from src.utils.load_challenge_data import load_function_data

# Week 1 results from portal (input list, output scalar per function)
WEEK1 = {
    1: (np.array([0.548112, 0.542075]), 3.870398707923271e-10),
    2: (np.array([0.797468, 0.974684]), -0.008790069868146483),
    3: (np.array([0.161923, 0.658102, 0.997749]), -0.43976094541827415),
    4: (np.array([0.382363, 0.443127, 0.376190, 0.367341]), 0.0974032573661181),
    5: (np.array([0.392458, 0.754265, 0.918426, 0.950929]), 1396.7301709182432),
    6: (np.array([0.129036, 0.430316, 0.274705, 0.887752, 0.040586]), -0.970704177193032),
    7: (np.array([0.200565, 0.215359, 0.291867, 0.172757, 0.379366, 0.910834]), 1.517066575612848),
    8: (np.array([0.050323, 0.062907, 0.187347, 0.032471, 0.743353, 0.723330, 0.136056, 0.836079]), 9.8985683697244),
}

# Tolerance for "same point" check (avoid duplicate append due to float noise)
ATOL = 1e-9


def _already_appended(X_saved: np.ndarray, x_new: np.ndarray) -> bool:
    """True if x_new (1, d) is already a row in X_saved (n, d) within ATOL."""
    if X_saved is None or len(X_saved) == 0:
        return False
    return np.any(np.all(np.isclose(X_saved, x_new.ravel(), atol=ATOL), axis=1))


def main():
    problems_dir = ROOT / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(1, 9):
        x_new, y_new = WEEK1[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        prefix = "initial_" if fid == 1 else ""
        inputs_path = out_dir / f"{prefix}inputs.npy"
        outputs_path = out_dir / f"{prefix}outputs.npy"

        # If we already have appended data, check whether Week 1 point is in it
        if inputs_path.exists() and outputs_path.exists():
            X_cur = np.load(inputs_path)
            y_cur = np.load(outputs_path)
            if y_cur.ndim > 1:
                y_cur = y_cur.squeeze()
            if _already_appended(X_cur, x_new):
                print(f"Function {fid}: already appended (Week 1 point in dataset), skip. Points: {len(y_cur)}")
                continue
            # Week 1 not in file: append once and save
            assert x_new.shape[1] == X_cur.shape[1], f"Function {fid}: dimension mismatch"
            X_updated = np.vstack([X_cur, x_new])
            y_updated = np.append(y_cur, y_new)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(inputs_path, X_updated)
            np.save(outputs_path, y_updated)
            print(f"Function {fid}: {len(y_cur)} -> {len(y_updated)} points (appended Week 1), y range [{y_updated.min():.6g}, {y_updated.max():.6g}]")
            continue

        # No existing appended file: initial_data + Week 1
        X_init, y_init = load_function_data(fid)
        assert x_new.shape[1] == X_init.shape[1], f"Function {fid}: dimension mismatch"
        X_updated = np.vstack([X_init, x_new])
        y_updated = np.append(y_init, y_new)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(inputs_path, X_updated)
        np.save(outputs_path, y_updated)
        print(f"Function {fid}: {X_init.shape[0]} -> {X_updated.shape[0]} points, y range [{y_updated.min():.6g}, {y_updated.max():.6g}]")
    print("Done. data/problems/function_1..8 updated. Re-run notebooks to use appended points.")


if __name__ == "__main__":
    main()
