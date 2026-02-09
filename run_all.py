#!/usr/bin/env python3
"""Run scripts, optionally execute notebooks, then print submission summary. Run from project root."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_scripts(skip=False):
    for p in sorted((ROOT / "scripts").glob("*.py")):
        if skip:
            yield p.name, True, "skipped"
            continue
        r = subprocess.run([sys.executable, str(p)], cwd=ROOT, capture_output=True, text=True, timeout=120)
        msg = (r.stdout or r.stderr or "").strip() or ("ok" if r.returncode == 0 else f"exit {r.returncode}")
        yield p.name, r.returncode == 0, msg


def execute_notebooks():
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        yield "notebooks", False, "pip install nbconvert"
        return
    nb_dir = ROOT / "notebooks"
    for i in range(1, 9):
        nbs = list(nb_dir.glob(f"function_{i}_*.ipynb")) or (list(nb_dir.glob("funtion_2_*.ipynb")) if i == 2 else [])
        if not nbs:
            yield f"function_{i}", False, "not found"
            continue
        nb_path = nbs[0]
        try:
            nb = nbformat.read(nb_path, as_version=4)
            ExecutePreprocessor(timeout=300).preprocess(nb, {"metadata": {"path": str(ROOT)}})
            nb_path.write_text(nbformat.writes(nb))
            yield nb_path.name, True, "ok"
        except Exception as e:
            yield nb_path.name, False, str(e)[:60]


def submission_summary():
    sub_dir = ROOT / "data" / "submissions"
    problems_dir = ROOT / "data" / "problems"
    portal = {}
    for n in range(1, 9):
        txt = sub_dir / f"function_{n}" / "next_input_portal.txt"
        npy = sub_dir / f"function_{n}" / "next_input.npy"
        if txt.exists():
            portal[n] = txt.read_text().strip()
        elif npy.exists():
            import numpy as np
            portal[n] = "-".join(f"{x:.6f}" for x in np.load(npy).ravel())
        else:
            portal[n] = "(not generated)"
    # Warn if appended data is missing — notebooks then fall back to initial_data only and may suggest the same point again
    missing = []
    for n in range(1, 9):
        fn_dir = problems_dir / f"function_{n}"
        csv_path = fn_dir / "observations.csv"
        has_data = csv_path.exists()
        if not has_data:
            missing.append(n)
    if missing:
        print(
            "NOTE: data/problems/ has no appended data for function(s)",
            missing,
            "\n  → Notebooks will load initial_data only and may suggest the SAME point as last time.\n"
            "  → Run scripts/append_weekN_results.py after portal feedback, then re-run notebooks.",
        )
    return portal


def main():
    p = argparse.ArgumentParser(description="Run scripts, optionally notebooks; print submission summary.")
    p.add_argument("--execute-notebooks", action="store_true", help="Execute all 8 function notebooks")
    p.add_argument("--skip-scripts", action="store_true", help="Skip scripts/; only show summary")
    args = p.parse_args()

    print("run_all.py", ROOT, "\n")

    if (ROOT / "scripts").exists():
        print("Scripts:")
        for name, ok, msg in run_scripts(skip=args.skip_scripts):
            print(f"  [{'OK' if ok else 'FAIL'}] {name}: {msg}")
        print()

    if args.execute_notebooks:
        print("Notebooks:")
        for name, ok, msg in execute_notebooks():
            print(f"  [{'OK' if ok else 'FAIL'}] {name}: {msg}")
        print()

    portal = submission_summary()
    print("=" * 60)
    print("SUBMISSION — portal strings (copy-paste per function)")
    print("=" * 60)
    for n in range(1, 9):
        s = portal.get(n, "(missing)")
        print(f"  {n} | {s}")
    print("=" * 60)
    print("Files: data/submissions/function_N/next_input_portal.txt")
    print("       submission-template/")


if __name__ == "__main__":
    main()
