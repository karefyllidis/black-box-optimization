# Phase A: Training Materials

Training materials and practice work for the black-box optimization capstone project.

Everything regarding the Training Phase of the project (i.e., week 3-12) is contained here

## Structure

- **docs/**: Course materials, video transcripts, and documentation
- **notebooks/**: Training notebooks and exercises
- **compare_wine_classification_methods.py**: Script to compare Logistic Regression, KNN, and Decision Tree models

## Wine Classification Comparison

To compare all three wine classification methods, run:

```bash
python compare_wine_classification_methods.py
```

This script will:
- Train Logistic Regression, KNN, and Decision Tree models
- Evaluate performance on train, validation, and test sets
- Generate comparison visualizations
- Create detailed classification reports
- Analyze overfitting patterns
- Save results as `wine_classification_comparison.png`

## Notes

- Practice work and experiments go here
- Production code belongs in `src/` at root level
- Training results can be stored in `data/results/training/`
