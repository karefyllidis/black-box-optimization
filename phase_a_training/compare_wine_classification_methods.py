"""
Wine Classification Methods Comparison

This script compares three classification methods for wine classification:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree

It trains all models, evaluates their performance, and creates comparison visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("WINE CLASSIFICATION METHODS COMPARISON")
print("="*80)
print()

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

print("Step 1: Loading and preparing data...")
print("-" * 80)

# Load wine dataset
data = load_wine()
X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(data.target_names)}")
print(f"Classes: {data.target_names}")
print()

# Split data into train (70%), validation (15%) and test (15%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)

print(f"Training set: {X_train.shape[0]} samples ({100*X_train.shape[0]/X.shape[0]:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({100*X_val.shape[0]/X.shape[0]:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({100*X_test.shape[0]/X.shape[0]:.1f}%)")
print()

# Scale features for KNN and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Data preparation complete!")
print()

# ============================================================================
# 2. MODEL TRAINING
# ============================================================================

print("Step 2: Training models...")
print("-" * 80)

models = {}
training_times = {}
results = {}

# ----------------------------------------------------------------------------
# 2.1 Logistic Regression
# ----------------------------------------------------------------------------

print("Training Logistic Regression...")
start_time = time.time()

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

training_times['Logistic Regression'] = time.time() - start_time
models['Logistic Regression'] = lr_model

# Predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_val_pred = lr_model.predict(X_val_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

results['Logistic Regression'] = {
    'train_acc': accuracy_score(y_train, lr_train_pred),
    'val_acc': accuracy_score(y_val, lr_val_pred),
    'test_acc': accuracy_score(y_test, lr_test_pred),
    'test_precision': precision_score(y_test, lr_test_pred, average='weighted'),
    'test_recall': recall_score(y_test, lr_test_pred, average='weighted'),
    'test_f1': f1_score(y_test, lr_test_pred, average='weighted'),
    'predictions': lr_test_pred
}

print(f"  Training time: {training_times['Logistic Regression']:.4f} seconds")
print(f"  Validation accuracy: {results['Logistic Regression']['val_acc']:.4f}")
print()

# ----------------------------------------------------------------------------
# 2.2 K-Nearest Neighbors (KNN)
# ----------------------------------------------------------------------------

print("Finding optimal k for KNN...")
k_range = range(1, 21)
val_scores = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    val_scores.append(knn_temp.score(X_val_scaled, y_val))

best_k = k_range[np.argmax(val_scores)]
print(f"  Best k: {best_k}")

print("Training KNN...")
start_time = time.time()

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

training_times['KNN'] = time.time() - start_time
models['KNN'] = knn_model

# Predictions
knn_train_pred = knn_model.predict(X_train_scaled)
knn_val_pred = knn_model.predict(X_val_scaled)
knn_test_pred = knn_model.predict(X_test_scaled)

results['KNN'] = {
    'train_acc': accuracy_score(y_train, knn_train_pred),
    'val_acc': accuracy_score(y_val, knn_val_pred),
    'test_acc': accuracy_score(y_test, knn_test_pred),
    'test_precision': precision_score(y_test, knn_test_pred, average='weighted'),
    'test_recall': recall_score(y_test, knn_test_pred, average='weighted'),
    'test_f1': f1_score(y_test, knn_test_pred, average='weighted'),
    'predictions': knn_test_pred,
    'best_k': best_k
}

print(f"  Training time: {training_times['KNN']:.4f} seconds")
print(f"  Validation accuracy: {results['KNN']['val_acc']:.4f}")
print()

# ----------------------------------------------------------------------------
# 2.3 Decision Tree
# ----------------------------------------------------------------------------

print("Tuning Decision Tree hyperparameters...")
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1]
}

dt_base = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    dt_base, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)
print(f"  Best parameters: {grid_search.best_params_}")

print("Training Decision Tree...")
start_time = time.time()

dt_model = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
dt_model.fit(X_train, y_train)

training_times['Decision Tree'] = time.time() - start_time
models['Decision Tree'] = dt_model

# Predictions
dt_train_pred = dt_model.predict(X_train)
dt_val_pred = dt_model.predict(X_val)
dt_test_pred = dt_model.predict(X_test)

results['Decision Tree'] = {
    'train_acc': accuracy_score(y_train, dt_train_pred),
    'val_acc': accuracy_score(y_val, dt_val_pred),
    'test_acc': accuracy_score(y_test, dt_test_pred),
    'test_precision': precision_score(y_test, dt_test_pred, average='weighted'),
    'test_recall': recall_score(y_test, dt_test_pred, average='weighted'),
    'test_f1': f1_score(y_test, dt_test_pred, average='weighted'),
    'predictions': dt_test_pred,
    'best_params': grid_search.best_params_
}

print(f"  Training time: {training_times['Decision Tree']:.4f} seconds")
print(f"  Validation accuracy: {results['Decision Tree']['val_acc']:.4f}")
print()

# ============================================================================
# 3. RESULTS SUMMARY
# ============================================================================

print("Step 3: Results Summary")
print("=" * 80)
print()

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_acc'] for m in results.keys()],
    'Val Accuracy': [results[m]['val_acc'] for m in results.keys()],
    'Test Accuracy': [results[m]['test_acc'] for m in results.keys()],
    'Test Precision': [results[m]['test_precision'] for m in results.keys()],
    'Test Recall': [results[m]['test_recall'] for m in results.keys()],
    'Test F1-Score': [results[m]['test_f1'] for m in results.keys()],
    'Training Time (s)': [training_times[m] for m in results.keys()]
})

print(comparison_df.to_string(index=False))
print()

# Find best model
best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
print(f"Best Model (by Test Accuracy): {best_model}")
print(f"  Test Accuracy: {comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Test Accuracy']:.4f}")
print()

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

print("Step 4: Creating visualizations...")
print("-" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# ----------------------------------------------------------------------------
# 4.1 Accuracy Comparison
# ----------------------------------------------------------------------------

ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(comparison_df))
width = 0.25

ax1.bar(x - width, comparison_df['Train Accuracy'], width, label='Train', alpha=0.8)
ax1.bar(x, comparison_df['Val Accuracy'], width, label='Validation', alpha=0.8)
ax1.bar(x + width, comparison_df['Test Accuracy'], width, label='Test', alpha=0.8)

ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy Comparison Across Datasets', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.1])

# Add value labels
for i, model in enumerate(comparison_df['Model']):
    ax1.text(i - width, comparison_df.loc[i, 'Train Accuracy'] + 0.02, 
             f"{comparison_df.loc[i, 'Train Accuracy']:.3f}", 
             ha='center', va='bottom', fontsize=9)
    ax1.text(i, comparison_df.loc[i, 'Val Accuracy'] + 0.02, 
             f"{comparison_df.loc[i, 'Val Accuracy']:.3f}", 
             ha='center', va='bottom', fontsize=9)
    ax1.text(i + width, comparison_df.loc[i, 'Test Accuracy'] + 0.02, 
             f"{comparison_df.loc[i, 'Test Accuracy']:.3f}", 
             ha='center', va='bottom', fontsize=9)

# ----------------------------------------------------------------------------
# 4.2 Test Set Metrics Comparison
# ----------------------------------------------------------------------------

ax2 = plt.subplot(2, 3, 2)
metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score']
x_pos = np.arange(len(comparison_df))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, model in enumerate(comparison_df['Model']):
    values = [comparison_df.loc[i, metric] for metric in metrics]
    ax2.plot(metrics, values, marker='o', linewidth=2, markersize=8, 
             label=model, color=colors[i])

ax2.set_xlabel('Metric', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Test Set Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticklabels(metrics, rotation=15, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.85, 1.05])

# ----------------------------------------------------------------------------
# 4.3 Training Time Comparison
# ----------------------------------------------------------------------------

ax3 = plt.subplot(2, 3, 3)
bars = ax3.bar(comparison_df['Model'], comparison_df['Training Time (s)'], 
               color=colors, alpha=0.8)
ax3.set_xlabel('Model', fontsize=12)
ax3.set_ylabel('Training Time (seconds)', fontsize=12)
ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax3.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}s', ha='center', va='bottom', fontsize=10)

# ----------------------------------------------------------------------------
# 4.4 Confusion Matrices
# ----------------------------------------------------------------------------

for idx, (model_name, model_results) in enumerate(results.items()):
    ax = plt.subplot(2, 3, 4 + idx)
    
    cm = confusion_matrix(y_test, model_results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=data.target_names,
                yticklabels=data.target_names)
    
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
print()

# ============================================================================
# 5. DETAILED CLASSIFICATION REPORTS
# ============================================================================

print("Step 5: Detailed Classification Reports")
print("=" * 80)
print()

for model_name, model_results in results.items():
    print(f"\n{model_name}")
    print("-" * 80)
    print(classification_report(y_test, model_results['predictions'], 
                                target_names=data.target_names))
    print()

# ============================================================================
# 6. OVERFITTING ANALYSIS
# ============================================================================

print("Step 6: Overfitting Analysis")
print("=" * 80)
print()

overfitting_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train-Val Gap': [results[m]['train_acc'] - results[m]['val_acc'] 
                      for m in results.keys()],
    'Val-Test Gap': [abs(results[m]['val_acc'] - results[m]['test_acc']) 
                     for m in results.keys()]
})

print(overfitting_df.to_string(index=False))
print()

print("Interpretation:")
print("- Large Train-Val Gap (>0.1): Indicates overfitting to training data")
print("- Large Val-Test Gap (>0.05): Indicates model instability or validation set overfitting")
print()

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()

print("Model Rankings by Test Accuracy:")
ranked = comparison_df.sort_values('Test Accuracy', ascending=False)
for i, (idx, row) in enumerate(ranked.iterrows(), 1):
    print(f"{i}. {row['Model']}: {row['Test Accuracy']:.4f}")

print()
print("Key Insights:")
print(f"- Fastest Training: {comparison_df.loc[comparison_df['Training Time (s)'].idxmin(), 'Model']}")
print(f"- Best Test Accuracy: {best_model}")
print(f"- Most Stable (smallest Val-Test gap): {overfitting_df.loc[overfitting_df['Val-Test Gap'].idxmin(), 'Model']}")
print()

print("=" * 80)
print("Comparison complete! Check 'wine_classification_comparison.png' for visualizations.")
print("=" * 80)

