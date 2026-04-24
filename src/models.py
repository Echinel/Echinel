"""
Modelling Module
================
Implements:
1. Monolithic XGBoost baseline (Section 3.4)
2. HEL Framework: segment-specific XGBoost classifiers with SMOTE (Section 3.4)

Hyperparameter search space:
- max_depth: [3, 5, 7, 10]
- learning_rate: [0.01, 0.05, 0.1, 0.3]
- n_estimators: [100, 200, 300, 500]
- reg_lambda (L2): [0.1, 0.5, 1.0]

SMOTE k_neighbors=5 (Chawla et al. default, confirmed by Elreedy & Atiya).

Grid-search scoring uses AUC-ROC. Threshold-free ranking quality is more
reliable than F1 on weakly-separable, imbalanced problems (Saito &
Rehmsmeier, 2015). Final classification thresholds are selected
post-hoc using Youden's J on cross-validated training predictions,
avoiding test-set leakage.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict
)
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# Hyperparameter grid (Section 3.4)
PARAM_GRID = {
    "classifier__max_depth": [3, 5, 7, 10],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.3],
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__reg_lambda": [0.1, 0.5, 1.0],
}

# Reduced grid for faster execution (subset of the full grid)
PARAM_GRID_REDUCED = {
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.1, 0.3],
    "classifier__n_estimators": [100, 200, 300],
    "classifier__reg_lambda": [0.1, 1.0],
}


def build_pipeline(random_state=42):
    """Create SMOTE + XGBoost pipeline."""
    pipeline = ImbPipeline([
        ("smote", SMOTE(k_neighbors=5, random_state=random_state)),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline


def grid_search_cv(X_train, y_train, param_grid=None, cv_folds=5,
                   scoring="roc_auc", random_state=42, verbose=1):
    """
    5-fold stratified cross-validation grid search (Section 3.4).

    Scoring defaults to roc_auc: threshold-independent ranking quality
    is more reliable than F1 for weakly-separable, imbalanced problems.

    Returns:
        best_pipeline: fitted pipeline with best hyperparameters
        cv_results: DataFrame of cross-validation results
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    pipeline = build_pipeline(random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    print(f"  Best {scoring}: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")

    return grid_search.best_estimator_, pd.DataFrame(grid_search.cv_results_)


def find_optimal_threshold(model, X_train, y_train, cv_folds=5,
                           random_state=42):
    """
    Find optimal classification threshold via Youden's J on
    cross-validated training predictions. This avoids data leakage
    from selecting a threshold on the test set.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)
    cv_proba = cross_val_predict(
        model, X_train, y_train, cv=cv,
        method="predict_proba", n_jobs=-1
    )[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, cv_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def train_monolithic_baseline(X_train, y_train, random_state=42, verbose=1):
    """
    Train monolithic XGBoost baseline model (single model on all data).

    Returns:
        model: fitted pipeline
        cv_results: cross-validation results
        threshold: Youden's J optimal threshold selected from CV
    """
    print("=" * 60)
    print("MONOLITHIC BASELINE MODEL")
    print("=" * 60)
    print(f"Training set: n={len(y_train)}, "
          f"churned={y_train.sum()} ({100*y_train.mean():.1f}%)")

    model, cv_results = grid_search_cv(
        X_train, y_train, random_state=random_state, verbose=verbose
    )

    threshold, j_stat = find_optimal_threshold(
        model, X_train, y_train, random_state=random_state
    )
    print(f"  CV Youden's J threshold: {threshold:.4f} (J={j_stat:.4f})")

    return model, cv_results, threshold


def train_hel_framework(X_train, y_train, segments_train, random_state=42,
                        verbose=1):
    """
    Train HEL Framework: segment-specific XGBoost classifiers (Section 3.4).

    Stage 1: Value-based segmentation (already done, passed as segments_train)
    Stage 2: Train separate XGBoost on each segment with SMOTE

    Returns:
        segment_models: dict of {segment_name: fitted_pipeline}
        segment_cv_results: dict of {segment_name: cv_results}
        segment_thresholds: dict of {segment_name: Youden's J threshold}
    """
    print("=" * 60)
    print("HIERARCHICAL ENSEMBLE LEARNING (HEL) FRAMEWORK")
    print("=" * 60)

    segment_models = {}
    segment_cv_results = {}
    segment_thresholds = {}

    for seg_name, seg_data in segments_train.items():
        X_seg = seg_data["X"]
        y_seg = seg_data["y"]

        print(f"\n--- Segment: {seg_name} ---")
        print(f"  Training set: n={len(y_seg)}, "
              f"churned={y_seg.sum()} ({100*y_seg.mean():.1f}%)")

        model, cv_results = grid_search_cv(
            X_seg, y_seg, random_state=random_state, verbose=verbose
        )

        threshold, j_stat = find_optimal_threshold(
            model, X_seg, y_seg, random_state=random_state
        )
        print(f"  CV Youden's J threshold: {threshold:.4f} (J={j_stat:.4f})")

        segment_models[seg_name] = model
        segment_cv_results[seg_name] = cv_results
        segment_thresholds[seg_name] = threshold

    return segment_models, segment_cv_results, segment_thresholds


def predict_with_threshold(model, X, threshold=0.5):
    """Predict with a custom classification threshold."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba


def predict_hel(segment_models, X_test, test_segment_indices,
                segment_thresholds=None):
    """
    Generate HEL predictions by routing test samples to appropriate
    segment-specific models. Uses segment-specific thresholds if
    provided, otherwise default 0.5.
    """
    y_pred = np.zeros(len(X_test), dtype=int)
    y_proba = np.zeros(len(X_test))

    for seg_name, model in segment_models.items():
        idx = test_segment_indices[seg_name]
        if len(idx) == 0:
            continue
        X_seg = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
        proba = model.predict_proba(X_seg)[:, 1]
        if segment_thresholds is not None:
            thr = segment_thresholds.get(seg_name, 0.5)
        else:
            thr = 0.5
        y_pred[idx] = (proba >= thr).astype(int)
        y_proba[idx] = proba

    return y_pred, y_proba


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Stratified 70:30 train-test split (Section 3.1).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: n={len(y_train)}, churn={y_train.sum()} "
          f"({100*y_train.mean():.1f}%)")
    print(f"Test:  n={len(y_test)}, churn={y_test.sum()} "
          f"({100*y_test.mean():.1f}%)")

    return X_train, X_test, y_train, y_test
