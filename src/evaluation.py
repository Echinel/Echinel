"""
Evaluation Module
=================
Implements evaluation metrics and statistical testing (Section 3.5)
and Financial Impact Assessment (Section 3.6).

Metrics:
- Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR
- Threshold: Youden's J-statistic
- Statistical: McNemar's test, Cohen's h
- Financial: Expected Financial Loss (EFL)
- Confidence Intervals: 1,000 bootstrap repetitions at 95% CI
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
from scipy import stats


def youdens_j_threshold(y_true, y_proba):
    """
    Find optimal classification threshold using Youden's J-statistic
    (J = Sensitivity + Specificity - 1) (Section 3.5).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr  # equivalent to sensitivity + specificity - 1
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_j = j_scores[best_idx]
    return best_threshold, best_j


def compute_metrics(y_true, y_pred, y_proba, label="Model"):
    """
    Compute comprehensive classification metrics (Section 3.5).
    All metrics computed on the minority (churn) class.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Model": label,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "AUC-PR": average_precision_score(y_true, y_proba),
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn,
        "False Negatives": fn,
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

    # Youden's J
    threshold, j_stat = youdens_j_threshold(y_true, y_proba)
    metrics["Youden_J"] = j_stat
    metrics["Optimal_Threshold"] = threshold

    return metrics, cm


def bootstrap_confidence_intervals(y_true, y_pred, y_proba,
                                   n_bootstrap=1000, ci=0.95,
                                   random_state=42):
    """
    95% confidence intervals via 1,000 bootstrap repetitions (Section 3.5).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    boot_metrics = {
        "Accuracy": [], "Precision": [], "Recall": [],
        "F1-Score": [], "AUC-ROC": [], "AUC-PR": []
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        y_t = y_true.values[idx] if hasattr(y_true, "values") else y_true[idx]
        y_p = y_pred[idx]
        y_pr = y_proba[idx]

        # Skip bootstrap samples with single class
        if len(np.unique(y_t)) < 2:
            continue

        boot_metrics["Accuracy"].append(accuracy_score(y_t, y_p))
        boot_metrics["Precision"].append(
            precision_score(y_t, y_p, zero_division=0))
        boot_metrics["Recall"].append(
            recall_score(y_t, y_p, zero_division=0))
        boot_metrics["F1-Score"].append(
            f1_score(y_t, y_p, zero_division=0))
        boot_metrics["AUC-ROC"].append(roc_auc_score(y_t, y_pr))
        boot_metrics["AUC-PR"].append(average_precision_score(y_t, y_pr))

    alpha = (1 - ci) / 2
    ci_results = {}
    for metric_name, values in boot_metrics.items():
        values = np.array(values)
        ci_results[metric_name] = {
            "mean": np.mean(values),
            "lower": np.percentile(values, 100 * alpha),
            "upper": np.percentile(values, 100 * (1 - alpha)),
        }

    return ci_results


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for comparing paired classifier predictions (Section 3.5).

    Tests whether the two models have the same error rate.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency: b_correct_a_wrong vs a_correct_b_wrong
    b01 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    b10 = np.sum(~correct_a & correct_b)  # A wrong, B correct

    # McNemar's test (with continuity correction)
    if b01 + b10 == 0:
        return 0, 1.0, 0, 0, 0

    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Cohen's h effect size for proportions
    p1 = np.mean(y_pred_a == y_true)
    p2 = np.mean(y_pred_b == y_true)
    cohen_h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    return chi2, p_value, cohen_h, b01, b10


def expected_financial_loss(y_true, y_pred, total_monetary_values,
                            retention_years=3, discount_rate=0.05,
                            intervention_cost=50.0):
    """
    Expected Financial Loss (EFL) computation (Section 3.6).

    EFL = Σ(FN × CLV_i) + Σ(FP × Intervention_Cost)

    CLV is calculated as Total Monetary Value × retention period
    with 5% discount rate (Gupta, Lehmann & Stuart methodology).
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    tmv = np.array(total_monetary_values)

    # Calculate CLV: TMV × present value annuity factor
    # PV annuity factor = Σ(1/(1+r)^t) for t=1..T
    pv_factor = sum(1 / (1 + discount_rate) ** t
                    for t in range(1, retention_years + 1))
    clv = tmv * pv_factor

    # False Negatives: actual churners predicted as retained
    fn_mask = (y_true_arr == 1) & (y_pred_arr == 0)
    fn_cost = np.sum(clv[fn_mask])

    # False Positives: retained customers predicted as churners
    fp_mask = (y_true_arr == 0) & (y_pred_arr == 1)
    fp_cost = np.sum(fp_mask) * intervention_cost

    # True Positives: churners correctly identified (intervention cost applied)
    tp_mask = (y_true_arr == 1) & (y_pred_arr == 1)
    tp_intervention_cost = np.sum(tp_mask) * intervention_cost

    # Total EFL: EFL = Σ(FN × CLV_i) + Σ(FP × Intervention_Cost)
    # Per Section 3.6, only FN and FP costs are included
    efl = fn_cost + fp_cost

    # Detailed breakdown
    results = {
        "EFL_Total": efl,
        "FN_Cost_Lost_CLV": fn_cost,
        "FP_Cost_Unnecessary_Intervention": fp_cost,
        "TP_Intervention_Cost": tp_intervention_cost,
        "N_False_Negatives": int(fn_mask.sum()),
        "N_False_Positives": int(fp_mask.sum()),
        "N_True_Positives": int(tp_mask.sum()),
        "Mean_CLV_Missed_Churners": float(np.mean(clv[fn_mask]))
        if fn_mask.sum() > 0 else 0,
        "PV_Annuity_Factor": pv_factor,
    }

    return results


def format_metrics_table(metrics_list, ci_list=None):
    """Format metrics as a presentation-ready DataFrame."""
    df = pd.DataFrame(metrics_list)

    if ci_list:
        for i, ci in enumerate(ci_list):
            for metric_name, ci_vals in ci.items():
                col = f"{metric_name}_CI"
                df.loc[i, col] = (
                    f"[{ci_vals['lower']:.3f}, {ci_vals['upper']:.3f}]"
                )

    return df
