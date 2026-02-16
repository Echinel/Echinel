"""
==========================================================================
Value-Driven Churn Mitigation: Hierarchical Ensemble Learning (HEL)
Framework for Banking Sector Retention
==========================================================================

Main analysis pipeline orchestrator.

Research Questions:
  RQ1: Does the HEL framework outperform a monolithic baseline?
  RQ2: What differential feature importance patterns exist across segments?
  RQ3: What is the financial impact of the HEL framework?

Execution:
  python main.py

Outputs:
  - figures/   : All dissertation figures (PNG, 300 DPI)
  - outputs/   : CSV tables of results
  - Console    : Full analysis log
==========================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generation import generate_dataset
from src.feature_engineering import (
    engineer_features, encode_and_scale, segment_by_value
)
from src.models import (
    split_data, train_monolithic_baseline, train_hel_framework, predict_hel
)
from src.evaluation import (
    compute_metrics, bootstrap_confidence_intervals, mcnemar_test,
    expected_financial_loss, format_metrics_table
)
from src.visualisations import (
    plot_churn_distribution, plot_correlation_heatmap,
    plot_roc_curves, plot_precision_recall_curves,
    plot_confusion_matrices, plot_financial_comparison,
    plot_segment_comparison, plot_metrics_comparison_table,
    plot_hel_architecture
)
from src.shap_analysis import run_full_shap_analysis

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3
DATA_PATH = Path("data/Customer_Churn_Data_Large.xlsx")
FIGURES_DIR = Path("figures")
OUTPUTS_DIR = Path("outputs")

np.random.seed(RANDOM_STATE)


def load_or_generate_data():
    """Load real data if available, otherwise generate synthetic data."""
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        print("Loading real dataset from Excel...")
        try:
            xls = pd.ExcelFile(DATA_PATH)
            sheet_names = xls.sheet_names
            print(f"  Sheets found: {sheet_names}")

            # Try to load from multi-sheet format
            if len(sheet_names) >= 3:
                demographics_df = pd.read_excel(DATA_PATH,
                                                sheet_name=sheet_names[0])
                transactions_df = pd.read_excel(DATA_PATH,
                                                sheet_name=sheet_names[1])
                service_df = pd.read_excel(DATA_PATH,
                                           sheet_name=sheet_names[2])
                # Online activity may be separate or combined
                if len(sheet_names) >= 4:
                    online_df = pd.read_excel(DATA_PATH,
                                              sheet_name=sheet_names[3])
                else:
                    online_df = None
                if len(sheet_names) >= 5:
                    churn_df = pd.read_excel(DATA_PATH,
                                             sheet_name=sheet_names[4])
                else:
                    churn_df = None

                print(f"  Loaded {len(sheet_names)} sheets successfully.")
                return demographics_df, transactions_df, service_df, \
                    online_df, churn_df

            else:
                # Single sheet - try to load and parse
                df = pd.read_excel(DATA_PATH)
                print(f"  Single sheet with {len(df)} rows, "
                      f"{len(df.columns)} columns")
                print(f"  Columns: {list(df.columns)}")
                return _parse_single_sheet(df)

        except Exception as e:
            print(f"  Error loading Excel: {e}")
            print("  Falling back to synthetic data generation...")

    print("Generating synthetic dataset matching dissertation statistics...")
    return generate_dataset(seed=RANDOM_STATE, output_path=str(DATA_PATH))


def _parse_single_sheet(df):
    """Parse a single-sheet dataset into component DataFrames."""
    # Attempt to identify columns and split into logical tables
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}

    # Try to identify CustomerID
    id_col = None
    for c in df.columns:
        if "customer" in c.lower() and "id" in c.lower():
            id_col = c
            break
    if id_col is None:
        # Use first column or create one
        df.insert(0, "CustomerID", range(1, len(df) + 1))
        id_col = "CustomerID"

    # Identify churn column
    churn_col = None
    for c in df.columns:
        if "churn" in c.lower() or "exit" in c.lower():
            churn_col = c
            break

    if churn_col is None:
        raise ValueError("Cannot identify churn/target column in dataset")

    # Create churn df
    churn_df = df[[id_col, churn_col]].copy()
    churn_df.columns = ["CustomerID", "Churn"]

    # Demographics: age, gender, marital, income-like columns
    demo_cols = [id_col]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["age", "gender", "sex", "marit", "income",
                                  "geography", "country", "surname", "name"]):
            if c != churn_col:
                demo_cols.append(c)

    demographics_df = df[demo_cols].copy()
    demographics_df = demographics_df.rename(columns={id_col: "CustomerID"})

    # Transaction-like columns
    tx_cols = [id_col]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["transaction", "amount", "balance",
                                  "credit", "product", "card", "salary",
                                  "estimated"]):
            if c != churn_col:
                tx_cols.append(c)

    transactions_df = df[tx_cols].copy()
    transactions_df = transactions_df.rename(columns={id_col: "CustomerID"})

    # Service interaction columns
    svc_cols = [id_col]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["complaint", "service", "interaction",
                                  "resolution", "satisf"]):
            if c != churn_col:
                svc_cols.append(c)

    service_df = df[svc_cols].copy() if len(svc_cols) > 1 else None
    if service_df is not None:
        service_df = service_df.rename(columns={id_col: "CustomerID"})

    # Online/digital columns
    online_cols = [id_col]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["login", "online", "active", "tenure",
                                  "member", "digital"]):
            if c != churn_col:
                online_cols.append(c)

    online_df = df[online_cols].copy() if len(online_cols) > 1 else None
    if online_df is not None:
        online_df = online_df.rename(columns={id_col: "CustomerID"})

    # Keep the full dataframe as fallback
    # Store the original df for direct feature engineering
    demographics_df._full_df = df

    return demographics_df, transactions_df, service_df, online_df, churn_df


def engineer_features_flexible(demographics_df, transactions_df,
                                service_df, online_df, churn_df):
    """
    Flexible feature engineering that handles both multi-table and
    single-table data formats.
    """
    # Check if we have the full df attached (single-sheet case)
    if hasattr(demographics_df, "_full_df"):
        df = demographics_df._full_df.copy()
        print("  Using single-sheet format for feature engineering...")

        # Identify and rename key columns
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "customer" in cl and "id" in cl:
                col_map[c] = "CustomerID"
            elif "churn" in cl or "exit" in cl:
                col_map[c] = "Churn"

        df = df.rename(columns=col_map)

        # Drop non-feature columns
        drop_cols = []
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["surname", "name", "row", "id"]):
                if c not in ["CustomerID", "Churn"]:
                    drop_cols.append(c)

        df = df.drop(columns=drop_cols, errors="ignore")

        # If no Total_Monetary_Value, try to create one
        if "Total_Monetary_Value" not in df.columns:
            for c in df.columns:
                cl = c.lower()
                if "balance" in cl:
                    df["Total_Monetary_Value"] = df[c]
                    break
                elif "estimatedsalary" in cl or "estimated" in cl:
                    if "Total_Monetary_Value" not in df.columns:
                        df["Total_Monetary_Value"] = df[c]

        # Remove CustomerID for modelling
        if "CustomerID" in df.columns:
            df = df.drop(columns=["CustomerID"])

        return df

    # Multi-table format (generated data)
    return engineer_features(demographics_df, transactions_df,
                             service_df, online_df, churn_df)


def main():
    """Run the complete HEL framework analysis pipeline."""
    print("=" * 70)
    print("VALUE-DRIVEN CHURN MITIGATION: HEL FRAMEWORK ANALYSIS")
    print("=" * 70)

    # =========================================================
    # STEP 1: Data Loading
    # =========================================================
    print("\n[STEP 1] DATA LOADING")
    print("-" * 40)
    demographics_df, transactions_df, service_df, online_df, churn_df = \
        load_or_generate_data()

    # =========================================================
    # STEP 2: Feature Engineering
    # =========================================================
    print("\n[STEP 2] FEATURE ENGINEERING")
    print("-" * 40)
    df_features = engineer_features_flexible(
        demographics_df, transactions_df, service_df, online_df, churn_df
    )
    print(f"  Feature matrix shape: {df_features.shape}")
    print(f"  Columns: {list(df_features.columns)}")
    print(f"\n  Churn distribution:")
    print(f"  {df_features['Churn'].value_counts().to_dict()}")

    # Save raw features for reference
    df_features.to_csv(OUTPUTS_DIR / "engineered_features.csv", index=False)

    # =========================================================
    # STEP 3: EDA Visualisations
    # =========================================================
    print("\n[STEP 3] EXPLORATORY DATA ANALYSIS")
    print("-" * 40)

    # Figure 1: HEL Architecture
    print("  Generating HEL architecture diagram...")
    plot_hel_architecture()

    # Figure 2: Churn Distribution
    print("  Generating churn distribution plot...")
    plot_churn_distribution(df_features["Churn"])

    # Figure 3: Correlation Heatmap
    print("  Generating correlation heatmap...")
    plot_correlation_heatmap(df_features)

    # Print key correlations
    numeric_features = df_features.select_dtypes(include=[np.number])
    if "Churn" in numeric_features.columns:
        churn_corr = numeric_features.corr()["Churn"].drop("Churn").sort_values()
        print("\n  Key correlations with Churn:")
        for feat, corr_val in churn_corr.items():
            print(f"    {feat:35s} r = {corr_val:+.4f}")

    # =========================================================
    # STEP 4: Encode and Scale
    # =========================================================
    print("\n[STEP 4] ENCODING AND SCALING")
    print("-" * 40)
    X, y, feature_names, scaler = encode_and_scale(df_features)
    print(f"  Encoded feature matrix: {X.shape}")
    print(f"  Number of features: {len(feature_names)}")

    # =========================================================
    # STEP 5: Train-Test Split
    # =========================================================
    print("\n[STEP 5] STRATIFIED TRAIN-TEST SPLIT (70:30)")
    print("-" * 40)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Keep track of raw features for segmentation
    df_raw = df_features.copy()
    df_raw_train = df_raw.iloc[X_train.index]
    df_raw_test = df_raw.iloc[X_test.index]

    # =========================================================
    # STEP 6: Value-Based Segmentation (Stage 1 of HEL)
    # =========================================================
    print("\n[STEP 6] VALUE-BASED SEGMENTATION (HEL Stage 1)")
    print("-" * 40)

    # Identify the monetary value column
    tmv_col = "Total_Monetary_Value"
    if tmv_col not in df_raw.columns:
        # Find a suitable proxy
        for c in df_raw.columns:
            if "balance" in c.lower() or "monetary" in c.lower():
                tmv_col = c
                break
            elif "salary" in c.lower() or "estimated" in c.lower():
                tmv_col = c

    print(f"  Using '{tmv_col}' for value segmentation")
    median_value = df_raw[tmv_col].median()
    print(f"  Median value: £{median_value:,.2f}")

    # Create segment masks for train and test
    train_high_mask = df_raw_train[tmv_col] >= median_value
    train_low_mask = ~train_high_mask
    test_high_mask = df_raw_test[tmv_col] >= median_value
    test_low_mask = ~test_high_mask

    segments_train = {
        "high_value": {
            "X": X_train[train_high_mask],
            "y": y_train[train_high_mask],
        },
        "low_value": {
            "X": X_train[train_low_mask],
            "y": y_train[train_low_mask],
        }
    }

    test_segment_indices = {
        "high_value": np.where(test_high_mask.values)[0],
        "low_value": np.where(test_low_mask.values)[0],
    }

    for seg_name, seg_data in segments_train.items():
        n = len(seg_data["y"])
        nc = seg_data["y"].sum()
        print(f"  {seg_name} train: n={n}, churned={nc} ({100*nc/n:.1f}%)")

    for seg_name, idx in test_segment_indices.items():
        y_seg = y_test.iloc[idx]
        print(f"  {seg_name} test:  n={len(idx)}, "
              f"churned={y_seg.sum()} ({100*y_seg.mean():.1f}%)")

    # =========================================================
    # STEP 7: Train Monolithic Baseline
    # =========================================================
    print("\n[STEP 7] MONOLITHIC BASELINE TRAINING")
    print("-" * 40)
    mono_model, mono_cv_results = train_monolithic_baseline(
        X_train, y_train, random_state=RANDOM_STATE, verbose=0
    )

    # Monolithic predictions
    mono_pred = mono_model.predict(X_test)
    mono_proba = mono_model.predict_proba(X_test)[:, 1]

    # =========================================================
    # STEP 8: Train HEL Framework (Stage 2)
    # =========================================================
    print("\n[STEP 8] HEL FRAMEWORK TRAINING (Stage 2)")
    print("-" * 40)
    segment_models, segment_cv_results = train_hel_framework(
        X_train, y_train, segments_train,
        random_state=RANDOM_STATE, verbose=0
    )

    # HEL combined predictions
    hel_pred, hel_proba = predict_hel(
        segment_models, X_test, test_segment_indices
    )

    # =========================================================
    # STEP 9: Evaluation (RQ1)
    # =========================================================
    print("\n[STEP 9] MODEL EVALUATION (RQ1)")
    print("-" * 40)

    # Compute metrics
    mono_metrics, mono_cm = compute_metrics(
        y_test, mono_pred, mono_proba, "Monolithic Baseline"
    )
    hel_metrics, hel_cm = compute_metrics(
        y_test, hel_pred, hel_proba, "HEL Framework"
    )

    # Per-segment HEL metrics
    segment_metrics_results = {}
    for seg_name in ["high_value", "low_value"]:
        idx = test_segment_indices[seg_name]
        y_seg_true = y_test.iloc[idx]
        y_seg_pred = hel_pred[idx]
        y_seg_proba = hel_proba[idx]
        seg_metrics, seg_cm = compute_metrics(
            y_seg_true, y_seg_pred, y_seg_proba,
            f"HEL {seg_name.replace('_', ' ').title()}"
        )
        segment_metrics_results[seg_name] = seg_metrics

    # Bootstrap confidence intervals
    print("\n  Computing bootstrap confidence intervals (1,000 reps)...")
    mono_ci = bootstrap_confidence_intervals(
        y_test, mono_pred, mono_proba, random_state=RANDOM_STATE
    )
    hel_ci = bootstrap_confidence_intervals(
        y_test, hel_pred, hel_proba, random_state=RANDOM_STATE
    )

    # McNemar's test
    print("\n  Running McNemar's test...")
    chi2, p_value, cohen_h, b01, b10 = mcnemar_test(
        y_test.values, mono_pred, hel_pred
    )
    print(f"  McNemar chi2 = {chi2:.4f}, p-value = {p_value:.4f}")
    print(f"  Cohen's h = {cohen_h:.4f}")
    print(f"  Discordant pairs: A correct/B wrong = {b01}, "
          f"A wrong/B correct = {b10}")

    # Format and save results
    all_metrics = [mono_metrics, hel_metrics] + \
        list(segment_metrics_results.values())
    metrics_df = format_metrics_table(all_metrics, [mono_ci, hel_ci])

    print("\n" + "=" * 60)
    print("CLASSIFICATION PERFORMANCE SUMMARY")
    print("=" * 60)
    display_cols = ["Model", "Accuracy", "Precision", "Recall",
                    "F1-Score", "AUC-ROC", "AUC-PR", "Youden_J"]
    print(metrics_df[[c for c in display_cols
                      if c in metrics_df.columns]].to_string(index=False))

    # Print CIs
    print("\n95% Bootstrap Confidence Intervals:")
    for label, ci in [("Monolithic", mono_ci), ("HEL", hel_ci)]:
        print(f"\n  {label}:")
        for metric, vals in ci.items():
            print(f"    {metric:12s}: {vals['mean']:.4f} "
                  f"[{vals['lower']:.4f}, {vals['upper']:.4f}]")

    # Save to CSV
    metrics_df.to_csv(OUTPUTS_DIR / "classification_metrics.csv", index=False)

    # Statistical test results
    stat_results = pd.DataFrame([{
        "Test": "McNemar's Test",
        "Chi-Square": chi2,
        "p-value": p_value,
        "Cohen_h": cohen_h,
        "Significant_0.05": p_value < 0.05,
        "Discordant_A_correct_B_wrong": b01,
        "Discordant_A_wrong_B_correct": b10,
    }])
    stat_results.to_csv(OUTPUTS_DIR / "statistical_tests.csv", index=False)

    # =========================================================
    # STEP 10: Financial Impact Assessment (RQ3)
    # =========================================================
    print("\n[STEP 10] FINANCIAL IMPACT ASSESSMENT (RQ3)")
    print("-" * 40)

    # Get Total Monetary Value for test set
    tmv_test = df_raw_test[tmv_col].values

    mono_efl = expected_financial_loss(
        y_test, mono_pred, tmv_test
    )
    hel_efl = expected_financial_loss(
        y_test, hel_pred, tmv_test
    )

    efl_results = {
        "Monolithic Baseline": mono_efl,
        "HEL Framework": hel_efl
    }

    print("\n  Expected Financial Loss Comparison:")
    print(f"  {'Metric':<40s} {'Monolithic':>15s} {'HEL':>15s}")
    print(f"  {'-'*70}")
    for key in ["EFL_Total", "FN_Cost_Lost_CLV",
                "FP_Cost_Unnecessary_Intervention", "TP_Intervention_Cost",
                "N_False_Negatives", "N_False_Positives"]:
        mv = mono_efl[key]
        hv = hel_efl[key]
        if isinstance(mv, float):
            print(f"  {key:<40s} £{mv:>14,.2f} £{hv:>14,.2f}")
        else:
            print(f"  {key:<40s} {mv:>15d} {hv:>15d}")

    efl_reduction = (mono_efl["EFL_Total"] - hel_efl["EFL_Total"])
    efl_pct = 100 * efl_reduction / mono_efl["EFL_Total"] \
        if mono_efl["EFL_Total"] > 0 else 0
    print(f"\n  EFL Reduction: £{efl_reduction:,.2f} ({efl_pct:.1f}%)")

    # Save financial results
    efl_df = pd.DataFrame([
        {"Model": "Monolithic Baseline", **mono_efl},
        {"Model": "HEL Framework", **hel_efl},
    ])
    efl_df.to_csv(OUTPUTS_DIR / "financial_impact.csv", index=False)

    # =========================================================
    # STEP 11: Generate Visualisations
    # =========================================================
    print("\n[STEP 11] GENERATING VISUALISATIONS")
    print("-" * 40)

    # ROC curves
    results_for_plots = {
        "Monolithic Baseline": {
            "y_true": y_test, "y_pred": mono_pred,
            "y_proba": mono_proba, "metrics": mono_metrics
        },
        "HEL Framework": {
            "y_true": y_test, "y_pred": hel_pred,
            "y_proba": hel_proba, "metrics": hel_metrics
        },
    }

    # Add segment-level results for ROC
    for seg_name in ["high_value", "low_value"]:
        idx = test_segment_indices[seg_name]
        label = f"HEL {seg_name.replace('_', ' ').title()}"
        results_for_plots[label] = {
            "y_true": y_test.iloc[idx],
            "y_pred": hel_pred[idx],
            "y_proba": hel_proba[idx],
            "metrics": segment_metrics_results[seg_name]
        }

    print("  Generating ROC curves...")
    plot_roc_curves(results_for_plots)

    print("  Generating Precision-Recall curves...")
    plot_precision_recall_curves(results_for_plots)

    print("  Generating confusion matrices...")
    plot_confusion_matrices({
        "Monolithic Baseline": results_for_plots["Monolithic Baseline"],
        "HEL Framework": results_for_plots["HEL Framework"],
    })

    print("  Generating financial comparison...")
    plot_financial_comparison(efl_results)

    print("  Generating segment metrics comparison...")
    plot_segment_comparison(segment_metrics_results)

    print("  Generating metrics comparison table...")
    plot_metrics_comparison_table(metrics_df)

    # =========================================================
    # STEP 12: SHAP Analysis (RQ2)
    # =========================================================
    print("\n[STEP 12] SHAP INTERPRETABILITY ANALYSIS (RQ2)")
    print("-" * 40)

    X_test_segments_for_shap = {
        seg_name: {"X": X_test.iloc[idx]}
        for seg_name, idx in test_segment_indices.items()
    }

    shap_results = run_full_shap_analysis(
        mono_model, segment_models,
        X_test, X_test_segments_for_shap, feature_names
    )

    # Save SHAP importance tables
    for key, data in shap_results.items():
        data["importance"].to_csv(
            OUTPUTS_DIR / f"shap_importance_{key}.csv", index=False
        )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    print(f"\n  RQ1: Classification Performance")
    print(f"    Monolithic F1-Score: {mono_metrics['F1-Score']:.4f}")
    print(f"    HEL F1-Score:       {hel_metrics['F1-Score']:.4f}")
    f1_diff = hel_metrics['F1-Score'] - mono_metrics['F1-Score']
    print(f"    Improvement:        {f1_diff:+.4f} "
          f"({100*f1_diff/mono_metrics['F1-Score']:.1f}%)")
    print(f"    McNemar p-value:    {p_value:.4f} "
          f"({'Significant' if p_value < 0.05 else 'Not significant'} "
          f"at alpha=0.05)")

    print(f"\n  RQ2: Differential Feature Importance")
    for seg_name in ["high_value", "low_value"]:
        top3 = shap_results[seg_name]["importance"].head(3)
        features_str = ", ".join(top3["Feature"].values)
        print(f"    {seg_name.replace('_', ' ').title()} top 3: {features_str}")

    print(f"\n  RQ3: Financial Impact")
    print(f"    Monolithic EFL: £{mono_efl['EFL_Total']:,.2f}")
    print(f"    HEL EFL:        £{hel_efl['EFL_Total']:,.2f}")
    print(f"    Reduction:      £{efl_reduction:,.2f} ({efl_pct:.1f}%)")

    print(f"\n  Output files saved to:")
    print(f"    Figures: {FIGURES_DIR.absolute()}/")
    print(f"    Data:    {OUTPUTS_DIR.absolute()}/")

    # Save cross-validation results
    mono_cv_results.to_csv(OUTPUTS_DIR / "cv_results_monolithic.csv",
                           index=False)
    for seg_name, cv_res in segment_cv_results.items():
        cv_res.to_csv(OUTPUTS_DIR / f"cv_results_{seg_name}.csv",
                      index=False)

    # Save confidence interval results
    ci_rows = []
    for label, ci in [("Monolithic Baseline", mono_ci),
                      ("HEL Framework", hel_ci)]:
        for metric, vals in ci.items():
            ci_rows.append({
                "Model": label,
                "Metric": metric,
                "Mean": vals["mean"],
                "CI_Lower": vals["lower"],
                "CI_Upper": vals["upper"],
            })
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(OUTPUTS_DIR / "bootstrap_confidence_intervals.csv",
                 index=False)

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
