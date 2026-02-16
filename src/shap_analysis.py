"""
SHAP Interpretability Module
============================
Applies SHAP analysis to identify differential feature importance
patterns across customer value segments (Section 4.2, RQ2).

Generates:
- Global SHAP summary plots
- Per-segment SHAP summary plots
- Feature importance comparison across segments
- SHAP dependence plots for top features
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

FIGURES_DIR = Path("figures")


def extract_xgb_model(pipeline):
    """Extract the XGBClassifier from an imblearn pipeline."""
    return pipeline.named_steps["classifier"]


def compute_shap_values(model, X, feature_names=None):
    """
    Compute SHAP values using TreeExplainer.

    Returns:
        shap_values: SHAP values array
        explainer: fitted SHAP explainer
    """
    xgb_model = extract_xgb_model(model)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    return shap_values, explainer


def get_feature_importance_df(shap_values, feature_names):
    """Get mean absolute SHAP values as feature importance."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    return importance_df


def plot_shap_summary_global(shap_values, X, feature_names,
                             name="fig_shap_global_summary"):
    """Global SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    X_display = X.copy()
    if hasattr(X_display, "columns"):
        X_display.columns = feature_names

    shap.summary_plot(shap_values, X_display, plot_type="dot",
                      show=False, max_display=15)
    plt.title("Global SHAP Feature Importance (Monolithic Baseline)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES_DIR / name}.png")


def plot_shap_summary_segments(segment_shap_data,
                               name_prefix="fig_shap_segment"):
    """Per-segment SHAP summary plots."""
    for seg_name, data in segment_shap_data.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        X_display = data["X"].copy()
        if hasattr(X_display, "columns"):
            X_display.columns = data["feature_names"]

        shap.summary_plot(data["shap_values"], X_display, plot_type="dot",
                          show=False, max_display=15)
        seg_title = seg_name.replace("_", " ").title()
        plt.title(f"SHAP Feature Importance: {seg_title} Segment",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{name_prefix}_{seg_name}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIGURES_DIR / name_prefix}_{seg_name}.png")


def plot_shap_bar_comparison(segment_importance_dfs, top_n=15,
                             name="fig_shap_segment_comparison"):
    """
    Side-by-side bar comparison of top SHAP feature importances
    across segments (key figure for RQ2).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    colors = {"high_value": "#FF9800", "low_value": "#4CAF50"}

    for ax, (seg_name, imp_df) in zip(axes, segment_importance_dfs.items()):
        top = imp_df.head(top_n).sort_values("Mean_Abs_SHAP")
        ax.barh(top["Feature"], top["Mean_Abs_SHAP"],
                color=colors.get(seg_name, "gray"),
                edgecolor="black", linewidth=0.3)
        seg_title = seg_name.replace("_", " ").title()
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"{seg_title} Segment", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Differential Feature Importance Across "
                 "Customer Value Segments\n(SHAP Analysis - RQ2)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES_DIR / name}.png")

    return fig


def plot_shap_dependence_top_features(shap_values, X, feature_names,
                                      top_n=4,
                                      name_prefix="fig_shap_dependence"):
    """SHAP dependence plots for top features."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    X_display = X.copy()
    if hasattr(X_display, "columns"):
        X_display.columns = feature_names

    for i, idx in enumerate(top_indices):
        ax = axes[i]
        shap.dependence_plot(
            idx, shap_values,
            X_display if isinstance(X_display, np.ndarray)
            else X_display.values,
            feature_names=feature_names,
            ax=ax, show=False
        )
        ax.set_title(f"{feature_names[idx]}", fontweight="bold")

    fig.suptitle("SHAP Dependence Plots: Top 4 Features",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name_prefix}.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIGURES_DIR / name_prefix}.png")


def run_full_shap_analysis(monolithic_model, segment_models,
                           X_test, X_test_segments, feature_names):
    """
    Run complete SHAP analysis pipeline.

    Returns:
        results: dict with all SHAP analysis outputs
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    results = {}

    print("\n" + "=" * 60)
    print("SHAP INTERPRETABILITY ANALYSIS (RQ2)")
    print("=" * 60)

    # 1. Global SHAP (monolithic)
    print("\n[1/4] Computing global SHAP values (monolithic)...")
    mono_shap, mono_explainer = compute_shap_values(
        monolithic_model, X_test, feature_names
    )
    plot_shap_summary_global(mono_shap, X_test, feature_names)
    results["monolithic"] = {
        "shap_values": mono_shap,
        "importance": get_feature_importance_df(mono_shap, feature_names)
    }

    # 2. Segment-specific SHAP
    print("\n[2/4] Computing segment-specific SHAP values...")
    segment_shap_data = {}
    segment_importance_dfs = {}

    for seg_name, seg_info in X_test_segments.items():
        X_seg = seg_info["X"]
        model = segment_models[seg_name]

        shap_vals, _ = compute_shap_values(model, X_seg, feature_names)
        imp_df = get_feature_importance_df(shap_vals, feature_names)

        segment_shap_data[seg_name] = {
            "shap_values": shap_vals,
            "X": X_seg,
            "feature_names": feature_names
        }
        segment_importance_dfs[seg_name] = imp_df
        results[seg_name] = {
            "shap_values": shap_vals,
            "importance": imp_df
        }

    # 3. Per-segment summary plots
    print("\n[3/4] Generating per-segment SHAP plots...")
    plot_shap_summary_segments(segment_shap_data)

    # 4. Cross-segment comparison
    print("\n[4/4] Generating cross-segment comparison...")
    plot_shap_bar_comparison(segment_importance_dfs)

    # Dependence plots for monolithic
    plot_shap_dependence_top_features(mono_shap, X_test, feature_names)

    # Print top features per segment
    print("\n--- Top 10 Features by Segment ---")
    for seg_name, imp_df in segment_importance_dfs.items():
        print(f"\n{seg_name.replace('_', ' ').title()} Segment:")
        for i, row in imp_df.head(10).iterrows():
            print(f"  {row['Feature']:35s} {row['Mean_Abs_SHAP']:.4f}")

    return results
