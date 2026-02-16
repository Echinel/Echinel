"""
Visualisation Module
====================
Generates all dissertation figures:
- Figure 2: Churn distribution
- Figure 3: Feature correlation heatmap
- ROC curves comparison
- Precision-Recall curves comparison
- SHAP summary plots (global and per-segment)
- Confusion matrices
- Feature importance comparison across segments
- Financial impact comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path

# Dissertation-quality plot settings
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.figsize": (8, 6),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

FIGURES_DIR = Path("figures")


def save_fig(fig, name):
    """Save figure to figures directory."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_churn_distribution(y, name="fig2_churn_distribution"):
    """Figure 2: Initial churn distribution analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Bar chart
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["Retained", "Churned"]
    colors = ["#2196F3", "#F44336"]
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black",
                       linewidth=0.5)
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(count), ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_title("(a) Churn Count Distribution")
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, explode=(0, 0.05),
                textprops={"fontsize": 11})
    axes[1].set_title("(b) Churn Proportion")

    fig.suptitle("Figure 2: Churn Distribution Analysis", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, name)


def plot_correlation_heatmap(df, target_col="Churn",
                             name="fig3_feature_correlation"):
    """Figure 3: Feature correlation heatmap."""
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("Figure 3: Feature Correlation Matrix", fontsize=14,
                 fontweight="bold", pad=20)
    fig.tight_layout()
    save_fig(fig, name)

    # Also create a targeted churn correlation bar chart
    if target_col in corr.columns:
        churn_corr = corr[target_col].drop(target_col).sort_values()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = ["#F44336" if v > 0 else "#2196F3" for v in churn_corr.values]
        churn_corr.plot(kind="barh", ax=ax2, color=colors, edgecolor="black",
                        linewidth=0.3)
        ax2.set_xlabel("Pearson Correlation with Churn")
        ax2.set_title("Feature Correlations with Churn Status",
                       fontsize=13, fontweight="bold")
        ax2.axvline(x=0, color="black", linewidth=0.8)
        fig2.tight_layout()
        save_fig(fig2, "fig3b_churn_correlations")


def plot_roc_curves(results_dict, name="fig_roc_comparison"):
    """ROC curve comparison between models."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {"Monolithic Baseline": "#2196F3", "HEL Framework": "#F44336",
              "HEL High-Value": "#FF9800", "HEL Low-Value": "#4CAF50"}
    linestyles = {"Monolithic Baseline": "-", "HEL Framework": "-",
                  "HEL High-Value": "--", "HEL Low-Value": "--"}

    for label, data in results_dict.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc = data["metrics"]["AUC-ROC"]
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})",
                color=colors.get(label, "gray"),
                linestyle=linestyles.get(label, "-"),
                linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
            label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison: Monolithic vs HEL Framework",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, name)


def plot_precision_recall_curves(results_dict, name="fig_pr_comparison"):
    """Precision-Recall curve comparison."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {"Monolithic Baseline": "#2196F3", "HEL Framework": "#F44336",
              "HEL High-Value": "#FF9800", "HEL Low-Value": "#4CAF50"}

    for label, data in results_dict.items():
        precision, recall, _ = precision_recall_curve(
            data["y_true"], data["y_proba"])
        auc_pr = data["metrics"]["AUC-PR"]
        ax.plot(recall, precision, label=f"{label} (AUC-PR={auc_pr:.3f})",
                color=colors.get(label, "gray"), linewidth=2)

    # Baseline: proportion of positive class
    baseline = results_dict[list(results_dict.keys())[0]]["y_true"].mean()
    ax.axhline(y=baseline, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label=f"Baseline ({baseline:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, name)


def plot_confusion_matrices(results_dict, name="fig_confusion_matrices"):
    """Side-by-side confusion matrices."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, results_dict.items()):
        cm = confusion_matrix(data["y_true"], data["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Retained", "Churned"],
                    yticklabels=["Retained", "Churned"],
                    linewidths=0.5, linecolor="gray")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(label, fontweight="bold")

    fig.suptitle("Confusion Matrix Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, name)


def plot_financial_comparison(efl_results, name="fig_financial_comparison"):
    """Financial impact comparison between models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(efl_results.keys())
    efls = [efl_results[m]["EFL_Total"] for m in models]
    fn_costs = [efl_results[m]["FN_Cost_Lost_CLV"] for m in models]
    fp_costs = [efl_results[m]["FP_Cost_Unnecessary_Intervention"] for m in models]
    tp_costs = [efl_results[m]["TP_Intervention_Cost"] for m in models]

    # Total EFL bar chart
    colors = ["#2196F3", "#F44336"]
    bars = axes[0].bar(models, efls, color=colors[:len(models)],
                       edgecolor="black", linewidth=0.5)
    for bar, efl in zip(bars, efls):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(efls) * 0.02,
                     f"£{efl:,.0f}", ha="center", va="bottom",
                     fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Expected Financial Loss (£)")
    axes[0].set_title("(a) Total Expected Financial Loss",
                       fontweight="bold")
    axes[0].set_ylim(0, max(efls) * 1.2)

    # Cost breakdown stacked bar
    x = np.arange(len(models))
    width = 0.5
    axes[1].bar(x, fn_costs, width, label="FN: Lost CLV",
                color="#F44336", edgecolor="black", linewidth=0.3)
    axes[1].bar(x, fp_costs, width, bottom=fn_costs,
                label="FP: Unnecessary Intervention",
                color="#FF9800", edgecolor="black", linewidth=0.3)
    bottoms = [fn + fp for fn, fp in zip(fn_costs, fp_costs)]
    axes[1].bar(x, tp_costs, width, bottom=bottoms,
                label="TP: Intervention Cost",
                color="#4CAF50", edgecolor="black", linewidth=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Cost (£)")
    axes[1].set_title("(b) Cost Component Breakdown", fontweight="bold")
    axes[1].legend(fontsize=9)

    fig.suptitle("Financial Impact Assessment: EFL Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, name)


def plot_segment_comparison(segment_metrics, name="fig_segment_metrics"):
    """Compare metrics across value segments."""
    metrics_to_plot = ["F1-Score", "AUC-ROC", "Precision", "Recall"]
    segments = list(segment_metrics.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.3
    colors = ["#FF9800", "#4CAF50"]

    for i, seg in enumerate(segments):
        values = [segment_metrics[seg][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=seg.replace("_", " ").title(),
                      color=colors[i], edgecolor="black", linewidth=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics by Customer Value Segment",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig(fig, name)


def plot_metrics_comparison_table(metrics_df, ci_df=None,
                                  name="fig_metrics_table"):
    """Formatted metrics comparison table as figure."""
    display_cols = ["Model", "Accuracy", "Precision", "Recall",
                    "F1-Score", "AUC-ROC", "AUC-PR"]
    df_display = metrics_df[
        [c for c in display_cols if c in metrics_df.columns]
    ].copy()

    # Format numeric columns
    for col in df_display.columns:
        if col != "Model":
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else x
            )

    fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(df_display)))
    ax.axis("off")

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(df_display.columns)):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(df_display) + 1):
        color = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(df_display.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("Classification Performance Comparison",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    save_fig(fig, name)


def plot_hel_architecture(name="fig1_hel_architecture"):
    """Figure 1: HEL Framework Architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_props = dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                     edgecolor="#1565C0", linewidth=2)
    box_props_stage = dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                           edgecolor="#E65100", linewidth=2)
    box_props_output = dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                            edgecolor="#2E7D32", linewidth=2)

    # Input
    ax.text(5, 7.5, "Banking Customer Dataset\n(1,000 customers, 5 tables)",
            ha="center", va="center", fontsize=12, fontweight="bold",
            bbox=box_props)

    # Stage 1
    ax.annotate("", xy=(5, 6.5), xytext=(5, 7.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    ax.text(5, 6.2, "STAGE 1: Value-Based Segmentation\n"
            "(Median Split on Total Monetary Value)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=box_props_stage)

    # Two segments
    ax.annotate("", xy=(2.5, 5.0), xytext=(4, 5.7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    ax.annotate("", xy=(7.5, 5.0), xytext=(6, 5.7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    ax.text(2.5, 4.6, "High-Value Segment\n(TMV >= Median)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=box_props)
    ax.text(7.5, 4.6, "Low-Value Segment\n(TMV < Median)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=box_props)

    # Stage 2: SMOTE + XGBoost
    ax.annotate("", xy=(2.5, 3.5), xytext=(2.5, 4.1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    ax.annotate("", xy=(7.5, 3.5), xytext=(7.5, 4.1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    ax.text(2.5, 3.1, "STAGE 2a:\nSMOTE + XGBoost\n(Grid Search CV)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=box_props_stage)
    ax.text(7.5, 3.1, "STAGE 2b:\nSMOTE + XGBoost\n(Grid Search CV)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=box_props_stage)

    # Merge predictions
    ax.annotate("", xy=(5, 1.8), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    ax.annotate("", xy=(5, 1.8), xytext=(7.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    ax.text(5, 1.4, "Combined HEL Predictions\n+ SHAP Interpretability\n"
            "+ Financial Impact Assessment",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=box_props_output)

    ax.set_title("Figure 1: Hierarchical Ensemble Learning (HEL) "
                 "Framework Architecture",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    save_fig(fig, name)
