# Value-Driven Churn Mitigation: Hierarchical Ensemble Learning (HEL) Framework

A two-stage Hierarchical Ensemble Learning framework that integrates value-based customer segmentation with segment-specific XGBoost classifiers for enhanced bank customer churn prediction and profit-oriented retention strategy development.

## Research Questions

- **RQ1**: Does the HEL framework outperform a monolithic XGBoost baseline in predicting customer churn?
- **RQ2**: What differential feature importance patterns exist across customer value segments?
- **RQ3**: What is the financial impact (Expected Financial Loss) of the HEL framework vs the baseline?

## HEL Framework Architecture

```
Banking Customer Dataset (1,000 customers, 5 tables)
                    │
        ┌───────────┴───────────┐
        │  STAGE 1: Value-Based │
        │  Segmentation (Median │
        │  Split on TMV)        │
        └───────────┬───────────┘
              ┌─────┴─────┐
              ▼           ▼
     ┌──────────────┐ ┌──────────────┐
     │  High-Value  │ │  Low-Value   │
     │  Segment     │ │  Segment     │
     └──────┬───────┘ └──────┬───────┘
            ▼                ▼
     ┌──────────────┐ ┌──────────────┐
     │ SMOTE +      │ │ SMOTE +      │
     │ XGBoost      │ │ XGBoost      │
     │ (Grid Search)│ │ (Grid Search)│
     └──────┬───────┘ └──────┬───────┘
            └─────┬──────────┘
                  ▼
     ┌────────────────────────┐
     │ Combined Predictions   │
     │ + SHAP Interpretability│
     │ + Financial Assessment │
     └────────────────────────┘
```

## Project Structure

```
├── main.py                       # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
├── src/
│   ├── data_generation.py        # Synthetic data matching Lloyds/Forage statistics
│   ├── feature_engineering.py    # Feature groups, encoding, value segmentation
│   ├── models.py                 # Monolithic baseline + HEL framework
│   ├── evaluation.py             # Metrics, McNemar's test, bootstrap CIs, EFL
│   ├── shap_analysis.py          # Global + per-segment SHAP analysis
│   └── visualisations.py         # All dissertation figures (300 DPI)
├── data/
│   └── Customer_Churn_Data_Large.xlsx
├── figures/                      # Generated figures (16 PNGs)
└── outputs/                      # Result tables (11 CSVs)
```

## Dataset

Based on the Lloyds Banking Group Job Simulation Task from the [Forage platform](https://www.theforage.com/). The dataset comprises 1,000 customer records across five interrelated tables:

| Component | Records | Key Variables |
|-----------|---------|---------------|
| Customer Demographics | 1,000 | Age, Gender, Marital Status, Income Level |
| Transaction History | 5,054 | Amount, Frequency, Category |
| Customer Service | 1,002 | Interaction Type, Resolution Status |
| Online Activity | 1,000 | Login Frequency, Service Channel |
| Churn Status | 1,000 | Binary: Churned (20.4%) vs Retained (79.6%) |

## Methodology

### Feature Engineering (3 groups)
1. **Transactional**: Transaction Frequency, Total Monetary Value, Average Transaction Value, Category Diversity
2. **Service Interaction**: Complaint Count, Unresolved Complaint Ratio, Total Interactions
3. **Digital Engagement**: Login Frequency, Days Since Last Login

### Model Training
- **SMOTE** (k=5) for class imbalance handling
- **5-fold stratified cross-validation** grid search over max_depth, learning_rate, n_estimators, reg_lambda
- **70:30 stratified train-test split**

### Evaluation
- Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR
- Threshold optimisation: Youden's J-statistic
- Statistical testing: McNemar's test with Cohen's h effect size
- Confidence intervals: 1,000 bootstrap repetitions at 95% CI
- Financial: Expected Financial Loss (CLV-based FN costs + intervention costs)
- Interpretability: SHAP TreeExplainer (global and per-segment)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis pipeline
python main.py
```

### Using the Real Forage Dataset

The pipeline auto-detects the data format. To use the real data:

1. Download the Excel file from the [Forage platform](https://cdn.theforage.com/vinternships/companyassets/Zbnc2o4ok6kD2NEXx/2kCX23cgKgCumeEam/1721851467492/Customer_Churn_Data_Large.xlsx)
2. Place it at `data/Customer_Churn_Data_Large.xlsx` (replacing the existing file)
3. Run `python main.py`

## Generated Outputs

### Figures (`figures/`)
| File | Description |
|------|-------------|
| `fig1_hel_architecture.png` | HEL framework architecture diagram |
| `fig2_churn_distribution.png` | Churn class distribution |
| `fig3_feature_correlation.png` | Feature correlation heatmap |
| `fig3b_churn_correlations.png` | Feature-churn correlation bar chart |
| `fig_roc_comparison.png` | ROC curves: Monolithic vs HEL |
| `fig_pr_comparison.png` | Precision-Recall curves comparison |
| `fig_confusion_matrices.png` | Side-by-side confusion matrices |
| `fig_financial_comparison.png` | EFL cost breakdown comparison |
| `fig_segment_metrics.png` | Metrics by customer value segment |
| `fig_metrics_table.png` | Formatted performance summary table |
| `fig_shap_global_summary.png` | Global SHAP feature importance |
| `fig_shap_segment_high_value.png` | SHAP importance: high-value segment |
| `fig_shap_segment_low_value.png` | SHAP importance: low-value segment |
| `fig_shap_segment_comparison.png` | Cross-segment SHAP comparison |
| `fig_shap_dependence.png` | SHAP dependence plots (top 4 features) |

### Data Tables (`outputs/`)
| File | Description |
|------|-------------|
| `classification_metrics.csv` | Full metrics for all models |
| `bootstrap_confidence_intervals.csv` | 95% CIs for all metrics |
| `statistical_tests.csv` | McNemar's test results |
| `financial_impact.csv` | EFL breakdown per model |
| `engineered_features.csv` | Complete feature matrix |
| `cv_results_*.csv` | Cross-validation results per model |
| `shap_importance_*.csv` | SHAP feature rankings per segment |

## Technologies

- Python 3.11+
- XGBoost (gradient boosting)
- scikit-learn (preprocessing, evaluation)
- imbalanced-learn (SMOTE)
- SHAP (model interpretability)
- matplotlib / seaborn (visualisation)
- pandas / numpy / scipy (data processing and statistics)
