# Value-Driven Churn Mitigation: Hierarchical Ensemble Learning (HEL) Framework

A two-stage Hierarchical Ensemble Learning framework that integrates value-based customer segmentation with segment-specific XGBoost classifiers for enhanced bank customer churn prediction and profit-oriented retention strategy development.

## Research Questions

- **RQ1**: Does the HEL framework outperform a monolithic XGBoost baseline in predicting customer churn?
- **RQ2**: What differential feature importance patterns exist across customer value segments?
- **RQ3**: What is the financial impact (Expected Financial Loss) of the HEL framework vs the baseline?

## HEL Framework Architecture

```
Banking Customer Dataset (1,000 customers, 5 tables)
                    |
        +-----------+-----------+
        |  STAGE 1: Value-Based |
        |  Segmentation (Median |
        |  Split on TMV)        |
        +-----------+-----------+
              +-----+-----+
              v           v
     +--------------+ +--------------+
     |  High-Value  | |  Low-Value   |
     |  Segment     | |  Segment     |
     +------+-------+ +------+-------+
            v                v
     +--------------+ +--------------+
     | SMOTE +      | | SMOTE +      |
     | XGBoost      | | XGBoost      |
     | (Grid Search)| | (Grid Search)|
     +------+-------+ +------+-------+
            +-----+----------+
                  v
     +------------------------+
     | Combined Predictions   |
     | + SHAP Interpretability|
     | + Financial Assessment |
     +------------------------+
```

## Project Structure

```
├── main.py                       # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
├── src/
│   ├── data_generation.py        # Synthetic data generator (fallback)
│   ├── feature_engineering.py    # Feature groups, encoding, value segmentation
│   ├── models.py                 # Monolithic baseline + HEL framework
│   ├── evaluation.py             # Metrics, McNemar's test, bootstrap CIs, EFL
│   ├── shap_analysis.py          # Global + per-segment SHAP analysis
│   └── visualisations.py         # All dissertation figures (300 DPI)
├── data/
│   └── Customer_Churn_Data_Large.xlsx   # Forage dataset
├── figures/                      # Generated figures (16 PNGs)
└── outputs/                      # Result tables (12 CSVs)
```

## Dataset

Source: **Lloyds Banking Group Job Simulation Task** from the [Forage education platform](https://www.theforage.com/).

The dataset comprises 1,000 customer records across five interrelated tables:

| Component | Records | Key Variables |
|-----------|---------|---------------|
| Customer Demographics | 1,000 | Age (M=43.3, SD=15.2), Gender (51.3% F), Marital Status, Income Level |
| Transaction History | 5,054 | Amount (M=£250.71), Transaction ID, Date, Product Category |
| Customer Service | 1,002 | Interaction Type (33.4% Complaints), Resolution Status (47.8% Unresolved) |
| Online Activity | 1,000 | Login Frequency (M=25.9), Last Login Date, Service Usage |
| Churn Status | 1,000 | Binary: Churned n=204 (20.4%) vs Retained n=796 (79.6%) |

The pipeline automatically handles the Forage column naming conventions (CamelCase) by standardising them to the internal schema (e.g., `AmountSpent` -> `Transaction_Amount`, `LastLoginDate` -> derived `Days_Since_Last_Login`).

## Methodology

### Feature Engineering (3 groups, 22 encoded features)
1. **Transactional**: Transaction Frequency, Total Monetary Value, Average Transaction Value, Category Diversity
2. **Service Interaction**: Complaint Count, Unresolved Complaint Ratio, Total Interactions
3. **Digital Engagement**: Login Frequency, Days Since Last Login

Categorical variables are one-hot encoded; continuous variables are z-score normalised.

### Value-Based Segmentation (HEL Stage 1)
- Median split on Total Monetary Value (median = £1,232.88)
- High-value segment: customers above median
- Low-value segment: customers below median

### Model Training (HEL Stage 2)
- **SMOTE** (k=5) for class imbalance handling
- **5-fold stratified cross-validation** grid search over:
  - `max_depth`: [3, 5, 7, 10]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.3]
  - `n_estimators`: [100, 200, 300, 500]
  - `reg_lambda`: [0.1, 0.5, 1.0]
- **70:30 stratified train-test split**

### Evaluation
- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR
- **Threshold optimisation**: Youden's J-statistic (J = Sensitivity + Specificity - 1)
- **Statistical testing**: McNemar's test with Cohen's h effect size
- **Confidence intervals**: 1,000 bootstrap repetitions at 95% CI
- **Financial**: Expected Financial Loss: EFL = Sum(FN x CLV_i) + Sum(FP x Intervention_Cost)
  - CLV calculated using 3-year retention period with 5% discount rate
  - Intervention cost: £50 per customer
- **Interpretability**: SHAP TreeExplainer (global and per-segment)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis pipeline
python main.py
```

The pipeline auto-detects whether the Excel file contains the real Forage data (CamelCase columns) or the internally generated format, and handles both transparently.

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
| `engineered_features.csv` | Complete feature matrix (1,000 x 15) |
| `cv_results_monolithic.csv` | Cross-validation results: monolithic baseline |
| `cv_results_high_value.csv` | Cross-validation results: high-value segment |
| `cv_results_low_value.csv` | Cross-validation results: low-value segment |
| `shap_importance_monolithic.csv` | SHAP feature rankings: global |
| `shap_importance_high_value.csv` | SHAP feature rankings: high-value segment |
| `shap_importance_low_value.csv` | SHAP feature rankings: low-value segment |

## Technologies

- Python 3.11+
- XGBoost (gradient boosting)
- scikit-learn (preprocessing, evaluation)
- imbalanced-learn (SMOTE)
- SHAP (model interpretability)
- matplotlib / seaborn (visualisation)
- pandas / numpy / scipy (data processing and statistics)
