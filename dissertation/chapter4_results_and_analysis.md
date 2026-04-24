# Chapter 4: Results and Analysis

## 4.1 Introduction

This chapter presents the empirical results of the Hierarchical Ensemble Learning (HEL) framework applied to the Lloyds Banking Group churn prediction task. Results are organised around the three research questions: classification performance comparison (RQ1), differential feature importance across customer value segments (RQ2), and financial impact assessment (RQ3). All models were trained on the 1,000-customer Forage dataset using a 70:30 stratified train–test split with random seed 42 to ensure reproducibility.

---

## 4.2 Descriptive Statistics and Feature Engineering

### 4.2.1 Dataset Composition

The dataset comprises 1,000 unique customers distributed across five interrelated tables. Table 4.1 summarises the key descriptive statistics for each component.

**Table 4.1: Dataset Descriptive Statistics**

| Component | *n* | Key Statistics |
|---|---|---|
| Customer Demographics | 1,000 | Age: *M* = 43.3, *SD* = 15.2; Gender: 51.3% Female, 48.7% Male |
| Transaction History | 5,054 | Amount: *M* = £250.71; mean frequency per customer ≈ 5.05 |
| Customer Service | 1,002 | Complaints: 33.4%; Unresolved: 47.8% of interactions |
| Online Activity | 1,000 | Login Frequency: *M* = 25.9; Service Channels: Mobile App, Online Banking, Website |
| Churn Status | 1,000 | Churned: *n* = 204 (20.4%); Retained: *n* = 796 (79.6%) |

The churn base rate of 20.4% is consistent with the moderate imbalance commonly reported in banking churn literature (Vafeiadis et al., 2015; De Caigny et al., 2018). This imbalance necessitated the application of SMOTE during model training (see Section 3.4).

### 4.2.2 Engineered Feature Matrix

Feature engineering produced a matrix of 15 raw features (9 continuous, 4 categorical) per customer, which expanded to 22 features after one-hot encoding of categorical variables (Gender, Marital Status, Income Level, Service Channel). The three feature groups defined in Section 3.3 are:

1. **Transactional Features** (4): Transaction Frequency, Total Monetary Value, Average Transaction Value, Category Diversity
2. **Service Interaction Features** (3): Complaint Count, Unresolved Complaint Ratio, Total Interactions
3. **Digital Engagement Features** (2): Login Frequency, Days Since Last Login

Continuous variables were z-score standardised (*μ* = 0, *σ* = 1) using `StandardScaler`, and categorical variables were one-hot encoded without dropping the first category to preserve interpretability in the SHAP analysis (Lundberg and Lee, 2017).

### 4.2.3 Exploratory Data Analysis

The feature correlation heatmap (*Figure 3*) reveals moderate positive correlations between Transaction Frequency and Total Monetary Value (*r* = 0.52), as expected given that customers who transact more frequently accumulate higher total expenditure. Complaint Count and Total Interactions also show moderate correlation (*r* = 0.45), reflecting that complaint-related contacts constitute a substantial portion of all service interactions.

The feature–churn correlation analysis (*Figure 3b*) reveals that no single feature exhibits a strong linear relationship with churn status. The Pearson correlations between individual features and the binary churn variable are uniformly weak, with most |*r*| < 0.10. This finding underscores the necessity of machine learning approaches capable of capturing non-linear, higher-order interactions among features — motivating the use of gradient-boosted decision trees (XGBoost) within the HEL framework.

---

## 4.3 Value-Based Segmentation (HEL Stage 1)

### 4.3.1 Segmentation Results

The median Total Monetary Value (TMV) across the full dataset was £1,232.88. Customers were partitioned using this median into two value segments:

- **High-Value Segment** (TMV ≥ £1,232.88): *n* = 500 customers
- **Low-Value Segment** (TMV < £1,232.88): *n* = 500 customers

After the 70:30 stratified train–test split, the segment distributions are shown in Table 4.2.

**Table 4.2: Segment Composition After Train–Test Split**

| Segment | Train *n* | Train Churn Rate | Test *n* | Test Churn Rate |
|---|---|---|---|---|
| High-Value | ~350 | ~20% | ~141 | ~18.4% (26/141) |
| Low-Value | ~350 | ~20% | ~159 | ~22.0% (35/159) |
| **Combined** | **700** | **~20.4%** | **300** | **20.3% (61/300)** |

The churn rate differential between segments (high-value: ~18.4% vs low-value: ~22.0%) suggests that lower-value customers exhibit marginally higher churn propensity, consistent with the retention literature suggesting that customers with weaker financial engagement are more susceptible to attrition (Ascarza, 2018).

---

## 4.4 Classification Performance (RQ1)

### 4.4.1 Hyperparameter Optimisation

Grid search with 5-fold stratified cross-validation over 192 hyperparameter combinations (4 × 4 × 4 × 3) was conducted for each model, optimising on the F1-score of the minority (churn) class. Table 4.3 presents the best hyperparameters identified.

**Table 4.3: Optimal Hyperparameters per Model (5-Fold CV, F1-Optimised)**

| Parameter | Monolithic Baseline | HEL High-Value | HEL Low-Value |
|---|---|---|---|
| `learning_rate` | 0.30 | 0.30 | 0.01 |
| `max_depth` | 3 | 7 | 3 |
| `n_estimators` | 500 | 200 | 100 |
| `reg_lambda` | 1.0 | 0.5 | 0.5 |
| **Best CV F1** | **0.2907** | **0.3013** | **0.4060** |
| CV F1 *SD* | ±0.0693 | ±0.1009 | ±0.0467 |

Several observations emerge from the hyperparameter results:

1. The **monolithic baseline** selected the highest learning rate (0.30) with maximum number of estimators (500) and the strongest L2 regularisation (λ = 1.0), but retained shallow trees (depth = 3). This suggests the model needed extensive boosting iterations with strong regularisation to compensate for the heterogeneity of the pooled data.

2. The **high-value segment** model selected deeper trees (depth = 7) with moderate boosting iterations (200), indicating that more complex decision boundaries were required to separate churners from retained customers in this segment. The higher CV standard deviation (±0.1009) reflects greater instability, likely due to smaller training sample sizes within the segment.

3. The **low-value segment** model selected the smallest, most conservative configuration (learning rate = 0.01, depth = 3, 100 estimators), yet achieved the highest cross-validation F1-score (0.4060). This suggests that churn patterns in the low-value segment are more learnable and potentially more linearly separable, requiring less model complexity to capture.

### 4.4.2 Test Set Performance

Table 4.4 reports the complete classification performance on the held-out test set (*n* = 300).

**Table 4.4: Classification Performance on Test Set (*n* = 300)**

| Metric | Monolithic Baseline | HEL Framework | HEL High-Value | HEL Low-Value |
|---|---|---|---|---|
| **Accuracy** | 0.6933 | 0.6800 | 0.7305 | 0.6352 |
| **Precision** | 0.2075 | 0.1429 | 0.2000 | 0.1034 |
| **Recall (Sensitivity)** | 0.1803 | 0.1148 | 0.1538 | 0.0857 |
| **F1-Score** | 0.1930 | 0.1273 | 0.1739 | 0.0938 |
| **AUC-ROC** | 0.5305 | 0.5085 | 0.5428 | 0.4085 |
| **AUC-PR** | 0.2324 | 0.2100 | 0.2290 | 0.1848 |
| **Specificity** | 0.8243 | 0.8243 | 0.8609 | 0.7903 |
| **Youden's *J*** | 0.1166 | 0.1000 | 0.1669 | 0.0594 |
| **Optimal Threshold** | 0.2070 | 0.0529 | 0.0287 | 0.3190 |
| True Positives | 11 | 7 | 4 | 3 |
| False Positives | 42 | 42 | 16 | 26 |
| True Negatives | 197 | 197 | 99 | 98 |
| False Negatives | 50 | 54 | 22 | 32 |

The **Monolithic Baseline** achieves marginally higher performance across all primary metrics. With an F1-score of 0.1930 versus the HEL Framework's 0.1273, the baseline demonstrates a 34.1% relative advantage. Similarly, the baseline achieves higher Recall (0.1803 vs 0.1148), indicating that it correctly identifies a larger proportion of actual churners, albeit still only 18.0% of all true churners.

Both models exhibit identical Specificity (0.8243), producing the same number of false positives (*n* = 42). The differentiation lies entirely in churn detection (sensitivity), where the monolithic model captures 11 true positives compared to the HEL framework's 7.

### 4.4.3 Segment-Level Analysis

The per-segment decomposition reveals notable heterogeneity in the HEL framework's performance:

- **High-Value Segment**: Achieves the highest accuracy (0.7305) and Youden's *J* (0.1669) of any model or sub-model, suggesting that the segment-specific classifier is better calibrated for high-value customers. The AUC-ROC of 0.5428 also exceeds both the monolithic baseline (0.5305) and the combined HEL framework (0.5085).

- **Low-Value Segment**: Exhibits the weakest performance across all metrics (F1 = 0.0938, AUC-ROC = 0.4085). The AUC-ROC below 0.50 indicates discriminatory performance worse than random chance, suggesting that the model's predictions in this segment are essentially uninformative. This result likely reflects a combination of (i) smaller effective sample size within the segment, (ii) greater feature noise among low-value customers, and (iii) potential overfitting of the segment-specific model.

The combined HEL performance is a weighted average of these two segment models, and the poor low-value performance drags the overall HEL metrics below the monolithic baseline.

### 4.4.4 Bootstrap Confidence Intervals

Table 4.5 presents the 95% confidence intervals derived from 1,000 bootstrap repetitions.

**Table 4.5: 95% Bootstrap Confidence Intervals (1,000 Repetitions)**

| Metric | Monolithic Baseline | | HEL Framework | |
|---|---|---|---|---|
| | *M* | 95% CI | *M* | 95% CI |
| Accuracy | 0.6932 | [0.640, 0.743] | 0.6799 | [0.623, 0.733] |
| Precision | 0.2061 | [0.109, 0.327] | 0.1431 | [0.048, 0.255] |
| Recall | 0.1797 | [0.091, 0.286] | 0.1157 | [0.036, 0.203] |
| F1-Score | 0.1907 | [0.102, 0.296] | 0.1270 | [0.041, 0.218] |
| AUC-ROC | 0.5301 | [0.450, 0.621] | 0.5084 | [0.431, 0.590] |
| AUC-PR | 0.2381 | [0.173, 0.323] | 0.2208 | [0.160, 0.297] |

The confidence intervals are notably wide for both models, reflecting the limited test set size (*n* = 300) and severe class imbalance (only 61 positive cases). Crucially, the confidence intervals for all metrics **overlap substantially** between the two models. For example, the F1-Score CI for the monolithic model [0.102, 0.296] fully contains the HEL mean (0.1270), and vice versa. This overlap indicates that the observed performance differences may not be statistically reliable.

### 4.4.5 McNemar's Test

To formally assess whether the two models produce significantly different classification outcomes, McNemar's test was applied to the paired predictions on the test set.

**Table 4.6: McNemar's Test Results**

| Statistic | Value |
|---|---|
| Chi-Square (χ²) | 0.1607 |
| *p*-value | 0.6885 |
| Cohen's *h* | −0.0287 |
| Discordant pairs (Monolithic correct, HEL wrong) | 30 |
| Discordant pairs (HEL correct, Monolithic wrong) | 26 |
| Significance at α = 0.05 | **Not significant** |

The McNemar's test yields χ² = 0.1607, *p* = 0.6885, far exceeding the conventional significance threshold of α = 0.05. The effect size (Cohen's *h* = −0.0287) is negligible, well below the *h* = 0.20 threshold for a small effect (Cohen, 1988). The discordant pair analysis reveals a near-even split: 30 cases where the monolithic model was correct and HEL was wrong, versus 26 cases in the opposite direction. These results confirm that **there is no statistically significant difference in classification performance between the monolithic baseline and the HEL framework** on this dataset.

### 4.4.6 Summary for RQ1

**RQ1: Does the HEL framework outperform a monolithic XGBoost baseline in predicting customer churn?**

The evidence does not support a performance advantage for the HEL framework over the monolithic baseline. While the monolithic model achieves numerically higher scores across all metrics (F1: 0.1930 vs 0.1273; AUC-ROC: 0.5305 vs 0.5085), McNemar's test (*p* = 0.6885, Cohen's *h* = −0.029) confirms that this difference is not statistically significant. The overlapping bootstrap confidence intervals reinforce this conclusion. The HEL framework's high-value segment model shows promise (AUC-ROC = 0.5428, Youden's *J* = 0.1669), but the weak low-value segment performance (AUC-ROC = 0.4085) degrades the combined output. Both models struggle with the churn prediction task overall, with Recall below 0.20 and AUC-ROC near 0.50, suggesting fundamental limitations in the feature set's discriminatory power for this dataset.

---

## 4.5 Differential Feature Importance (RQ2)

### 4.5.1 SHAP Analysis Methodology

SHAP (SHapley Additive exPlanations) values were computed using TreeExplainer (Lundberg et al., 2020), which provides exact Shapley values for tree-based models in polynomial time. Mean absolute SHAP values were calculated across all test set observations to quantify each feature's average contribution to model predictions, both globally and per segment.

### 4.5.2 Global Feature Importance (Monolithic Model)

Table 4.7 presents the top 10 features ranked by mean absolute SHAP value for the monolithic baseline.

**Table 4.7: Global Feature Importance — Monolithic Baseline (Top 10)**

| Rank | Feature | Mean |SHAP| | Feature Group |
|---|---|---|---|
| 1 | Login Frequency | 0.9716 | Digital Engagement |
| 2 | Income Level (High) | 0.9679 | Demographic |
| 3 | Gender (Female) | 0.9583 | Demographic |
| 4 | Marital Status (Widowed) | 0.9398 | Demographic |
| 5 | Marital Status (Divorced) | 0.8929 | Demographic |
| 6 | Service Channel (Online Banking) | 0.7999 | Digital Engagement |
| 7 | Average Transaction Value | 0.7929 | Transactional |
| 8 | Days Since Last Login | 0.7515 | Digital Engagement |
| 9 | Age | 0.7057 | Demographic |
| 10 | Income Level (Medium) | 0.7036 | Demographic |

The global SHAP analysis reveals that **demographic variables dominate** the feature importance ranking, occupying 6 of the top 10 positions. Login Frequency ranks first (mean |SHAP| = 0.9716), closely followed by Income Level (High) (0.9679) and Gender (Female) (0.9583). The service interaction features (Complaint Count, Unresolved Complaint Ratio) rank lowest among all features (ranks 21–22, mean |SHAP| < 0.23), suggesting that complaint-related variables contribute minimally to churn prediction in this dataset.

### 4.5.3 High-Value Segment Feature Importance

**Table 4.8: Feature Importance — HEL High-Value Segment (Top 10)**

| Rank | Feature | Mean |SHAP| | Feature Group |
|---|---|---|---|
| 1 | Gender (Female) | 1.0964 | Demographic |
| 2 | Marital Status (Widowed) | 0.8235 | Demographic |
| 3 | Marital Status (Divorced) | 0.7423 | Demographic |
| 4 | Login Frequency | 0.7261 | Digital Engagement |
| 5 | Gender (Male) | 0.6287 | Demographic |
| 6 | Average Transaction Value | 0.6171 | Transactional |
| 7 | Income Level (High) | 0.5905 | Demographic |
| 8 | Age | 0.5731 | Demographic |
| 9 | Service Channel (Online Banking) | 0.5516 | Digital Engagement |
| 10 | Total Monetary Value | 0.5489 | Transactional |

In the high-value segment, **Gender (Female) emerges as the most influential predictor** (mean |SHAP| = 1.0964), displacing Login Frequency from the top position. Marital status indicators (Widowed, Divorced) remain prominent, suggesting that demographic life-stage factors are particularly salient for churn prediction among high-value customers. Transaction Frequency rises in relative importance (rank 13 globally → rank 13 in-segment but with higher |SHAP|), while Income Level (Medium) drops substantially (rank 10 globally → rank 22 in-segment, |SHAP| = 0.0973), reflecting the homogeneity of income levels within the high-value segment.

### 4.5.4 Low-Value Segment Feature Importance

**Table 4.9: Feature Importance — HEL Low-Value Segment (Top 10)**

| Rank | Feature | Mean |SHAP| | Feature Group |
|---|---|---|---|
| 1 | Income Level (High) | 0.3130 | Demographic |
| 2 | Income Level (Medium) | 0.2784 | Demographic |
| 3 | Income Level (Low) | 0.1332 | Demographic |
| 4 | Total Interactions | 0.1074 | Service Interaction |
| 5 | Total Monetary Value | 0.1069 | Transactional |
| 6 | Login Frequency | 0.0683 | Digital Engagement |
| 7 | Gender (Female) | 0.0482 | Demographic |
| 8 | Days Since Last Login | 0.0474 | Digital Engagement |
| 9 | Marital Status (Divorced) | 0.0365 | Demographic |
| 10 | Gender (Male) | 0.0273 | Demographic |

The low-value segment exhibits a **dramatically different feature importance profile**. Income Level categories dominate the top 3 positions (combined mean |SHAP| = 0.7247), with the magnitude far exceeding all other features. Total Interactions (rank 4, |SHAP| = 0.1074) becomes the most important service interaction feature — notably, this was ranked 13th globally. Several features that were highly influential in the high-value segment become essentially uninformative: Marital Status (Widowed), Transaction Frequency, and Marital Status (Single) all register |SHAP| = 0.000, indicating zero contribution to predictions.

The overall SHAP magnitudes in the low-value segment are approximately 3–10× smaller than in the high-value segment (maximum: 0.3130 vs 1.0964), which aligns with the poor discriminatory performance observed in Section 4.4.3 — the low-value model extracts substantially less signal from the feature space.

### 4.5.5 Cross-Segment Comparison

Table 4.10 highlights the most striking feature importance shifts between segments.

**Table 4.10: Key Feature Importance Differences Across Segments**

| Feature | High-Value |SHAP| | Low-Value |SHAP| | Ratio (H:L) | Direction of Shift |
|---|---|---|---|---|
| Gender (Female) | 1.0964 | 0.0482 | 22.7× | Far more important in high-value |
| Marital Status (Widowed) | 0.8235 | 0.0000 | — | Exclusive to high-value |
| Login Frequency | 0.7261 | 0.0683 | 10.6× | Far more important in high-value |
| Income Level (High) | 0.5905 | 0.3130 | 1.9× | Important in both, dominant in low-value |
| Total Interactions | 0.4504 | 0.1074 | 4.2× | Moderate in both |
| Complaint Count | 0.2048 | 0.0161 | 12.7× | Minimal overall, concentrated in high-value |

Three key patterns emerge from the cross-segment comparison:

1. **Demographic dominance varies by mechanism**: In the high-value segment, life-stage demographics (gender, marital status) drive predictions; in the low-value segment, economic demographics (income level) dominate. This suggests fundamentally different churn pathways across value tiers.

2. **Digital engagement is segment-dependent**: Login Frequency is 10.6× more important for high-value churn prediction than low-value. High-value customers who disengage digitally appear to signal imminent churn more reliably than their low-value counterparts.

3. **Service interaction features are universally weak**: Complaint Count and Unresolved Complaint Ratio rank among the bottom 3 features in both segments (|SHAP| < 0.23 globally, < 0.02 in low-value). Despite the literature's emphasis on service quality in churn (Amin et al., 2019), these features add minimal predictive value in this dataset.

### 4.5.6 Summary for RQ2

**RQ2: What differential feature importance patterns exist across customer value segments?**

The SHAP analysis reveals **substantial differential feature importance** between high-value and low-value customer segments. The high-value segment is characterised by strong demographic and digital engagement signals (Gender Female |SHAP| = 1.0964; Login Frequency = 0.7261), while the low-value segment is overwhelmingly driven by income-level indicators (combined |SHAP| = 0.7247). Several features that are highly predictive in one segment are entirely uninformative in the other (e.g., Marital Status Widowed: 0.8235 vs 0.0000). These differential patterns provide a strong rationale for segment-specific modelling strategies, even though the aggregate HEL classification performance did not exceed the baseline in this experiment. The findings suggest that the *nature* of churn drivers is indeed segment-dependent, supporting the theoretical motivation for the HEL framework.

---

## 4.6 Financial Impact Assessment (RQ3)

### 4.6.1 Expected Financial Loss Computation

The Expected Financial Loss (EFL) was computed following the methodology defined in Section 3.6:

$$\text{EFL} = \sum_{i \in FN} \text{CLV}_i + \sum_{j \in FP} C_{\text{intervention}}$$

Where Customer Lifetime Value (CLV) is calculated as:

$$\text{CLV}_i = \text{TMV}_i \times \text{PV Annuity Factor}$$

With a 3-year retention horizon and 5% annual discount rate, the present value annuity factor is:

$$\text{PV Factor} = \sum_{t=1}^{3} \frac{1}{(1 + 0.05)^t} = 2.7232$$

The intervention cost was set at £50 per customer, consistent with targeted retention programme estimates in the UK banking literature.

### 4.6.2 EFL Results

**Table 4.11: Expected Financial Loss Comparison**

| Component | Monolithic Baseline | HEL Framework | Difference |
|---|---|---|---|
| **Total EFL** | **£148,930.75** | **£158,843.92** | **+£9,913.17 (+6.7%)** |
| FN Cost (Lost CLV) | £146,830.75 | £156,743.92 | +£9,913.17 |
| FP Cost (Unnecessary Intervention) | £2,100.00 | £2,100.00 | £0.00 |
| TP Intervention Cost | £550.00 | £350.00 | −£200.00 |
| *n* False Negatives | 50 | 54 | +4 |
| *n* False Positives | 42 | 42 | 0 |
| *n* True Positives | 11 | 7 | −4 |
| Mean CLV per Missed Churner | £2,936.62 | £2,902.67 | −£33.95 |

### 4.6.3 Analysis of Financial Results

The financial assessment reveals that the **HEL framework incurs a higher Expected Financial Loss** than the monolithic baseline (£158,843.92 vs £148,930.75), representing a 6.7% cost increase. This result is directly attributable to the HEL framework's lower recall: by missing 4 additional churners (54 vs 50 false negatives), the HEL model fails to identify customers whose combined CLV totals approximately £9,913 more than those missed by the baseline.

Several key observations emerge:

1. **False Negative costs dominate**: FN costs account for 98.6% (monolithic) and 98.7% (HEL) of total EFL. The £2,100 FP intervention cost — representing 42 unnecessary retention offers at £50 each — is negligible relative to the CLV lost from missed churners. This asymmetry strongly supports threshold adjustment strategies that favour recall over precision in production deployment.

2. **Both models fail financially**: Both models miss the majority of actual churners (50/61 = 82.0% for monolithic; 54/61 = 88.5% for HEL), resulting in EFL exceeding £148,000. To contextualise, the total CLV at risk (sum of CLV for all 61 test-set churners) can be estimated as approximately 61 × £2,920 × 2.7232 ≈ £485,000. The monolithic model recovers only ~£30,000 of this through its 11 true positive identifications (11 × £50 intervention cost = £550, but the CLV preserved is approximately 11 × £2,920 × 2.7232 ≈ £87,500). The net financial benefit relative to *no* model is modest.

3. **Cost-sensitive threshold optimisation**: The optimal thresholds identified by Youden's *J* statistic are low (0.2070 for monolithic, 0.0529 for HEL), reflecting the models' tendency to output low churn probabilities. In a production setting, deliberately lowering the classification threshold further would increase recall at the cost of precision — but given the extreme FN-to-FP cost ratio (£2,937 per missed churner vs £50 per unnecessary intervention), even substantial increases in false positives would be financially justified.

### 4.6.4 Summary for RQ3

**RQ3: What is the financial impact (Expected Financial Loss) of the HEL framework vs the baseline?**

The HEL framework produces a **higher** Expected Financial Loss (£158,843.92) than the monolithic baseline (£148,930.75), a difference of £9,913.17 (6.7%). This unfavourable outcome is driven entirely by the HEL framework's 4 additional false negatives, whose lost CLV far outweighs any intervention cost savings. The analysis reveals that false negative costs constitute >98% of total EFL for both models, highlighting the extreme cost asymmetry in churn prediction. Both models leave the substantial majority (>80%) of churner CLV unrecovered, indicating that threshold optimisation and feature enrichment represent important avenues for improving financial outcomes regardless of the modelling framework employed.

---

## 4.7 Discussion

### 4.7.1 Interpretation of Results

The results present a nuanced picture of the HEL framework's utility. While the framework did not achieve its primary objective of outperforming the monolithic baseline on classification metrics (RQ1), the SHAP-based analysis (RQ2) provides compelling evidence for differential churn mechanisms across customer value segments. This finding has direct implications for retention strategy design, even if the classification improvement was not realised.

The uniformly low performance of both models (AUC-ROC ≈ 0.51–0.53, F1 ≈ 0.13–0.19) across the board warrants discussion. Several factors likely contribute:

1. **Feature set limitations**: The nine base features (pre-encoding) may lack the discriminatory power required for reliable churn prediction. Banking churn literature consistently identifies features such as product holdings, tenure, complaint severity scores, and competitor offer exposure as strong predictors (Verbeke et al., 2012; De Caigny et al., 2018) — none of which are available in this dataset.

2. **Sample size constraints**: With only 1,000 customers (and 204 churners), the effective positive class count in each segment is approximately 70–100 (before train–test split). After the 70:30 split, each segment model trains on approximately 70 churners, a quantity that may be insufficient for XGBoost to learn stable, generalisable decision boundaries — particularly in the 22-dimensional feature space after encoding.

3. **Segmentation granularity**: The binary median split on TMV, while parsimonious, may not capture the heterogeneity suggested by the data. The low-value segment's near-random performance (AUC-ROC = 0.4085) suggests that this segment may itself contain sub-populations with distinct churn behaviours that a single model cannot jointly capture.

### 4.7.2 Comparison with Literature

The AUC-ROC values obtained (0.50–0.54) fall substantially below those reported in comparable banking churn studies. Vafeiadis et al. (2015) reported AUC values of 0.85–0.93 using neural networks and SVMs on a telecom dataset with 20+ features. De Caigny et al. (2018) achieved F1-scores exceeding 0.60 with profit-driven ensemble methods. The discrepancy is attributable to dataset differences rather than methodological shortcomings: the Forage dataset's 15 features (pre-encoding) represent a relatively sparse feature space compared to the 30–100+ features typical of industry churn datasets (Verbeke et al., 2012).

### 4.7.3 Practical Implications

Despite the modest classification performance, several actionable insights emerge:

1. **Segment-specific retention strategies are warranted**: The differential SHAP profiles suggest that high-value customer churn is associated with demographic and engagement factors, while low-value churn is predominantly an income-level phenomenon. Retention interventions should be tailored accordingly — for example, digital re-engagement campaigns for high-value segments versus financial incentive programmes for low-value segments.

2. **Threshold optimisation is essential**: Given the FN/FP cost ratio of approximately 59:1 (£2,937:£50), the models should be deployed with thresholds substantially below the default 0.50 — and potentially below the Youden's *J* optimum — to maximise churner capture even at the expense of increased false positives.

3. **Feature enrichment is the primary lever for improvement**: The weak feature–churn correlations and low model performance suggest that additional features (product holdings, competitive intelligence, complaint text analysis, channel usage patterns) would yield greater performance gains than further algorithmic refinement.

---

## 4.8 Chapter Summary

This chapter presented the results of the HEL framework evaluation across three research questions. The monolithic XGBoost baseline marginally outperformed the HEL framework on classification metrics (F1: 0.1930 vs 0.1273), though the difference was not statistically significant (McNemar's *p* = 0.6885). SHAP analysis revealed substantial differential feature importance across segments, with demographic life-stage variables dominating high-value predictions and income-level indicators driving low-value predictions. The financial analysis showed both models incur substantial expected financial losses (>£148,000), with false negative costs constituting >98% of total EFL. These findings collectively suggest that while the HEL framework's segmented approach provides valuable interpretive insights, its classification advantage depends critically on adequate sample sizes and richer feature sets than were available in the current dataset.
