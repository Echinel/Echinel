"""
Feature Engineering Module
==========================
Converts raw transactional and interaction data into customer-level
aggregated features as described in Section 3.3.

Feature Groups:
(i)   Transactional: Transaction Frequency, Total Monetary Value,
      Average Transaction Value, Category Diversity
(ii)  Service Interaction: Complaint Count, Unresolved Complaint Ratio,
      Service Channel Utilisation
(iii) Digital Engagement: Login Frequency, Days Since Last Login
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def engineer_features(demographics_df, transactions_df, service_df,
                      online_df, churn_df):
    """
    Build customer-level feature matrix from raw tables.

    Returns:
        df_features: DataFrame with engineered features and churn label
    """
    # --- (i) Transactional Features ---
    tx_agg = transactions_df.groupby("CustomerID").agg(
        Transaction_Frequency=("Transaction_Amount", "count"),
        Total_Monetary_Value=("Transaction_Amount", "sum"),
        Avg_Transaction_Value=("Transaction_Amount", "mean"),
        Category_Diversity=("Transaction_Category", "nunique")
    ).reset_index()

    # --- (ii) Service Interaction Features ---
    # Total interactions per customer
    svc_total = service_df.groupby("CustomerID").size().reset_index(
        name="Total_Interactions"
    )

    # Complaint count
    complaints = service_df[
        service_df["Interaction_Type"] == "Complaint"
    ].groupby("CustomerID").size().reset_index(name="Complaint_Count")

    # Unresolved complaint ratio
    complaint_records = service_df[
        service_df["Interaction_Type"] == "Complaint"
    ].copy()
    complaint_records["Is_Unresolved"] = (
        complaint_records["Resolution_Status"] == "Unresolved"
    ).astype(int)

    unresolved_ratio = complaint_records.groupby("CustomerID").agg(
        Unresolved_Complaint_Ratio=("Is_Unresolved", "mean")
    ).reset_index()

    # --- (iii) Digital Engagement Features ---
    # Already at customer level: Login_Frequency, Days_Since_Last_Login
    digital = online_df[
        ["CustomerID", "Login_Frequency", "Days_Since_Last_Login",
         "Service_Channel"]
    ].copy()

    # --- Merge all features ---
    df = demographics_df.merge(tx_agg, on="CustomerID", how="left")
    df = df.merge(svc_total, on="CustomerID", how="left")
    df = df.merge(complaints, on="CustomerID", how="left")
    df = df.merge(unresolved_ratio, on="CustomerID", how="left")
    df = df.merge(digital, on="CustomerID", how="left")
    df = df.merge(churn_df, on="CustomerID", how="left")

    # Fill NaN for customers with no complaints/service interactions
    df["Total_Interactions"] = df["Total_Interactions"].fillna(0)
    df["Complaint_Count"] = df["Complaint_Count"].fillna(0)
    df["Unresolved_Complaint_Ratio"] = df[
        "Unresolved_Complaint_Ratio"
    ].fillna(0)

    # Drop CustomerID to prevent data leakage (Section 3.3)
    df = df.drop(columns=["CustomerID"])

    return df


def encode_and_scale(df, target_col="Churn"):
    """
    One-hot encode categorical variables and z-score standardise
    continuous variables (Section 3.3).

    Returns:
        X: feature matrix (DataFrame)
        y: target series
        feature_names: list of feature names
        scaler: fitted StandardScaler
    """
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)

    # Z-score standardisation for continuous variables
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    feature_names = X.columns.tolist()

    return X, y, feature_names, scaler


def segment_by_value(X, df_raw, target_col="Churn"):
    """
    Stage 1 of HEL: Value-based segmentation using median split
    on Total Monetary Value (Section 3.4).

    Returns:
        segments: dict with 'high' and 'low' keys, each containing
                  (X_segment, y_segment, indices)
        median_value: the median Total Monetary Value used for splitting
    """
    # Use the raw (unscaled) Total Monetary Value for segmentation
    total_monetary = df_raw["Total_Monetary_Value"] if "Total_Monetary_Value" in df_raw.columns else None

    if total_monetary is None:
        raise ValueError("Total_Monetary_Value not found in raw features")

    median_value = total_monetary.median()
    print(f"Median Total Monetary Value for segmentation: £{median_value:.2f}")

    high_mask = total_monetary >= median_value
    low_mask = ~high_mask

    y = df_raw[target_col]

    segments = {
        "high_value": {
            "X": X[high_mask].copy(),
            "y": y[high_mask].copy(),
            "indices": np.where(high_mask)[0],
            "raw": df_raw[high_mask].copy()
        },
        "low_value": {
            "X": X[low_mask].copy(),
            "y": y[low_mask].copy(),
            "indices": np.where(low_mask)[0],
            "raw": df_raw[low_mask].copy()
        }
    }

    for seg_name, seg_data in segments.items():
        n = len(seg_data["y"])
        n_churn = seg_data["y"].sum()
        print(f"  {seg_name}: n={n}, churned={n_churn} ({100*n_churn/n:.1f}%)")

    return segments, median_value
