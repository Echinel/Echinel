"""
Data Generation Module
======================
Generates synthetic banking customer data matching the descriptive statistics
from the Lloyds Banking Group Job Simulation Task (Forage platform).

Dataset Composition (Table 1):
- Customer Demographics: 1,000 records
- Transaction History: 5,054 records
- Customer Service: 1,002 records
- Online Activity: 1,000 records
- Churn Status: 1,000 records (20.4% churned)

Feature-churn relationships are calibrated to produce realistic predictive
signal strengths consistent with banking churn literature.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_dataset(seed=42, output_path="data/Customer_Churn_Data_Large.xlsx"):
    """Generate synthetic banking dataset matching dissertation specifications."""
    np.random.seed(seed)
    n_customers = 1000
    n_churned = 204
    n_retained = 796

    customer_ids = np.arange(1, n_customers + 1)

    # --- Churn Status ---
    churn_labels = np.array([1] * n_churned + [0] * n_retained)
    np.random.shuffle(churn_labels)
    churn_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Churn": churn_labels
    })

    # --- Customer Demographics ---
    # Age: M=43.3, SD=14.2 — churners tend to be younger
    age = np.zeros(n_customers)
    for i in range(n_customers):
        if churn_labels[i] == 1:
            age[i] = np.random.normal(38.5, 13.0)
        else:
            age[i] = np.random.normal(44.5, 14.5)
    age = np.clip(age, 18, 85).astype(int)

    # Gender: 51.3% Female
    gender = np.random.choice(
        ["Female", "Male"], n_customers, p=[0.513, 0.487]
    )

    # Marital Status — single customers churn more
    marital_status = np.empty(n_customers, dtype=object)
    for i in range(n_customers):
        if churn_labels[i] == 1:
            marital_status[i] = np.random.choice(
                ["Married", "Single", "Divorced", "Widowed"],
                p=[0.30, 0.42, 0.18, 0.10]
            )
        else:
            marital_status[i] = np.random.choice(
                ["Married", "Single", "Divorced", "Widowed"],
                p=[0.45, 0.30, 0.15, 0.10]
            )

    # Income Level: High 34.9%, Medium 32.6%, Low 32.5%
    # Low income churns more
    income_level = np.empty(n_customers, dtype=object)
    for i in range(n_customers):
        if churn_labels[i] == 1:
            income_level[i] = np.random.choice(
                ["High", "Medium", "Low"], p=[0.25, 0.30, 0.45]
            )
        else:
            income_level[i] = np.random.choice(
                ["High", "Medium", "Low"], p=[0.38, 0.33, 0.29]
            )

    demographics_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": age,
        "Gender": gender,
        "Marital_Status": marital_status,
        "Income_Level": income_level
    })

    # --- Transaction History ---
    # Total ~5,054 transactions, Mean freq ~5.05, Mean amount ~£248.81
    # Churners: fewer transactions, lower amounts
    tx_counts = np.zeros(n_customers, dtype=int)
    for i in range(n_customers):
        if churn_labels[i] == 1:
            tx_counts[i] = max(1, np.random.poisson(3.8))
        else:
            tx_counts[i] = max(1, np.random.poisson(5.4))

    # Adjust to hit ~5054 total
    total_diff = 5054 - tx_counts.sum()
    if total_diff > 0:
        idx = np.random.choice(n_customers, total_diff, replace=True)
        for i in idx:
            tx_counts[i] += 1
    elif total_diff < 0:
        idx = np.random.choice(
            np.where(tx_counts > 1)[0], abs(total_diff), replace=True
        )
        for i in idx:
            tx_counts[i] = max(1, tx_counts[i] - 1)

    categories = ["Transfer", "Payment", "Withdrawal", "Deposit", "Purchase"]
    tx_rows = []
    for cid_idx in range(n_customers):
        cid = customer_ids[cid_idx]
        is_churner = churn_labels[cid_idx]
        n_tx = tx_counts[cid_idx]
        # Churners: lower amounts
        mean_amt = 190.0 if is_churner else 265.0
        amounts = np.random.lognormal(
            mean=np.log(mean_amt) - 0.5 * 0.7**2, sigma=0.7, size=n_tx
        )
        amounts = np.clip(amounts, 5, 5000)
        # Churners: less category diversity
        if is_churner:
            n_cats = np.random.choice([2, 3], p=[0.6, 0.4])
        else:
            n_cats = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        used_cats = np.random.choice(categories, n_cats, replace=False)
        cats = np.random.choice(used_cats, n_tx)
        for j in range(n_tx):
            tx_rows.append({
                "CustomerID": cid,
                "Transaction_Amount": round(float(amounts[j]), 2),
                "Transaction_Category": cats[j]
            })

    transactions_df = pd.DataFrame(tx_rows)

    # --- Customer Service Interactions ---
    # 1,002 records, 33.5% Complaints, 47.8% Unresolved
    # Churners: significantly more complaints and unresolved
    n_service = 1002
    # Weight service records towards churners
    churn_idx = np.where(churn_labels == 1)[0]
    retain_idx = np.where(churn_labels == 0)[0]

    # Churners get ~1.5 service interactions on average, retained ~0.8
    n_churn_svc = int(n_service * 0.35)
    n_retain_svc = n_service - n_churn_svc
    service_cids = np.concatenate([
        np.random.choice(customer_ids[churn_idx], n_churn_svc, replace=True),
        np.random.choice(customer_ids[retain_idx], n_retain_svc, replace=True)
    ])
    np.random.shuffle(service_cids)

    interaction_types = np.empty(n_service, dtype=object)
    resolution_status = np.empty(n_service, dtype=object)

    for i in range(n_service):
        cid = service_cids[i]
        is_churner = churn_labels[cid - 1]
        if is_churner:
            interaction_types[i] = np.random.choice(
                ["Complaint", "Inquiry", "Request"], p=[0.55, 0.25, 0.20]
            )
            resolution_status[i] = np.random.choice(
                ["Unresolved", "Resolved"], p=[0.65, 0.35]
            )
        else:
            interaction_types[i] = np.random.choice(
                ["Complaint", "Inquiry", "Request"], p=[0.22, 0.45, 0.33]
            )
            resolution_status[i] = np.random.choice(
                ["Unresolved", "Resolved"], p=[0.38, 0.62]
            )

    service_df = pd.DataFrame({
        "CustomerID": service_cids,
        "Interaction_Type": interaction_types,
        "Resolution_Status": resolution_status
    })

    # --- Online Activity ---
    # Login Frequency: M=25.9 — churners login significantly less
    login_freq = np.zeros(n_customers, dtype=int)
    for i in range(n_customers):
        if churn_labels[i] == 1:
            login_freq[i] = max(0, int(np.random.normal(18.0, 8.0)))
        else:
            login_freq[i] = max(0, int(np.random.normal(28.0, 9.0)))

    # Service channel — churners more on phone/branch (less digital)
    service_channels = np.empty(n_customers, dtype=object)
    for i in range(n_customers):
        if churn_labels[i] == 1:
            service_channels[i] = np.random.choice(
                ["Mobile", "Web", "Branch", "Phone"],
                p=[0.20, 0.20, 0.30, 0.30]
            )
        else:
            service_channels[i] = np.random.choice(
                ["Mobile", "Web", "Branch", "Phone"],
                p=[0.40, 0.35, 0.15, 0.10]
            )

    # Days since last login: churners much higher
    days_since_login = np.zeros(n_customers, dtype=int)
    for i in range(n_customers):
        if churn_labels[i] == 1:
            days_since_login[i] = max(0, int(np.random.exponential(35)))
        else:
            days_since_login[i] = max(0, int(np.random.exponential(10)))
    days_since_login = np.clip(days_since_login, 0, 120)

    online_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Login_Frequency": login_freq,
        "Service_Channel": service_channels,
        "Days_Since_Last_Login": days_since_login
    })

    # --- Save to Excel with multiple sheets ---
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        demographics_df.to_excel(writer, sheet_name="Demographics", index=False)
        transactions_df.to_excel(writer, sheet_name="Transactions", index=False)
        service_df.to_excel(writer, sheet_name="Customer_Service", index=False)
        online_df.to_excel(writer, sheet_name="Online_Activity", index=False)
        churn_df.to_excel(writer, sheet_name="Churn_Status", index=False)

    print(f"Dataset saved to {output}")
    print(f"  Demographics: {len(demographics_df)} records")
    print(f"  Transactions: {len(transactions_df)} records")
    print(f"  Customer Service: {len(service_df)} records")
    print(f"  Online Activity: {len(online_df)} records")
    print(f"  Churn Status: {len(churn_df)} records "
          f"(Churned={churn_labels.sum()}, Retained={n_customers - churn_labels.sum()})")

    return demographics_df, transactions_df, service_df, online_df, churn_df


if __name__ == "__main__":
    generate_dataset()
