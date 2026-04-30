# %%
import polars as pl
import json
import os
import sys

sys.path.append("..")

from settings import (
    PROJECT_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# ---------------------- Data Loading ---------------------------
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "real_estate_transactions_engineered.parquet")
)

X = transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
y_regression = transactions[REGRESSION_TARGET]
y_classification = transactions[CLASSIFICATION_TARGET]

# %%
feature_names = [col for col in X.columns if col != "transaction_id"]

# %%
categorical_features = [col for col in feature_names if "building_type" in col]
categorical_features.extend(["off_plan", "city_requested"])
categorical_features.extend([col for col in feature_names if "region_name" in col])

# %%
numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
correlation_matrix = X.to_pandas()[numerical_features].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

"""
The previous month's average price per m2 is redundant with the 6-month rolling average,
indicating that price per m2 evolves slowly in most cases.
Same observation for the number of transactions.
The year is highly correlated with several other features.
The bank deposit fraction is highly correlated with other financial asset features.
"""

# %%
highly_correlated_features = [
    "building_type_house",             # Keep one or the other
    "num_rooms",                         # Keep living area instead
    # "transaction_year",              # Keep debt ratio instead
    "euros_per_capita",               # Highly correlated with debt ratio
    "life_insurance_share",
    "mutual_fund_share",
    "non_equity_securities_share",
    "equity_share",
    "debt_ratio",
    "pension_fund_share",
    "debt_ratio_change",
    "debt_ratio_acceleration",
    "rent_reference_index",
    "rolling_avg_price_per_m2_6mo",
    "rolling_avg_num_transactions_6mo",
    "bank_deposit_share",
    "rolling_avg_interest_rate_change_6mo",
]
# %%
feature_names = [col for col in feature_names if col not in highly_correlated_features]
feature_names = [
    col for col in feature_names if col not in ["avg_price_per_m2", "price_per_m2_nombre"]
]

# %%
feature_names

# %%
numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
import json

with open("features_used.json", "w") as f:
    json.dump(feature_names, f)

with open("categorical_features_used.json", "w") as f:
    json.dump(categorical_features, f)

# %%
X = X[feature_names]
