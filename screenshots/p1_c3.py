import polars as pl
import os
from settings import PROJECT_PATH
import matplotlib.pyplot as plt
import seaborn as sns

transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "real_estate_transactions_engineered.parquet")
)

feature_names = [col for col in transactions.columns if col != "transaction_id"]

# %%
categorical_features = [col for col in feature_names if "building_type" in col]
categorical_features.append("off_plan")
categorical_features.extend([col for col in feature_names if "region_name" in col])

# %%
numerical_features = [col for col in feature_names if col not in categorical_features]
numerical_features.remove("department_name")
# %%
correlation_matrix = transactions.to_pandas()[numerical_features].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


highly_correlated_features = [
    "building_type_house",     # Keep one or the other
    "num_rooms",                 # Keep living area instead
    # "transaction_year",      # Keep debt ratio instead
    "euros_per_capita",       # Highly correlated with debt ratio
    "montant_impot_moyen",      # Highly correlated with average taxable income
    "life_insurance_share",
    "mutual_fund_share",
    "non_equity_securities_share",
    "equity_share",
    "taux",
    "IRL",
]
