# %%
import polars as pl
import os
import sys

sys.path.append("..")
from settings import (
    PROJECT_PATH,
    DEBT_RATIO_FILE_PATH,
    FINANCIAL_ASSETS_FILE_PATH,
    INTEREST_RATE_FILE_PATH,
    NEW_LOANS_FILE_PATH,
    RENT_REFERENCE_INDEX_FILE_PATH,
)

from data_processing_functions import (
    load_annual_macro_eco_context_data,
    load_monthly_macro_eco_context_data,
    add_economical_context_features,
)
from feature_engineering_functions import (
    compute_city_features,
    compute_price_per_m2_features,
    create_debt_ratio_features,
    calculate_interest_rate_features,
)

# %%


transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "real_estate_transactions.parquet")
)

transactions = transactions.with_columns(
    pl.col("department").cast(pl.Int32),
    pl.col("transaction_month").cast(pl.Int32),
    pl.col("off_plan").cast(pl.Int32),
)

transactions_per_city = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_by_city.parquet")
)

transactions_per_city = transactions_per_city.with_columns(
    pl.col("department").cast(pl.Int32),
    pl.col("transaction_month").cast(pl.Int32),
)
# %%
transactions.head()

# %%
# -------------- Feature creation from a threshold on another feature ---------------

# %%
transactions_per_city.head()

# %%

transactions_per_city = compute_city_features(
    transactions_per_city,
    feature_name="city_requested",
    quantile_threshold=0.8,
    verbose=True,
)


# %%
transactions_per_city["city_requested"].describe()

# %%

# ---------------- Feature creation using lagging --------------

transactions_per_city
# %%

transactions_per_city = compute_price_per_m2_features(
    transactions_per_city,
    sort_columns=[
        "department",
        "city",
        "city_id",
        "transaction_year",
        "transaction_month",
    ],
)

# %%

transactions = transactions.join(
    transactions_per_city,
    on=[
        "department",
        "city",
        "city_id",
        "transaction_year",
        "transaction_month",
        "avg_price_per_m2",
        "num_transactions_month",
    ],
    how="inner",
)

# %%
# ------------ Feature creation representing short-term variations -------
annual_macro_eco_context = load_annual_macro_eco_context_data(
    DEBT_RATIO_FILE_PATH, FINANCIAL_ASSETS_FILE_PATH
)

# %%
annual_macro_eco_context
# %%

annual_macro_eco_context = create_debt_ratio_features(annual_macro_eco_context)


# %%
# ------------ Feature creation representing longer-term trends -------
monthly_macro_eco_context = load_monthly_macro_eco_context_data(
    INTEREST_RATE_FILE_PATH, NEW_LOANS_FILE_PATH, RENT_REFERENCE_INDEX_FILE_PATH
)

monthly_macro_eco_context = monthly_macro_eco_context.with_columns(
    pl.col("mois").cast(pl.Int32)
)

# %%
monthly_macro_eco_context
# %%

monthly_macro_eco_context = calculate_interest_rate_features(
    monthly_macro_eco_context,
    aggregation_period="6mo",
    interest_rate_col="interest_rate",
)


# %%

transactions = add_economical_context_features(
    transactions,
    annual_macro_eco_context,
    monthly_macro_eco_context,
)


# %%
# ----------------- One Hot Encoding -------------------------
transactions = transactions.to_dummies(columns=["building_type"])
transactions = transactions.to_dummies(columns=["region_name"])

# %%
transactions

# %%
transactions.write_parquet(
    os.path.join(PROJECT_PATH, "real_estate_transactions_engineered.parquet")
)


# %%

"""
We build a DataFrame with "extra" information kept separate from transactions.
This info can be useful for interpretation purposes.
"""

cols_extra_info = [
    "transaction_id",
    "transaction_date",
    "city_id",
    "city",
    "department_code",
    "department",
    "region_code",
    "region",
    "address",
    "postal_code",
    "cadastral_parcel_id",
    "price_per_m2",          # Excluded to avoid data leakage
    "avg_price_per_m2",    # Excluded to avoid data leakage
    "department_name",  # Excluded for now
]

transactions_metadata = transactions.select(
    [e for e in cols_extra_info if e in transactions.columns]
)

cols_to_drop = [
    "USD_par_habitant",
    "date",
    "city_right",
]
cols_to_drop.extend(cols_extra_info)
cols_to_drop.remove("transaction_id")

transactions = transactions.drop(
    [col for col in transactions.columns if col in cols_to_drop]
)


# %%
transactions
# %%

transactions = transactions.filter(
    pl.all_horizontal(pl.col(pl.Float32, pl.Float64, pl.Int32, pl.Int64).is_not_nan())
)

# %%


# %%
