# %%
import polars as pl
import os

pl.Config(tbl_cols=50)  # This is the equivalent of Pandas' number of columns extension
from settings import (
    PROJECT_PATH,
    TRANSACTIONS_FILE_PATH,
    REGIONS_FILE_PATH,
    NB_TRANSACTIONS_PER_MONTH,
    TRANSACTION_YEAR,
    TRANSACTION_MONTH,
    DEPARTEMENT,
    VEFA,
    CITY_UNIQUE_ID,
)
from datetime import datetime


from data_processing_functions import (
    load_and_process_transactions,
    get_info_per_month_cities_enough_transactions,
    add_classification_target_to_transactions,
    remove_departments_with_few_transactions,
    remove_regions_with_few_transactions,
    load_regions_data,
)


# %%
filtered_transactions = load_and_process_transactions(
    file_path=TRANSACTIONS_FILE_PATH, lower_bound_date=datetime(2018, 1, 1)
)

original_columns = filtered_transactions.columns


# %%
average_per_month_per_city_enough_transactions, average_per_month_per_city = (
    get_info_per_month_cities_enough_transactions(
        filtered_transactions,
        grouping_cols=[
            "department",
            "city",
            "city_id",
            TRANSACTION_YEAR,
            TRANSACTION_MONTH,
        ],
        verbose=False,
    )
)

# %%


# %%
"""
EXPLORATORY
This analysis shows that only 260 cities have more than 2 real-estate transactions per month.
Useful in Part 1 Chapter 4 to illustrate that the model can only work in months
where enough data is available.
"""

min_nb_transaction_per_city = average_per_month_per_city.group_by(CITY_UNIQUE_ID).agg(
    pl.min(NB_TRANSACTIONS_PER_MONTH)
)

min_nb_transaction_per_city.filter(
    pl.col(NB_TRANSACTIONS_PER_MONTH) > pl.quantile(NB_TRANSACTIONS_PER_MONTH, 0.75)
).select(NB_TRANSACTIONS_PER_MONTH).describe()

"""
END EXPLORATORY
"""

# %%
filtered_transactions = filtered_transactions.join(
    average_per_month_per_city_enough_transactions,
    on=["department", "city", "city_id", TRANSACTION_YEAR, TRANSACTION_MONTH],
    how="inner",
)


# %%

# average_per_month_per_city_enough_transactions = average_per_month_per_city_enough_transactions.drop("city")
filtered_transactions = add_classification_target_to_transactions(
    filtered_transactions, "below_market", 0.1
)


# %%
filtered_transactions, departments_to_keep = remove_departments_with_few_transactions(
    filtered_transactions, threshold_percentile=0.25, verbose=False
)

# %%
filtered_transactions = filtered_transactions.with_columns(
    pl.col(TRANSACTION_YEAR).cast(pl.Int32),
    pl.col(TRANSACTION_MONTH).cast(pl.Int32),
    pl.col(DEPARTEMENT).cast(pl.Int32),
    pl.col(VEFA).cast(pl.Int32),
)

# %%


departments_regions = load_regions_data(REGIONS_FILE_PATH, departments_to_keep)


# %%
filtered_transactions = filtered_transactions.join(
    departments_regions, how="left", on="department"
)


filtered_transactions = remove_regions_with_few_transactions(filtered_transactions)


# %%
filtered_transactions.columns

# %%

filtered_transactions.write_parquet(
    os.path.join(PROJECT_PATH, "real_estate_transactions.parquet")
)

# %%
average_per_month_per_city_enough_transactions.write_parquet(
    os.path.join(PROJECT_PATH, "transactions_by_city.parquet")
)
