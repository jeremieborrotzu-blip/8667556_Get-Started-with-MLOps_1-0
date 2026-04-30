import polars as pl
from settings import (
    NB_TRANSACTIONS_PER_MONTH,
    DEPARTEMENT,
    TRANSACTION_YEAR,
    TRANSACTION_MONTH,
    AVERAGE_PRICE_PER_SQUARE_METER,
    CITY_UNIQUE_ID,
)


def compute_city_features(
    transactions_per_city: pl.DataFrame,
    feature_name: str = "city_requested",
    grouping_columns: list = [DEPARTEMENT, TRANSACTION_YEAR, TRANSACTION_MONTH],
    quantile_threshold=0.8,
    verbose: bool = False,
):
    nb_transactions_per_department = transactions_per_city.group_by(grouping_columns).agg(
        pl.sum(NB_TRANSACTIONS_PER_MONTH).alias("nb_transactions_department")
    )

    transactions_per_city = transactions_per_city.join(
        nb_transactions_per_department,
        on=grouping_columns,
        how="left",
    )

    transactions_per_city = transactions_per_city.with_columns(
        (
            100
            * pl.col(NB_TRANSACTIONS_PER_MONTH)
            / pl.col("nb_transactions_department")
        ).alias("city_transaction_ratio")
    ).drop("nb_transactions_department")

    if verbose:
        print(transactions_per_city.select("city_transaction_ratio").describe())
    else:
        pass

    transactions_per_city = transactions_per_city.with_columns(
        pl.when(
            pl.col("city_transaction_ratio")
            > pl.quantile("city_transaction_ratio", quantile_threshold)
        )
        .then(1)
        .otherwise(0)
        .alias(feature_name),
    ).drop("city_transaction_ratio")

    return transactions_per_city


def create_debt_ratio_features(
    annual_macro_eco_context: pl.DataFrame,
    debt_ratio_col: str = "debt_ratio",
):
    # Simple cast used for a join downstream
    annual_macro_eco_context = annual_macro_eco_context.with_columns(
        pl.col("date").cast(pl.Int32),
        pl.col(debt_ratio_col).diff().alias("variation_" + debt_ratio_col),
        pl.col(debt_ratio_col).diff().diff().alias("acceleration_" + debt_ratio_col),
    )

    return annual_macro_eco_context


def calculate_interest_rate_features(
    monthly_macro_eco_context: pl.DataFrame,
    aggregation_period: str,
    interest_rate_col: str,
):
    monthly_macro_eco_context = monthly_macro_eco_context.with_columns(
        pl.col(interest_rate_col).diff().name.prefix("variation_"),
        pl.col(interest_rate_col).diff().diff().name.prefix("acceleration_"),
    ).with_columns(
        pl.mean("variation_" + interest_rate_col)
        .rolling(index_column="date", period=aggregation_period)
        .name.prefix("moyenne_glissante_" + aggregation_period + "_"),
    )

    return monthly_macro_eco_context


def compute_price_per_m2_features(
    average_per_month_per_city: pl.DataFrame,
    sort_columns: list = CITY_UNIQUE_ID,
    aggregation_period="6mo",
):
    average_per_month_per_city = (
        average_per_month_per_city.sort(sort_columns)
        .with_columns(
            pl.col(AVERAGE_PRICE_PER_SQUARE_METER)
            .shift()
            .over(CITY_UNIQUE_ID)
            .alias("avg_price_per_m2_previous_month"),
            pl.col(NB_TRANSACTIONS_PER_MONTH)
            .shift()
            .over(CITY_UNIQUE_ID)
            .alias("num_transactions_previous_month"),
            pl.col(AVERAGE_PRICE_PER_SQUARE_METER)
            .rolling_mean(window_size=6)
            .over(CITY_UNIQUE_ID)
            .alias("avg_price_per_m2_glissant_" + aggregation_period),
            pl.col(NB_TRANSACTIONS_PER_MONTH)
            .rolling_mean(window_size=6)
            .over(CITY_UNIQUE_ID)
            .alias("nb_transaction_moyen_glissant_" + aggregation_period),
        )
        .filter(
            pl.all_horizontal(
                pl.col(pl.Float32, pl.Float64, pl.Int32, pl.Int64).is_not_nan()
            )
        )
    )

    return average_per_month_per_city


"""
VERSION OF THE FUNCTION USED IN THE SCREENCAST P1_C3_FEATURE_ENGINEERING

def compute_price_per_m2_features(
    average_per_month_per_city: pl.DataFrame,
    sort_columns: list = [
        "department",
        "city",
        "city_id",
    ],
):
    average_per_month_per_city = (
        average_per_month_per_city.sort(sort_columns)
        .with_columns(
            pl.col("avg_price_per_m2")
            .shift()
            .over(["department", "city", "city_id"])
            .alias("avg_price_per_m2_previous_month"),
            pl.col(NB_TRANSACTIONS_PER_MONTH)
            .shift()
            .over(["department", "city", "city_id"])
            .alias("num_transactions_previous_month"),
        )
        .filter(
            pl.all_horizontal(
                pl.col(pl.Float32, pl.Float64, pl.Int32, pl.Int64).is_not_nan()
            )
        )
    )

    return average_per_month_per_city
"""
