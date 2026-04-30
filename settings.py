import os

# ---------------------- Filepaths ---------------------
# Using environment variables to avoid hardcoding the data path
PROJECT_PATH = os.environ.get("SUPERVISED_LEARNING_PROJECT_PATH")
TRANSACTIONS_FILE_PATH = os.path.join(PROJECT_PATH, "transactions.npz")
TAX_HOUSEHOLDS_FILE_PATH = os.path.join(PROJECT_PATH, "tax_households.csv")
DEBT_RATIO_FILE_PATH = os.path.join(PROJECT_PATH, "debt_ratio.csv")
FINANCIAL_ASSETS_FILE_PATH = os.path.join(PROJECT_PATH, "financial_assets.csv")
INTEREST_RATE_FILE_PATH = os.path.join(PROJECT_PATH, "interest_rate.csv")
NEW_LOANS_FILE_PATH = os.path.join(PROJECT_PATH, "new_loans.csv")
REGIONS_FILE_PATH = os.path.join(PROJECT_PATH, "departments-france.csv")
RENT_REFERENCE_INDEX_FILE_PATH = os.path.join(PROJECT_PATH, "rent_reference_index.csv")


# ---------------------- Raw Column names ------------------
TRANSACTION_DATE = "transaction_date"
TRANSACTION_MONTH = "transaction_month"
TRANSACTION_YEAR = "transaction_year"
DEPARTEMENT = "department"
REGION = "region"
CITY_UNIQUE_ID = ["department", "city", "city_id"]
SURFACE = "living_area"
PRICE_PER_SQUARE_METER = "price_per_m2"

# ---------------------- Feature & Target Column names ------------------
AVERAGE_PRICE_PER_SQUARE_METER = "avg_price_per_m2"
NB_TRANSACTIONS_PER_MONTH = "num_transactions_month"
VEFA = "off_plan"

REGRESSION_TARGET = "price"
CLASSIFICATION_TARGET = "below_market"

random_state = 42
