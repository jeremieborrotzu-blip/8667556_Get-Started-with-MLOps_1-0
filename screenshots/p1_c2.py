# %%

# ----------------------------- Screenshot Linear Regression 2D ----------------------------- #

REGRESSION_TARGET = "price"
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import polars as pl


"""
We restrict the scope to the screenshot's data range.
Here we use Polars syntax. The Pandas equivalent would be:

transactions_regression_2D = transactions[
    (transactions["department"] == 75)
    & (transactions[REGRESSION_TARGET] >= transactions[REGRESSION_TARGET].quantile(0.1))
][["living_area", REGRESSION_TARGET]]
"""

transactions_regression_2D = transactions.filter(
    pl.col("department") == 75,
    pl.col(REGRESSION_TARGET) >= pl.quantile(REGRESSION_TARGET, 0.1),
).select(["living_area", REGRESSION_TARGET])

# Train a linear regression model to explain the target from the chosen feature
linear_regressor_2D = LinearRegression()

"""
At the time this code was written, Scikit-learn does not support Polars DataFrames directly.
A conversion to a numpy array is required, as numpy is the core format for most Scikit-learn operations.
"""
linear_regressor_2D.fit(
    transactions_regression_2D["living_area"].to_numpy().reshape(-1, 1),
    transactions_regression_2D[REGRESSION_TARGET].to_numpy(),
)

# Scatter plot of all properties
plt.scatter(
    transactions_regression_2D["living_area"],
    transactions_regression_2D[REGRESSION_TARGET],
    label="Properties",
)

# Generate dummy points to draw the regression line
living_area_range = np.linspace(
    transactions_regression_2D["living_area"].min(),
    transactions_regression_2D["living_area"].max(),
    100,
).reshape(-1, 1)
predictions = linear_regressor_2D.predict(living_area_range)

# Add the regression line
plt.plot(living_area_range, predictions, color="red", label="Regression Line")
plt.xlabel("Living Area")
plt.ylabel(REGRESSION_TARGET)
plt.title("Relationship between Living Area and Transaction Price")
plt.legend()
plt.show()

# %%
# ---------------------- Screenshot Logistic Regression Decision Boundary ---------------------- #

CLASSIFICATION_TARGET = "below_market"
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression

# We choose these 2 features only to illustrate the concept
transactions_classification_3D = transactions.filter(
    pl.col("department") == 4,
    pl.col(REGRESSION_TARGET) >= pl.quantile(REGRESSION_TARGET, 0.1),
).select(["living_area", REGRESSION_TARGET, CLASSIFICATION_TARGET])

# Train the logistic regression model
logistic_regressor = LogisticRegression()
X = transactions_classification_3D.select(["living_area", "price"]).to_numpy()
y = transactions_classification_3D[CLASSIFICATION_TARGET].to_numpy()
logistic_regressor.fit(X, y)


# Use this library to plot the decision boundary
plot_decision_regions(X, y, clf=logistic_regressor, legend=2)
plt.xlabel("Living Area")
plt.ylabel(REGRESSION_TARGET)
plt.title("Decision Boundary of Logistic Regression")
plt.legend()
plt.show()


# ----------------- Same for the Random Forest model ----------------- #

from sklearn.ensemble import RandomForestClassifier

from mlxtend.plotting import plot_decision_regions

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)

plot_decision_regions(X, y, clf=rf_classifier, legend=2)
plt.xlabel("Living Area")
plt.ylabel("price")
plt.title("Decision Boundary of Random Forest")
plt.legend()
plt.show()
