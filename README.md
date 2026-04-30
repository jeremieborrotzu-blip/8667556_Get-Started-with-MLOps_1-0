# Master Supervised Learning


## Context

This repository centralizes all resources (screencast code, exercise starters and solutions) for the running project associated with the course "Master Supervised Learning". The folder structure is as follows:

* **exercises:** Starter and solution notebooks for each chapter of the course
* **screencasts:** Code presented during each screencast
* **screenshots:** Code used to generate certain screenshots in the course


## How to use this repository

### Downloading the data

You can download all the datasets you will need via [this zip file](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/8667556/data_original_raw.zip). It contains:
* **transactions_immobilieres.parquet**: the main dataset used throughout Part 1
* **transactions_par_ville.parquet**: a dataset with city-level aggregated data, used across all parts of the course for feature and target computation
* **transactions_post_feature_engineering.parquet**: the dataset used in Parts 2 and 3, where feature engineering is considered complete
* **transactions_extra_infos.parquet**: a dataset containing observation metadata that cannot be used as features (id_transaction, date_transaction) but is useful for exploratory analysis. Use it alongside transactions_post_feature_engineering.parquet
* **features_used.json**: a list file containing all features used in the modeling phase
* **categorical_features_used.json**: a subset of the previous file, containing only categorical features

The datasets are either in parquet format (more convenient than CSV and compatible with both Pandas and Polars) or in JSON format for feature lists.

If you want to rebuild the transactions_immobilieres.parquet file from scratch, you can start from the raw data in [this zip file](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/8667556/data_en.zip) and use the preprocessing.py script.

To keep the repository lightweight, the data files and models saved via MLflow have not been included here. You have the code available to rebuild them.

### Important notes about the data

Each city in the dataset can be uniquely identified by a combination of the columns: id_ville, ville, and departement.

Regarding the classification target column, it is computed as follows:
* The average price per m2 is computed at the department level for a given month
* For each transaction, if its price per m2 is more than 10% below that average, the target value is 1
* Otherwise, the target value is 0


### How to use the code

**This repo uses Poetry for package management and virtual environment creation.** The pyproject.toml file contains all package versions used in the course.

Due to dependency conflicts with MLflow, the BentoML package (used in Part 3, Chapter 2) is not listed among the installed dependencies. However, you will find the exact version used for the screencast and exercise code as a comment in the pyproject.toml file.

For the exercises, we recommend storing all datasets in the same location and pointing to that path **via an environment variable** (as done in the settings.py file). You can implement this by following [this tutorial](https://www.youtube.com/watch?v=IolxqkL7cD8&t=201s) for example.
