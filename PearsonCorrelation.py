import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# -----------------------------
# Utility Functions
# -----------------------------

def fill_missing(df):
    """Forward fill missing values."""
    return df.ffill()


def normalize_dataframe(df):
    """Min-Max normalization."""
    return (df - df.min()) / (df.max() - df.min())


#Ensure pearson corrolation applied only on training set
def prepare_company_dataframe(df, start_date="2001-12-31", end_date="2015-12-31"):
    """
    Reindex dataframe to a fixed daily date range
    and fill missing values.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.reindex(date_range)
    df = fill_missing(df)
    return df


# -----------------------------
# Feature Selection
# -----------------------------

def get_correlated_features(file_path, target_column="Close", threshold=0.5):
    """
    Select features correlated with the target variable using Pearson correlation.
    """

    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    df = prepare_company_dataframe(df)

    target = df[target_column]

    correlated_features = []

    for column in df.columns:

        if column == target_column:
            continue

        corr, _ = pearsonr(df[column], target)

        if abs(corr) >= threshold:
            correlated_features.append(column)

    return correlated_features


# -----------------------------
# Companies Paths Examples (Could be applied on any company)
# -----------------------------

companies = {
    "AAPL": "/.../data/AAPL.csv",
    "BAC": "/.../data/BAC.csv",
    "JNJ": "/.../data/JNJ.csv",
    "GOOG": "/.../data/GOOG.csv"
}


# -----------------------------
# Run Feature Selection
# -----------------------------

company_features = {}

for company, path in companies.items():

    features = get_correlated_features(path)

    company_features[company] = set(features)

    print(f"{company} selected features: {len(features)}")


# -----------------------------
# Find Common Features
# -----------------------------

common_features = set.intersection(*company_features.values())

print("\nCommon Features Across All Companies:")
print(common_features)
print("Total:", len(common_features))


# -----------------------------
# Compare Feature Sets
# -----------------------------

def compare_feature_sets(set1, set2):

    print("\nCommon Features:")
    print(set1.intersection(set2))

    print("\nOnly in Set1:")
    print(set1 - set2)

    print("\nOnly in Set2:")
    print(set2 - set1)