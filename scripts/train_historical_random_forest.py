import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE = Path("data/raw/STG_EHP_DATASET")
OUT = Path("output")
MODELS = Path("models")

OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

print("Loading visit and patient data...")

visits = pd.read_csv(
    BASE / "STG_EHP__VIST.csv",
    usecols=["PAT_ID", "REFR_NO", "VIS_EN", "VSTAT_DES", "VTYPE_DES"],
    dtype={"PAT_ID": str, "REFR_NO": str}
)

patients = pd.read_csv(
    BASE / "STG_EHP__PATN.csv",
    usecols=["PAT_ID", "GEN_DES", "DT_BRT"],
    dtype={"PAT_ID": str}
).drop_duplicates("PAT_ID")

df = visits.merge(patients, on="PAT_ID", how="left")

df["VIS_EN"] = pd.to_datetime(df["VIS_EN"], errors="coerce")
df["DT_BRT"] = pd.to_datetime(df["DT_BRT"], errors="coerce")
df = df.dropna(subset=["PAT_ID", "VIS_EN", "VSTAT_DES"])
df = df.sort_values(["VIS_EN", "PAT_ID"]).reset_index(drop=True)

df["age"] = (
    (df["VIS_EN"] - df["DT_BRT"]).dt.days / 365.25
)
df.loc[~df["age"].between(0, 110), "age"] = np.nan

df["visit_month"] = df["VIS_EN"].dt.month
df["visit_weekday"] = df["VIS_EN"].dt.dayofweek
df["visit_hour"] = df["VIS_EN"].dt.hour

# Chronological 80:20 split
cutoff = df["VIS_EN"].quantile(0.80)

train = df[df["VIS_EN"] < cutoff].copy()
test = df[df["VIS_EN"] >= cutoff].copy()

print("Training records:", len(train))
print("Testing records:", len(test))
print("Date cutoff:", cutoff)

# Historical features for training: previous visits only
train = train.sort_values(["PAT_ID", "VIS_EN"])

train["previous_visit_count"] = (
    train.groupby("PAT_ID").cumcount()
)

train["previous_admission_count"] = (
    train["VSTAT_DES"].eq("Admitted")
    .groupby(train["PAT_ID"])
    .transform(lambda s: s.shift().cumsum())
)

train["days_since_previous_visit"] = (
    train.groupby("PAT_ID")["VIS_EN"]
    .diff().dt.total_seconds() / 86400
)

# Test history uses training-period records only
history = train.groupby("PAT_ID").agg(
    previous_visit_count=("REFR_NO", "count"),
    previous_admission_count=(
        "VSTAT_DES",
        lambda s: s.eq("Admitted").sum()
    ),
    last_visit=("VIS_EN", "max")
)

test = test.join(
    history,
    on="PAT_ID",
    rsuffix="_history"
)

test["previous_visit_count"] = (
    test["previous_visit_count"].fillna(0)
)
test["previous_admission_count"] = (
    test["previous_admission_count"].fillna(0)
)
test["days_since_previous_visit"] = (
    (test["VIS_EN"] - test["last_visit"]).dt.total_seconds() / 86400
)
test["days_since_previous_visit"] = (
    test["days_since_previous_visit"].fillna(-1)
)

numeric_features = [
    "age",
    "visit_month",
    "visit_weekday",
    "visit_hour",
    "previous_visit_count",
    "previous_admission_count",
    "days_since_previous_visit"
]

categorical_features = ["GEN_DES", "VTYPE_DES"]

X_train = train[numeric_features + categorical_features]
y_train = train["VSTAT_DES"]

X_test = test[numeric_features + categorical_features]
y_test = test["VSTAT_DES"]

print("\nTraining outcome distribution:")
print(y_train.value_counts())

print("\nTesting outcome distribution:")
print(y_test.value_counts())

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        max_samples=0.7,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining historical Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

majority_class = y_train.value_counts().idxmax()
baseline_predictions = np.full(len(y_test), majority_class)

def calculate_metrics(name, actual, predicted):
    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_precision": precision_score(
            actual, predicted, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            actual, predicted, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            actual, predicted, average="macro", zero_division=0
        )
    }

results = pd.DataFrame([
    calculate_metrics("Majority Baseline", y_test, baseline_predictions),
    calculate_metrics("Historical Random Forest", y_test, predictions)
])

results.to_csv(
    OUT / "historical_random_forest_metrics.csv", index=False
)

labels = model.classes_
cm = confusion_matrix(y_test, predictions, labels=labels)

pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(OUT / "historical_random_forest_confusion_matrix.csv")

joblib.dump(
    model,
    MODELS / "historical_random_forest.joblib"
)

print("\n========== MODEL RESULTS ==========")
print(results.to_string(index=False))

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion matrix:")
print(pd.DataFrame(cm, index=labels, columns=labels))

print("\nSaved metrics and confusion matrix in output/")
print("Saved model in models/")
