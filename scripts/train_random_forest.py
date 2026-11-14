import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE = Path(__file__).resolve().parents[1]

DATA = BASE / "data/processed/patient_disease_records.csv"
OUTPUT = BASE / "output"
MODELS = BASE / "models"

OUTPUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

print("Loading healthcare dataset...")

df = pd.read_csv(
    DATA,
    usecols=[
        "patient_id",
        "age",
        "gender",
        "disease_category",
        "visit_status"
    ]
)

print("Original records:", len(df))

# Use a reproducible sample to control memory and training time.
df = df.sample(
    n=min(250000, len(df)),
    random_state=42
).reset_index(drop=True)

print("Sample records:", len(df))
print("Unique patients:", df["patient_id"].nunique())

# Input features and target
X = df[["age", "gender", "disease_category"]]
y = df["visit_status"]
groups = df["patient_id"]

# Split by patient to prevent patient-level data leakage.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_patients = set(groups.iloc[train_idx])
test_patients = set(groups.iloc[test_idx])

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Patient overlap:", len(train_patients & test_patients))

# Preprocess categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         ["gender", "disease_category"])
    ],
    remainder="passthrough"
)

# Baseline model
baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

print("\nTraining baseline model...")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

# Random Forest model
random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
random_forest.fit(X_train, y_train)

predictions = random_forest.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, average="macro"
)

recall = recall_score(
    y_test, predictions, average="macro"
)

f1 = f1_score(
    y_test, predictions, average="macro"
)

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Random Forest Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Macro F1-score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save evaluation metrics
metrics = pd.DataFrame([
    ["Baseline", baseline_accuracy],
    ["Random Forest", accuracy]
], columns=["Model", "Accuracy"])

metrics.to_csv(
    OUTPUT / "random_forest_metrics.csv",
    index=False
)

# Save confusion matrix
labels = random_forest.named_steps["classifier"].classes_

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(
    OUTPUT / "random_forest_confusion_matrix.csv"
)

# Save model
joblib.dump(
    random_forest,
    MODELS / "random_forest_visit_status.pkl"
)

print("\nModel saved:", MODELS / "random_forest_visit_status.pkl")
print("Metrics saved:", OUTPUT / "random_forest_metrics.csv")
print("Confusion matrix saved.")

print("\nRandom Forest experiment completed!")
