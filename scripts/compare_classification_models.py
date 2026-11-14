import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data/processed/patient_disease_records.csv"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

print("Loading healthcare dataset...")

df = pd.read_csv(
    DATA,
    usecols=["patient_id", "age", "gender", "disease_category", "visit_status"]
)

df = df.sample(n=min(250000, len(df)), random_state=42).reset_index(drop=True)

X = df[["age", "gender", "disease_category"]]
y = df["visit_status"]
groups = df["patient_id"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Patient overlap:", len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])))

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     ["gender", "disease_category"])
], remainder="passthrough")

models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(
        max_iter=300, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=16, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced,
        "Macro_F1": macro_f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

results_df = pd.DataFrame(results).sort_values(
    by="Accuracy", ascending=False
)

results_df.to_csv(
    OUTPUT / "classification_model_comparison.csv", index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))
print("\nResults saved to output/classification_model_comparison.csv")
