import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/patient_disease_records.csv")
OUTPUT = Path("data/processed/patient_clustering_features.csv")

totals = None
gender_map = {}

print("Reading diagnosis records in chunks...")

for i, chunk in enumerate(
    pd.read_csv(
        INPUT,
        chunksize=100000,
        dtype={
            "patient_id": "string",
            "gender": "string",
            "disease_category": "string",
            "diagnosis_code": "string"
        }
    )
):
    chunk["age"] = pd.to_numeric(chunk["age"], errors="coerce")

    stats = chunk.groupby("patient_id").agg(
        age_sum=("age", "sum"),
        age_count=("age", "count"),
        diagnosis_count=("diagnosis_code", "size")
    )

    category_counts = pd.crosstab(
        chunk["patient_id"],
        chunk["disease_category"]
    )

    partial = stats.join(category_counts)

    if totals is None:
        totals = partial
    else:
        totals = totals.add(partial, fill_value=0)

    genders = (
        chunk.dropna(subset=["gender"])
        .drop_duplicates("patient_id")
        .set_index("patient_id")["gender"]
    )

    for patient, gender in genders.items():
        gender_map.setdefault(patient, gender)

    print(f"Processed chunk {i + 1}")

print("\nCreating patient-level features...")

features = pd.DataFrame(index=totals.index)

features["age"] = totals["age_sum"] / totals["age_count"]
features["diagnosis_count"] = totals["diagnosis_count"]

category_columns = [
    col for col in totals.columns
    if col not in ["age_sum", "age_count", "diagnosis_count"]
]

for col in category_columns:
    features[col] = totals[col]

gender_series = pd.Series(gender_map).reindex(features.index).fillna("Unknown")
gender_features = pd.get_dummies(
    gender_series,
    prefix="gender",
    dtype=int
)

features = pd.concat([features, gender_features], axis=1)

features.index.name = "patient_id"
features = features.reset_index()

features.to_csv(OUTPUT, index=False)

print("\nFEATURE DATASET CREATED")
print("Patients:", len(features))
print("Features:", len(features.columns) - 1)
print("Output:", OUTPUT)
print("\nFirst five rows:")
print(features.head())
