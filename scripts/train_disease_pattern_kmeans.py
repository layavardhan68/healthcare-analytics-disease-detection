import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/processed/patient_clustering_features.csv")

category_cols = [
    c for c in df.columns
    if c not in ["patient_id", "age", "diagnosis_count"]
    and not c.startswith("gender_")
]

gender_cols = [c for c in df.columns if c.startswith("gender_")]

# Convert category counts into proportions.
proportions = df[category_cols].div(
    df["diagnosis_count"].replace(0, np.nan),
    axis=0
).fillna(0)

# Use disease proportions, age, and gender.
X = pd.concat(
    [df[["age"]], proportions, df[gender_cols]],
    axis=1
)

print("Patients:", len(X))
print("Features:", len(X.columns))
print("Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rng = np.random.default_rng(42)
sample_indices = rng.choice(
    len(X_scaled),
    size=min(5000, len(X_scaled)),
    replace=False
)
X_sample = X_scaled[sample_indices]

results = []
best_score = -1
best_model = None
best_k = None

print("\nTesting cluster counts...")

for k in range(2, 9):
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=4096,
        n_init=3,
        max_iter=100
    )

    model.fit(X_scaled)
    sample_labels = model.predict(X_sample)
    score = silhouette_score(X_sample, sample_labels)

    print(f"K={k}, silhouette={score:.4f}")

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": score
    })

    if score > best_score:
        best_score = score
        best_model = model
        best_k = k

print("\nBest K:", best_k)
print("Best silhouette:", round(best_score, 4))

labels = best_model.predict(X_scaled)

result = pd.DataFrame({
    "patient_id": df["patient_id"],
    "cluster": labels
})

Path("models").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

result.to_csv("output/disease_pattern_clusters.csv", index=False)
pd.DataFrame(results).to_csv(
    "output/disease_pattern_evaluation.csv",
    index=False
)

joblib.dump(scaler, "models/disease_pattern_scaler.pkl")
joblib.dump(best_model, "models/disease_pattern_kmeans.pkl")

print("\nCLUSTER SIZES:")
print(result["cluster"].value_counts().sort_index())

print("\nSaved disease-pattern clustering results.")
