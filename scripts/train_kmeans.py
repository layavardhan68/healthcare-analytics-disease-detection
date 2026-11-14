import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

INPUT = "data/processed/patient_clustering_features.csv"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading patient features...")
df = pd.read_csv(INPUT)

patient_ids = df["patient_id"]
X = df.drop(columns=["patient_id"])

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use a reproducible sample for silhouette evaluation.
rng = np.random.default_rng(42)
sample_size = min(5000, len(X_scaled))
sample_indices = rng.choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[sample_indices]

results = []
best_score = -1
best_model = None
best_k = None

print("\nTraining MiniBatchKMeans models...")

for k in range(2, 9):
    print(f"\nTesting K = {k}")

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

    print("Inertia:", round(model.inertia_, 2))
    print("Silhouette score:", round(score, 4))

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
print("Best silhouette score:", round(best_score, 4))

print("\nAssigning clusters to all patients...")
labels = best_model.predict(X_scaled)

result = pd.DataFrame({
    "patient_id": patient_ids,
    "cluster": labels
})

result.to_csv(OUTPUT_DIR / "patient_clusters.csv", index=False)

pd.DataFrame(results).to_csv(
    OUTPUT_DIR / "kmeans_evaluation.csv",
    index=False
)

joblib.dump(scaler, MODEL_DIR / "patient_scaler.pkl")
joblib.dump(best_model, MODEL_DIR / "best_kmeans_model.pkl")

print("\nCLUSTER SIZES:")
print(result["cluster"].value_counts().sort_index())

print("\nSaved:")
print("output/patient_clusters.csv")
print("output/kmeans_evaluation.csv")
print("models/patient_scaler.pkl")
print("models/best_kmeans_model.pkl")
