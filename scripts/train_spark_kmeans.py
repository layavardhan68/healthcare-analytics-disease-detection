from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("Healthcare-Spark-MLlib-KMeans")
    .master("local[2]")
    .config("spark.driver.memory", "3g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

INPUT = "hdfs:///user/anits-csm/healthcare-analytics/features/patient_clustering_features.csv"

print("\n===== LOADING PATIENT FEATURES =====")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(INPUT)
)

feature_columns = [c for c in df.columns if c != "patient_id"]

df = df.fillna(0, subset=feature_columns)

print("Patient count:", df.count())
print("Feature count:", len(feature_columns))

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

assembled = assembler.transform(df)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(assembled)
scaled = scaler_model.transform(assembled).cache()

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

results = []
best_score = float("-inf")
best_model = None
best_predictions = None
best_k = None

print("\n===== SPARK MLlib K-MEANS EVALUATION =====")

for k in range(2, 9):
    print(f"\nTraining K={k}...")

    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="features",
        predictionCol="prediction",
        maxIter=50,
        tol=0.0001
    )

    model = kmeans.fit(scaled)
    predictions = model.transform(scaled)

    score = evaluator.evaluate(predictions)

    print(f"K={k}, Silhouette={score:.4f}")

    results.append((k, float(score), float(model.summary.trainingCost)))

    if score > best_score:
        best_score = score
        best_model = model
        best_predictions = predictions
        best_k = k

print("\n===== BEST SPARK MODEL =====")
print("Best K:", best_k)
print("Best silhouette:", round(best_score, 4))

print("\n===== CLUSTER SIZES =====")
best_predictions.groupBy("prediction").count().orderBy("prediction").show()

print("\n===== SAVING RESULTS =====")

import os
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

best_predictions.select(
    "patient_id", "prediction"
).write.mode("overwrite").option("header", "true").csv(
    "output/spark_patient_clusters"
)

best_model.write().overwrite().save("models/spark_kmeans_model")
scaler_model.write().overwrite().save("models/spark_scaler_model")

import csv

with open("output/spark_kmeans_evaluation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "silhouette", "training_cost"])
    writer.writerows(results)

print("Saved cluster assignments: output/spark_patient_clusters/")
print("Saved K-Means model: models/spark_kmeans_model/")
print("Saved evaluation: output/spark_kmeans_evaluation/")

scaled.unpersist()
spark.stop()

print("\n===== SPARK MLlib COMPLETED SUCCESSFULLY =====")
