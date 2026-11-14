from pathlib import Path
import glob

import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
OUTPUT = BASE / "output"

st.set_page_config(
    page_title="Healthcare Analytics",
    page_icon="🏥",
    layout="wide"
)

st.title("Healthcare Analytics & Disease Pattern Detection")
st.caption("Big Data Analytics using Hadoop, Hive, Spark MLlib and Machine Learning")

@st.cache_data(show_spinner="Loading healthcare data...")
def load_data():
    features = pd.read_csv(DATA / "patient_clustering_features.csv")
    sklearn_clusters = pd.read_csv(OUTPUT / "patient_clusters.csv")

    spark_files = glob.glob(str(OUTPUT / "spark_patient_clusters" / "part-*.csv"))
    spark_clusters = pd.concat(
        [pd.read_csv(file) for file in spark_files],
        ignore_index=True
    )
    spark_clusters = spark_clusters.rename(columns={"prediction": "spark_cluster"})

    features = features.merge(sklearn_clusters, on="patient_id", how="left")
    features = features.merge(
        spark_clusters[["patient_id", "spark_cluster"]],
        on="patient_id",
        how="left"
    )

    sklearn_eval = pd.read_csv(OUTPUT / "kmeans_evaluation.csv")
    spark_eval = pd.read_csv(OUTPUT / "spark_kmeans_evaluation.csv")

    return features, sklearn_eval, spark_eval


@st.cache_data(show_spinner="Analyzing diagnosis records...")
def load_diagnosis_summary():
    file_path = DATA / "patient_disease_records.csv"

    category_counts = {}
    age_counts = {}
    gender_counts = {}
    monthly_counts = {}

    columns = ["disease_category", "age_group", "gender", "diagnosis_date"]
    total = 0

    for chunk in pd.read_csv(file_path, usecols=columns, chunksize=150000):
        total += len(chunk)

        for col, store in [
            ("disease_category", category_counts),
            ("age_group", age_counts),
            ("gender", gender_counts)
        ]:
            counts = chunk[col].value_counts()
            for key, value in counts.items():
                store[key] = store.get(key, 0) + int(value)

        months = pd.to_datetime(
            chunk["diagnosis_date"], errors="coerce"
        ).dt.to_period("M").astype(str).value_counts()

        for key, value in months.items():
            monthly_counts[key] = monthly_counts.get(key, 0) + int(value)

    return (
        pd.Series(category_counts, name="diagnoses").sort_values(ascending=False),
        pd.Series(age_counts, name="diagnoses"),
        pd.Series(gender_counts, name="diagnoses"),
        pd.Series(monthly_counts, name="diagnoses").sort_index(),
        total
    )


try:
    features, sklearn_eval, spark_eval = load_data()
    disease, age, gender, monthly, diagnosis_total = load_diagnosis_summary()
except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.info("Check that the processed data and model output files exist.")
    st.stop()


# Sidebar filters
st.sidebar.header("Dashboard Filters")

available_clusters = sorted(features["cluster"].dropna().unique())
selected_clusters = st.sidebar.multiselect(
    "Scikit-learn cluster",
    available_clusters,
    default=available_clusters
)

filtered = features[features["cluster"].isin(selected_clusters)]

# Overview metrics
st.header("Project Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Patients", f"{len(features):,}")
c2.metric("Diagnosis Records", f"{diagnosis_total:,}")
c3.metric("Disease Categories", f"{len(disease):,}")
c4.metric("Patient Clusters", f"{features['cluster'].nunique()}")

st.divider()

# Disease analytics
st.header("Disease Category Analytics")

left, right = st.columns(2)

with left:
    disease_df = disease.reset_index()
    disease_df.columns = ["Disease Category", "Diagnosis Count"]
    fig = px.bar(
        disease_df,
        x="Diagnosis Count",
        y="Disease Category",
        orientation="h",
        title="Diagnosis Records by Category"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.pie(
        disease.reset_index(),
        names="index",
        values="diagnoses",
        title="Diagnosis Category Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Demographic analytics
st.header("Patient Demographic Analytics")

left, right = st.columns(2)

with left:
    age_df = age.rename_axis("Age Group").reset_index(name="Diagnosis Count")
    fig = px.bar(
        age_df,
        x="Age Group",
        y="Diagnosis Count",
        title="Diagnosis Records by Age Group"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    gender_df = gender.rename_axis("Gender").reset_index(name="Diagnosis Count")
    fig = px.pie(
        gender_df,
        names="Gender",
        values="Diagnosis Count",
        title="Diagnosis Records by Gender"
    )
    st.plotly_chart(fig, use_container_width=True)

# Monthly trend
st.header("Diagnosis Timeline")
monthly_df = monthly.rename_axis("Month").reset_index(name="Diagnosis Count")
fig = px.line(
    monthly_df,
    x="Month",
    y="Diagnosis Count",
    markers=True,
    title="Monthly Diagnosis Records"
)
st.plotly_chart(fig, use_container_width=True)

# Patient clustering
st.header("Patient Clustering Analysis")

left, right = st.columns(2)

with left:
    cluster_counts = (
        filtered["cluster"].value_counts().sort_index()
        .rename_axis("Cluster").reset_index(name="Patients")
    )
    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Patients",
        color="Cluster",
        title="Scikit-learn Patient Cluster Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    spark_counts = (
        features["spark_cluster"].value_counts().sort_index()
        .rename_axis("Spark Cluster").reset_index(name="Patients")
    )
    fig = px.pie(
        spark_counts,
        names="Spark Cluster",
        values="Patients",
        title="Spark MLlib Patient Cluster Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Cluster profile
st.subheader("Cluster Characteristics")

profile_columns = ["age", "diagnosis_count"]
profile = filtered.groupby("cluster")[profile_columns].mean().round(2)
profile.index.name = "Cluster"
st.dataframe(profile, use_container_width=True)


# Disease trend analysis
st.header("Disease Trend Analysis")

trend_file = OUTPUT / "disease_trends" / "monthly_disease_trends.csv"
trends = pd.read_csv(trend_file)
trends["diagnosis_count"] = pd.to_numeric(trends["diagnosis_count"])

categories = ["All Categories"] + sorted(trends["disease_category"].unique())
selected_category = st.selectbox("Select disease category", categories)

if selected_category == "All Categories":
    trend_plot = (
        trends.groupby("month", as_index=False)["diagnosis_count"].sum()
    )
    fig = px.line(
        trend_plot,
        x="month",
        y="diagnosis_count",
        markers=True,
        title="Total Monthly Diagnosis Records"
    )
else:
    trend_plot = trends[trends["disease_category"] == selected_category]
    fig = px.line(
        trend_plot,
        x="month",
        y="diagnosis_count",
        markers=True,
        title=f"Monthly Trend: {selected_category}"
    )

st.plotly_chart(fig, width="stretch")

st.caption(
    "These trends describe diagnosis records in the selected dataset; "
    "they do not establish real-world disease prevalence."
)

# Patient utilization group analysis
st.header("Patient Utilization Group Analysis")

group_file = OUTPUT / "patient_utilization_groups.csv"
groups = pd.read_csv(group_file)

left, right = st.columns(2)

with left:
    group_counts = (
        groups["utilization_group"]
        .value_counts()
        .reindex(["Low", "Moderate", "High"], fill_value=0)
        .rename_axis("Utilization Group")
        .reset_index(name="Patients")
    )
    fig = px.bar(
        group_counts,
        x="Utilization Group",
        y="Patients",
        color="Utilization Group",
        title="Patients by Utilization Group"
    )
    st.plotly_chart(fig, width="stretch")

with right:
    group_profile = (
        groups.groupby("utilization_group")
        .agg(
            patients=("patient_id", "count"),
            average_age=("age", "mean"),
            average_diagnoses=("diagnosis_count", "mean"),
            minimum_diagnoses=("diagnosis_count", "min"),
            maximum_diagnoses=("diagnosis_count", "max")
        )
        .reindex(["Low", "Moderate", "High"])
        .reset_index()
    )
    fig = px.bar(
        group_profile,
        x="utilization_group",
        y="average_diagnoses",
        color="utilization_group",
        title="Average Diagnoses per Patient"
    )
    st.plotly_chart(fig, width="stretch")

st.subheader("Patient Utilization Group Summary")
st.dataframe(group_profile.round(2), width="stretch")

st.info(
    "Low, Moderate and High are statistical healthcare-utilization groups "
    "based on diagnosis frequency. They are not clinical risk categories "
    "or medical diagnoses."
)

# ML model evaluation
st.header("Machine Learning Model Evaluation")

left, right = st.columns(2)

with left:
    st.subheader("Scikit-learn MiniBatchKMeans")
    st.dataframe(sklearn_eval, use_container_width=True)
    best_sklearn = sklearn_eval.loc[sklearn_eval["silhouette_score"].idxmax()]
    st.metric(
        "Best K",
        int(best_sklearn["k"]),
        help="Number of clusters with the highest tested silhouette score."
    )
    st.metric(
        "Best Silhouette Score",
        f"{best_sklearn['silhouette_score']:.4f}"
    )

with right:
    st.subheader("Spark MLlib K-Means")
    st.dataframe(spark_eval, use_container_width=True)
    best_spark = spark_eval.loc[spark_eval["silhouette"].idxmax()]
    st.metric("Best K", int(best_spark["k"]))
    st.metric("Best Silhouette Score", f"{best_spark['silhouette']:.4f}")

st.warning(
    "The displayed Spark and Scikit-learn silhouette scores were calculated "
    "using different evaluation methods and sample sizes. Do not interpret "
    "them as a direct performance comparison. A common evaluation is still needed."
)

# Explain accuracy
st.info(
    "This project uses unsupervised K-Means clustering. Since there are no "
    "ground-truth cluster labels, classification accuracy is not applicable. "
    "Silhouette score is used to evaluate cluster separation; higher values "
    "generally indicate better-separated clusters."
)

# Data preview
with st.expander("View anonymized patient feature records"):
    st.dataframe(filtered.head(100), use_container_width=True)

st.caption(
    "Educational analytics only. Clusters are exploratory statistical groups, "
    "not medical diagnoses or validated clinical risk predictions."
)

# Classification model comparison
st.header("Classification Model Comparison")

comparison_file = OUTPUT / "classification_model_comparison.csv"

if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)

    st.subheader("Model Performance Metrics")

    col1, col2, col3 = st.columns(3)

    ml_models = comparison[comparison["Model"] != "Baseline"]
    best_model = ml_models.loc[ml_models["Macro_F1"].idxmax()]

    col1.metric("Best ML Model", best_model["Model"])
    col2.metric("Best ML Accuracy", f'{best_model["Accuracy"] * 100:.2f}%')
    col3.metric("Best Macro F1", f'{best_model["Macro_F1"] * 100:.2f}%')

    st.subheader("Accuracy Comparison")

    accuracy_chart = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        color="Model",
        text=comparison["Accuracy"].apply(lambda x: f"{x:.2%}"),
        title="Classification Accuracy"
    )
    accuracy_chart.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(accuracy_chart, use_container_width=True)

    st.subheader("Balanced Accuracy and Macro F1")

    score_data = comparison.melt(
        id_vars="Model",
        value_vars=["Balanced_Accuracy", "Macro_F1"],
        var_name="Metric",
        value_name="Score"
    )

    score_chart = px.bar(
        score_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Balanced Accuracy vs Macro F1"
    )
    score_chart.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(score_chart, use_container_width=True)

    st.subheader("Detailed Classification Results")
    st.dataframe(comparison, use_container_width=True)

    st.info(
        "Random Forest achieved the highest Macro F1 and balanced accuracy "
        "among the four trained classifiers in this experiment. The majority-class "
        "baseline has higher raw accuracy, so accuracy alone does not identify "
        "the best-performing classifier."
    )

    st.warning(
        "The target is visit status. These results measure performance on this "
        "dataset and should not be interpreted as clinical predictions."
    )

else:
    st.warning("Classification results not found. Run the model comparison script first.")

# Random Forest confusion matrix
st.subheader("Random Forest Confusion Matrix")

confusion_file = OUTPUT / "random_forest_confusion_matrix.csv"

if confusion_file.exists():
    confusion = pd.read_csv(confusion_file, index_col=0)

    heatmap = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted Visit Status",
            "y": "Actual Visit Status",
            "color": "Number of Records"
        },
        title="Random Forest Prediction Results"
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.caption(
        "Diagonal values represent correct predictions. "
        "Off-diagonal values represent incorrect predictions."
    )

# Visit Outcome Distribution Shift
st.header("Visit Outcome Distribution Shift")

shift_file = OUTPUT / "visit_status_temporal_analysis.csv"

if shift_file.exists():
    shift = pd.read_csv(shift_file)

    percentage_columns = [
        "Admitted_percent",
        "Discharged_percent",
        "Need Follow-Up_percent"
    ]

    shift_chart = px.line(
        shift,
        x="month",
        y=percentage_columns,
        markers=True,
        title="Monthly Visit Outcome Distribution",
        labels={
            "month": "Month",
            "value": "Percentage of Visits",
            "variable": "Visit Outcome"
        }
    )

    shift_chart.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(shift_chart, use_container_width=True)

    st.subheader("Monthly Outcome Data")
    st.dataframe(shift, use_container_width=True)

    st.warning(
        "Significant temporal distribution shift detected. Admitted visits "
        "increase from approximately 0% in early 2024 to 93.7% in September "
        "2025. This affects classification performance and limits the "
        "reliability of predictions on future records."
    )

else:
    st.warning(
        "Temporal analysis file not found. Run "
        "scripts/analyze_visit_status_shift.py first."
    )
