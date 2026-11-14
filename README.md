# Healthcare Analytics and Disease Pattern Detection

## Project Objective
Develop a Big Data healthcare analytics system to process large-scale patient records, perform distributed analytics, identify disease patterns and explore patient groups.

## Technologies
- Apache Hadoop HDFS
- MapReduce
- Apache Hive
- Apache Spark MLlib
- Python, Pandas and Scikit-learn
- Streamlit and Plotly

## Dataset
Healthcare Data Set – 20 Tables  
Source: Kaggle  
https://www.kaggle.com/datasets/moid1234/health-care-data-set-20-tables

The dataset is synthetic/training-oriented healthcare data.

## Dataset Processing Results
- 1,599,000 valid diagnosis records
- 256,079 unique patients
- 40 disease categories
- 21 months of disease trend analysis
- 3 patient utilization groups

## Big Data Processing
- Distributed storage using Hadoop HDFS
- Data processing using MapReduce
- Healthcare queries using Apache Hive
- Distributed machine learning using Spark MLlib

## Machine Learning Models

### K-Means Clustering
- Scikit-learn MiniBatchKMeans
- Apache Spark MLlib K-Means
- Patient segmentation
- Silhouette score evaluation

K-Means is unsupervised; classification accuracy does not apply.

### Classification
Models evaluated:
- Random Forest
- Logistic Regression
- Decision Tree
- Extra Trees
- Majority-class baseline

### Classification Results

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Baseline | 42.11% | 33.33% | 19.76% |
| Random Forest | 33.40% | 33.73% | 32.18% |
| Logistic Regression | 32.03% | 33.60% | 31.31% |
| Extra Trees | 31.85% | 33.45% | 31.13% |
| Decision Tree | 27.14% | 33.60% | 25.54% |

Random Forest achieved the highest Macro F1 and balanced accuracy among the four trained classifiers. The baseline had higher raw accuracy because it predicts the majority class.

## Streamlit Dashboard
The dashboard provides:
- Healthcare dataset overview
- Disease category distribution
- Age and gender analysis
- Monthly disease trends
- K-Means cluster visualizations
- Classification model comparison charts
- Accuracy, balanced accuracy and Macro F1
- Random Forest confusion matrix heatmap
- Patient utilization group analysis

## Run the Dashboard

Activate the environment:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
