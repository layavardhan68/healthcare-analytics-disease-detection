# Healthcare Analytics and Disease Pattern Detection

## Project Objective
Design and develop a Big Data-based healthcare analytics system to process large-scale patient health records, enable distributed storage, analytical querying, and machine-learning-based pattern identification to detect disease trends and patient utilization groups.

## Technologies
- Apache Hadoop HDFS
- MapReduce
- Apache Hive
- Apache Spark MLlib
- Python and Scikit-learn
- Pandas and NumPy
- Streamlit and Plotly

## Dataset
Healthcare Data Set – 20 Tables  
Source: Kaggle  
https://www.kaggle.com/datasets/moid1234/health-care-data-set-20-tables

## Project Results
- 1,599,000 valid diagnosis records
- 256,079 unique patients
- 40 disease categories
- 21 months of disease trend analysis
- 3 patient utilization groups
- K-Means clustering using Scikit-learn and Spark MLlib

## Machine Learning
- Scikit-learn MiniBatchKMeans
- Apache Spark MLlib K-Means
- Evaluation using silhouette scores

K-Means is an unsupervised algorithm, so classification accuracy is not applicable.

## Dashboard
Run locally:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
