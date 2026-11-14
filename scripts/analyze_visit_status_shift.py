import pandas as pd
from pathlib import Path

base = Path("data/raw/STG_EHP_DATASET")
out = Path("output")
out.mkdir(exist_ok=True)

df = pd.read_csv(
    base / "STG_EHP__VIST.csv",
    usecols=["VIS_EN", "VTYPE_DES", "VSTAT_DES"]
)

df["VIS_EN"] = pd.to_datetime(df["VIS_EN"], errors="coerce")
df = df.dropna(subset=["VIS_EN", "VSTAT_DES"])

df["month"] = df["VIS_EN"].dt.to_period("M").astype(str)

summary = (
    df.groupby(["month", "VSTAT_DES"])
      .size()
      .unstack(fill_value=0)
)

percentages = summary.div(summary.sum(axis=1), axis=0) * 100

result = summary.add_suffix("_count").join(
    percentages.add_suffix("_percent")
)

result.index.name = "month"
result.to_csv(out / "visit_status_temporal_analysis.csv")

print("Monthly visit outcome analysis:")
print(result.round(2).to_string())

print("\nSaved: output/visit_status_temporal_analysis.csv")
