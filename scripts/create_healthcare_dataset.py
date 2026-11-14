import csv
import hashlib
import os
from datetime import datetime
from collections import defaultdict, Counter

BASE = os.path.expanduser("~/healthcare-bigdata-project")
RAW = BASE + "/data/raw/STG_EHP_DATASET/"
OUT = BASE + "/data/processed/"

# Reuse the existing secret salt to keep IDs consistent
salt_file = BASE + "/.privacy_salt"

if not os.path.exists(salt_file):
    raise FileNotFoundError("Privacy salt missing. Do not generate a new one.")

with open(salt_file, "rb") as f:
    SALT = f.read()


def anonymize(value):
    return hashlib.sha256(SALT + value.encode()).hexdigest()


# Load patient demographics
patients = {}

with open(RAW + "STG_EHP__PATN.csv", newline="") as f:
    for row in csv.DictReader(f):
        try:
            dob = datetime.strptime(
                row["DT_BRT"], "%m/%d/%Y"
            ).date()

            patients[row["PAT_ID"]] = {
                "dob": dob,
                "gender": row["GEN_DES"]
            }
        except (ValueError, KeyError):
            continue


# Index visits by reference AND exact visit timestamp
visits = defaultdict(list)

with open(RAW + "STG_EHP__VIST.csv", newline="") as f:
    for row in csv.DictReader(f):
        key = (row["REFR_NO"], row["VIS_EN"])
        visits[key].append(row)


# Generate corrected dataset
temp_file = OUT + "patient_disease_records.tmp"
output_file = OUT + "patient_disease_records.csv"

stats = Counter()
age_groups = Counter()

with open(OUT + "STG_EHP__DIAG.csv", newline="") as source:
    reader = csv.DictReader(source)

    with open(temp_file, "w", newline="") as output:
        writer = csv.writer(output)

        writer.writerow([
            "patient_id", "visit_id", "age", "age_group",
            "gender", "diagnosis_code", "disease_category",
            "diagnosis_date", "visit_type", "visit_status"
        ])

        for diagnosis in reader:
            stats["total_diagnoses"] += 1

            key = (diagnosis["REFR_NO"], diagnosis["DIG_EN"])
            candidates = visits.get(key, [])

            if len(candidates) == 0:
                stats["unmatched_visits"] += 1
                continue

            if len(candidates) > 1:
                stats["ambiguous_visits"] += 1
                continue

            visit = candidates[0]
            patient = patients.get(visit["PAT_ID"])

            if not patient:
                stats["unmatched_patients"] += 1
                continue

            try:
                diagnosis_date = datetime.strptime(
                    diagnosis["DIG_EN"],
                    "%Y-%m-%d %H:%M:%S"
                ).date()

                dob = patient["dob"]

                age = diagnosis_date.year - dob.year
                age -= (
                    (diagnosis_date.month, diagnosis_date.day)
                    < (dob.month, dob.day)
                )

                if age < 0 or age > 120:
                    stats["invalid_age"] += 1
                    continue

            except ValueError:
                stats["invalid_date"] += 1
                continue

            if age < 18:
                group = "0-17"
            elif age < 36:
                group = "18-35"
            elif age < 51:
                group = "36-50"
            elif age < 66:
                group = "51-65"
            else:
                group = "66+"

            writer.writerow([
                anonymize(visit["PAT_ID"]),
                anonymize(diagnosis["REFR_NO"] + "|" + diagnosis["DIG_EN"]),
                age,
                group,
                patient["gender"],
                diagnosis["DIG_CD"],
                diagnosis["DIG_DES"],
                diagnosis_date.isoformat(),
                visit["VTYPE_DES"],
                visit["VSTAT_DES"]
            ])

            stats["valid_records"] += 1
            age_groups[group] += 1

os.replace(temp_file, output_file)

print("\nCORRECTED DATASET GENERATED")
for key, value in stats.items():
    print(key, ":", value)

print("\nAge groups:")
for group, count in age_groups.items():
    print(group, ":", count)

print("\nOutput:", output_file)
