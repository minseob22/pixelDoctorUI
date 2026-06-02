import json

from core.nodes.preprocessing.feature_filter import (
    filter_tumor_cases
)

from core.nodes.preprocessing.feature_selector import (
    feature_selector
)

from core.nodes.preprocessing.feature_interpreter import (
    feature_interpreter
)

from core.nodes.doctors.doctor_explainer import (
    doctor_explainer
)


with open(
    "data/sample_ct_result.json",
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


cases = dataset["cases"]

tumor_cases = filter_tumor_cases(cases)

print(f"Valid tumor cases: {len(tumor_cases)}")


case_data = tumor_cases[1]

print("\n===== RAW CASE =====\n")

print(case_data.keys())


selected = feature_selector(case_data)

print("\n===== SELECTED =====\n")

print(selected)


semantic = feature_interpreter(selected)

print("\n===== SEMANTIC =====\n")

print(semantic)


doctor_summary = doctor_explainer(semantic)

print("\n===== DOCTOR SUMMARY =====\n")

print(doctor_summary)