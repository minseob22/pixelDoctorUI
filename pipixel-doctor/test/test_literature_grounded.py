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

from rag.vector_store import (
    load_vector_store
)
from rag.rag_pipeline import (
    build_rag_context
)


# =========================================
# JSON Load
# =========================================

with open (
    "data/sample_ct_result.json",
    "r",
    encoding="utf-8"
) as f:
    dataset=json.load(f)
    
## Tumor filtering

cases=dataset["cases"]
tumor_cases=filter_tumor_cases(
    cases
)

case_data=tumor_cases[0]

#feature selection

selected=feature_selector(
    case_data
)

#semantic abstraction

semantic=feature_interpreter(
    selected
)

print("\n=== SEMANTIC ====\n")

print(json.dumps(
    semantic,
    indent=2,
    ensure_ascii=False
))


# Load Vector DB

vector_store=load_vector_store()

#build literature context



literature_context = build_rag_context(
    vector_store,
    semantic
)

print("\n===== LITERATURE CONTEXT =====\n")

print(literature_context)

# =========================================
# Doctor explanation
# =========================================

doctor_summary = doctor_explainer(
    semantic,
    literature_context
)


print("\n===== DOCTOR SUMMARY =====\n")

print(doctor_summary)