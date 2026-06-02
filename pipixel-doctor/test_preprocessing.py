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


with open(
    "data/sample_ct_result.json",
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


cases = dataset["cases"]

tumor_cases = filter_tumor_cases(cases)

case_data = tumor_cases[0]


selected = feature_selector(case_data)

semantic = feature_interpreter(selected)

print(json.dumps(
    semantic,
    indent=2,
    ensure_ascii=False
))