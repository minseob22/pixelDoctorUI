from core.nodes.preprocessing.qc_filter import (
    is_valid_case
)


def filter_tumor_cases(cases):

    tumor_cases = []

    for case in cases:

        if (
            case.get("label_name") == "tumor"
            and is_valid_case(case)
        ):
            tumor_cases.append(case)

    return tumor_cases