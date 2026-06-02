def feature_selector(
    case_data,
    selected_keys
):
    features = case_data["features"]

    selected = {}

    for key in selected_keys:

        selected[key] = features.get(
            key
        )

    return {
        "case_id":
            case_data["case_id"],
        "label_name":
            case_data["label_name"],
        "selected_features":
            selected
    }