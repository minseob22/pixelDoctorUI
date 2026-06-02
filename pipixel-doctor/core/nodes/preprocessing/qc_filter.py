def is_valid_case(
    case,
    required_features
):
    features = case.get(
        "features",
        {}
    )

    if not features:
        return False

    for key in required_features:

        value = features.get(key)

        if value is None:
            return False

    return True