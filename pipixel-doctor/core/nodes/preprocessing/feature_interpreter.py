def feature_interpreter(
    selected_data
):

    features = selected_data[
        "selected_features"
    ]

    print(
        "\n===== INTERPRETER INPUT ====="
    )
    print(features)

    # =========================
    # raw values
    # =========================

    diameter = features.get(
        "original_shape_Maximum3DDiameter"
    )

    volume = features.get(
        "original_shape_VoxelVolume"
    )

    sphericity = features.get(
        "original_shape_Sphericity"
    )

    surface_ratio = features.get(
        "original_shape_SurfaceVolumeRatio"
    )

    mean_intensity = features.get(
        "original_firstorder_Mean"
    )

    variance = features.get(
        "original_firstorder_Variance"
    )

    entropy = features.get(
        "original_firstorder_Entropy"
    )

    glcm_contrast = features.get(
        "original_glcm_Contrast"
    )

    glcm_corr = features.get(
        "original_glcm_Correlation"
    )

    run_entropy = features.get(
        "original_glrlm_RunEntropy"
    )

    zone_entropy = features.get(
        "original_glszm_ZoneEntropy"
    )

    # =========================
    # size
    # =========================

    if diameter is None:
        size_category = "unknown"

    elif diameter < 30:
        size_category = "small"

    elif diameter < 100:
        size_category = "medium"

    else:
        size_category = "large"

    # =========================
    # shape
    # =========================

    if sphericity is None:
        shape = "unknown"

    elif sphericity < 0.5:
        shape = "irregular"

    elif sphericity < 0.75:
        shape = "moderately round"

    else:
        shape = "round"

    # =========================
    # surface
    # =========================

    if surface_ratio is None:
        surface = "unknown"

    elif surface_ratio < 0.1:
        surface = "smooth"

    elif surface_ratio < 0.3:
        surface = "mildly irregular"

    else:
        surface = "irregular"

    # =========================
    # density
    # =========================

    if mean_intensity is None:
        density = "unknown"

    elif mean_intensity < 40:
        density = "low density"

    elif mean_intensity < 80:
        density = "moderate density"

    else:
        density = "high density"

    # =========================
    # variance
    # =========================

    if variance is None:
        variability = "unknown"

    elif variance < 100:
        variability = "low"

    elif variance < 500:
        variability = "moderate"

    else:
        variability = "high"

    # =========================
    # heterogeneity
    # =========================

    if entropy is None:
        heterogeneity = "unknown"

    elif entropy < 2:
        heterogeneity = "low"

    elif entropy < 4:
        heterogeneity = "moderate"

    else:
        heterogeneity = "high"

    # =========================
    # texture complexity
    # =========================

    if run_entropy is None:
        texture_complexity = "unknown"

    elif run_entropy < 2:
        texture_complexity = "simple"

    elif run_entropy < 4:
        texture_complexity = "moderate"

    else:
        texture_complexity = "complex"

    # =========================
    # organization
    # =========================

    if glcm_corr is None:
        organization = "unknown"

    elif glcm_corr < 0.3:
        organization = "disorganized"

    else:
        organization = "organized"

    # =========================
    # contrast
    # =========================

    if glcm_contrast is None:
        texture_contrast = "unknown"

    elif glcm_contrast < 1:
        texture_contrast = "low"

    else:
        texture_contrast = "high"

    return {

        "case_id":
            selected_data["case_id"],

        "finding": {

            "type":
                selected_data[
                    "label_name"
                ],

            "size_mm":
                round(diameter, 2)
                if diameter else None,

            "volume_mm3":
                round(volume, 2)
                if volume else None,

            "size_category":
                size_category,

            "shape":
                shape,

            "surface_irregularity":
                surface,

            "density":
                density,

            "intensity_variability":
                variability,

            "internal_heterogeneity":
                heterogeneity,

            "texture_complexity":
                texture_complexity,

            "texture_organization":
                organization,

            "texture_contrast":
                texture_contrast,
        },
    }