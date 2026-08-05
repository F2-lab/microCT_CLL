"""Shared column names, labels, and plotting constants."""

from __future__ import annotations

ID_COLUMN = "CC#   Seq"
GROUP_COLUMN = "group"
CLL_COLUMN = "CLL Involvement (0% of marrow)"
AGE_COLUMN = "Age (years)"
BMI_COLUMN = "BMI"
GENDER_COLUMN = "Gender"

GROUP_ORDER = ("Lo-CLL", "Hi-CLL")
GROUP_COLORS = {"Lo-CLL": "#2E5A88", "Hi-CLL": "#CC3311"}

REQUIRED_COLUMNS = (
    ID_COLUMN,
    GROUP_COLUMN,
    "Specimen Classification",
    "Collect Date",
    AGE_COLUMN,
    GENDER_COLUMN,
    BMI_COLUMN,
    CLL_COLUMN,
    "median_lacuna_volume",
    "median_lacuna_aspect_ratio",
    "lacuna_density_per_mm3",
    "mineral_peak",
    "intertrabecular_marrow_density",
    "intertrabecular_marrow_sa",
    "intertrabecular_marrow_volume",
    "intertrabecular_marrow_sa_over_volume_normalized",
    "BV_mm3",
    "TV_mm3",
    "BV/TV",
    "SV_mm3",
    "SV/TV",
)

OUTCOME_LABELS = {
    "median_lacuna_volume": "Lacuna volume",
    "median_lacuna_aspect_ratio": "Lacuna aspect ratio",
    "lacuna_density_per_mm3": "Lacuna density (#/mm³)",
    "mineral_peak": "Tissue mineral peak density",
    "intertrabecular_marrow_density": "Intertrabecular marrow density",
    "intertrabecular_marrow_sa_over_volume_normalized": "Normalized intertrabecular marrow SA/V",
    "BV/TV": "BV/TV",
    "SV/TV": "IMV/TV",
}

OUTCOMES = tuple(OUTCOME_LABELS)
HIGHLIGHT_OUTCOMES = (
    "lacuna_density_per_mm3",
    "intertrabecular_marrow_sa_over_volume_normalized",
)

CORRELATION_VARIABLES = OUTCOMES + (BMI_COLUMN, AGE_COLUMN, CLL_COLUMN)
CORRELATION_LABELS = {
    **OUTCOME_LABELS,
    BMI_COLUMN: "BMI",
    AGE_COLUMN: "Age",
    CLL_COLUMN: "CLL involvement",
}

CRITICAL_NONMISSING_COLUMNS = (
    ID_COLUMN,
    GROUP_COLUMN,
    AGE_COLUMN,
    GENDER_COLUMN,
    CLL_COLUMN,
) + OUTCOMES

NUMERIC_COLUMNS = (
    AGE_COLUMN,
    BMI_COLUMN,
    CLL_COLUMN,
    "median_lacuna_volume",
    "median_lacuna_aspect_ratio",
    "lacuna_density_per_mm3",
    "mineral_peak",
    "intertrabecular_marrow_density",
    "intertrabecular_marrow_sa",
    "intertrabecular_marrow_volume",
    "intertrabecular_marrow_sa_over_volume_normalized",
    "BV_mm3",
    "TV_mm3",
    "BV/TV",
    "SV_mm3",
    "SV/TV",
)

