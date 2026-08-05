"""Input loading and strict validation for the supplementary workbook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import (
    BMI_COLUMN,
    CRITICAL_NONMISSING_COLUMNS,
    GENDER_COLUMN,
    GROUP_COLUMN,
    GROUP_ORDER,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised when the supplementary workbook cannot support the analysis."""


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the sole analysis workbook and return a validated, independent frame."""

    input_path = Path(path)
    if not input_path.is_file():
        raise DataValidationError(f"Input workbook does not exist: {input_path}")
    if input_path.suffix.lower() != ".xlsx":
        raise DataValidationError("The input must be an .xlsx workbook.")

    frame = pd.read_excel(input_path, sheet_name="Sheet1", engine="openpyxl")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    unexpected_columns = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {missing_columns}")
        if unexpected_columns:
            details.append(f"unexpected columns: {unexpected_columns}")
        raise DataValidationError("Workbook schema mismatch (" + "; ".join(details) + ").")

    duplicate_ids = frame.loc[frame[ID_COLUMN].duplicated(keep=False), ID_COLUMN].tolist()
    if duplicate_ids:
        raise DataValidationError(f"Duplicate sample IDs found: {duplicate_ids}")

    missing_critical = {
        column: int(frame[column].isna().sum())
        for column in CRITICAL_NONMISSING_COLUMNS
        if frame[column].isna().any()
    }
    if missing_critical:
        raise DataValidationError(f"Critical analysis values are missing: {missing_critical}")

    invalid_groups = sorted(set(frame[GROUP_COLUMN].dropna()) - set(GROUP_ORDER))
    if invalid_groups or set(frame[GROUP_COLUMN]) != set(GROUP_ORDER):
        raise DataValidationError(
            f"Expected both {GROUP_ORDER}; invalid or missing group labels: {invalid_groups}"
        )

    invalid_genders = sorted(set(frame[GENDER_COLUMN].dropna()) - {"M", "F"})
    if invalid_genders:
        raise DataValidationError(f"Unexpected Gender values: {invalid_genders}")

    nonnumeric = [
        column for column in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise DataValidationError(f"Expected numeric columns: {nonnumeric}")

    # BMI is intentionally not imputed. All analyses using it perform pairwise deletion.
    if frame[BMI_COLUMN].notna().sum() < 2:
        raise DataValidationError("BMI has fewer than two observed values.")

    return frame.copy(deep=True)

