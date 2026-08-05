"""Paper-primary statistical analyses with labeled robustness checks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .constants import (
    AGE_COLUMN,
    BMI_COLUMN,
    CLL_COLUMN,
    CORRELATION_VARIABLES,
    GENDER_COLUMN,
    GROUP_COLUMN,
    GROUP_ORDER,
    HIGHLIGHT_OUTCOMES,
    OUTCOME_LABELS,
    OUTCOMES,
)


@dataclass(frozen=True)
class StatisticalResults:
    """All tables and matrices produced by the statistical analysis."""

    cohort_characteristics: pd.DataFrame
    group_comparisons: pd.DataFrame
    cll_outcome_correlations: pd.DataFrame
    correlation_matrix_long: pd.DataFrame
    tissue_associations: pd.DataFrame
    pearson_matrix: pd.DataFrame
    pearson_p_matrix: pd.DataFrame


def _fdr_bh(values: pd.Series | np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg only to finite p-values and preserve missing cells."""

    p_values = np.asarray(values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if finite.any():
        adjusted[finite] = stats.false_discovery_control(p_values[finite], method="bh")
    return adjusted


def _continuous_group_test(
    frame: pd.DataFrame,
    variable: str,
) -> dict[str, float | int | str]:
    low = frame.loc[frame[GROUP_COLUMN] == GROUP_ORDER[0], variable].dropna()
    high = frame.loc[frame[GROUP_COLUMN] == GROUP_ORDER[1], variable].dropna()

    student = stats.ttest_ind(low, high, equal_var=True, alternative="two-sided")
    welch = stats.ttest_ind(low, high, equal_var=False, alternative="two-sided")
    mann_whitney = stats.mannwhitneyu(low, high, alternative="two-sided")

    low_mean = float(low.mean())
    high_mean = float(high.mean())
    percent_difference = (
        float((high_mean - low_mean) / low_mean * 100.0) if low_mean != 0 else np.nan
    )

    return {
        "variable": variable,
        "label": OUTCOME_LABELS.get(variable, variable),
        "n_lo_cll": int(len(low)),
        "n_hi_cll": int(len(high)),
        "mean_lo_cll": low_mean,
        "sd_lo_cll": float(low.std(ddof=1)),
        "sem_lo_cll": float(low.sem(ddof=1)),
        "median_lo_cll": float(low.median()),
        "mean_hi_cll": high_mean,
        "sd_hi_cll": float(high.std(ddof=1)),
        "sem_hi_cll": float(high.sem(ddof=1)),
        "median_hi_cll": float(high.median()),
        "mean_difference_hi_minus_lo": high_mean - low_mean,
        "percent_difference": percent_difference,
        "student_t_statistic": float(student.statistic),
        "student_p": float(student.pvalue),
        "welch_t_statistic": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "mannwhitney_u": float(mann_whitney.statistic),
        "mannwhitney_p": float(mann_whitney.pvalue),
    }


def group_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the eight prespecified microCT outcomes between CLL groups."""

    result = pd.DataFrame([_continuous_group_test(frame, variable) for variable in OUTCOMES])
    result["student_fdr_q"] = _fdr_bh(result["student_p"])
    result["welch_fdr_q"] = _fdr_bh(result["welch_p"])
    result["mannwhitney_fdr_q"] = _fdr_bh(result["mannwhitney_p"])
    result["analysis_family"] = "eight_microct_group_comparisons"
    return result


def cohort_characteristics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize age, BMI, gender, and CLL involvement by analysis group."""

    rows: list[dict[str, object]] = []
    for variable in (AGE_COLUMN, BMI_COLUMN):
        test = _continuous_group_test(frame, variable)
        rows.append(
            {
                "characteristic": variable,
                "type": "continuous",
                **{key: value for key, value in test.items() if key != "label"},
                "primary_test": "two-sided Student t-test",
                "primary_statistic": test["student_t_statistic"],
                "primary_p": test["student_p"],
                "sensitivity_test": "two-sided Welch t-test",
                "sensitivity_statistic": test["welch_t_statistic"],
                "sensitivity_p": test["welch_p"],
                "nonparametric_test": "two-sided Mann-Whitney U",
                "nonparametric_statistic": test["mannwhitney_u"],
                "nonparametric_p": test["mannwhitney_p"],
            }
        )

    contingency = (
        pd.crosstab(frame[GROUP_COLUMN], frame[GENDER_COLUMN])
        .reindex(index=GROUP_ORDER, columns=["M", "F"], fill_value=0)
        .astype(int)
    )
    fisher = stats.fisher_exact(contingency.to_numpy(), alternative="two-sided")
    rows.append(
        {
            "characteristic": GENDER_COLUMN,
            "type": "categorical",
            "n_lo_cll": int(contingency.loc[GROUP_ORDER[0]].sum()),
            "n_hi_cll": int(contingency.loc[GROUP_ORDER[1]].sum()),
            "male_n_lo_cll": int(contingency.loc[GROUP_ORDER[0], "M"]),
            "female_n_lo_cll": int(contingency.loc[GROUP_ORDER[0], "F"]),
            "male_n_hi_cll": int(contingency.loc[GROUP_ORDER[1], "M"]),
            "female_n_hi_cll": int(contingency.loc[GROUP_ORDER[1], "F"]),
            "male_percent_lo_cll": float(
                contingency.loc[GROUP_ORDER[0], "M"]
                / contingency.loc[GROUP_ORDER[0]].sum()
                * 100
            ),
            "male_percent_hi_cll": float(
                contingency.loc[GROUP_ORDER[1], "M"]
                / contingency.loc[GROUP_ORDER[1]].sum()
                * 100
            ),
            "primary_test": "two-sided Fisher exact test",
            "primary_statistic": float(fisher.statistic),
            "primary_p": float(fisher.pvalue),
            "sensitivity_test": "two-sided Fisher exact test",
            "sensitivity_statistic": float(fisher.statistic),
            "sensitivity_p": float(fisher.pvalue),
            "nonparametric_test": "two-sided Fisher exact test",
            "nonparametric_statistic": float(fisher.statistic),
            "nonparametric_p": float(fisher.pvalue),
        }
    )

    low_cll = frame.loc[frame[GROUP_COLUMN] == GROUP_ORDER[0], CLL_COLUMN]
    high_cll = frame.loc[frame[GROUP_COLUMN] == GROUP_ORDER[1], CLL_COLUMN]
    rows.append(
        {
            "characteristic": CLL_COLUMN,
            "type": "descriptive",
            "n_lo_cll": int(len(low_cll)),
            "n_hi_cll": int(len(high_cll)),
            "median_lo_cll": float(low_cll.median()),
            "min_lo_cll": float(low_cll.min()),
            "max_lo_cll": float(low_cll.max()),
            "median_hi_cll": float(high_cll.median()),
            "min_hi_cll": float(high_cll.min()),
            "max_hi_cll": float(high_cll.max()),
            "primary_test": "not tested; CLL involvement defines the groups",
        }
    )

    result = pd.DataFrame(rows)
    tested = result["primary_p"].notna()
    result.loc[tested, "primary_fdr_q"] = _fdr_bh(result.loc[tested, "primary_p"])
    result.loc[tested, "sensitivity_fdr_q"] = _fdr_bh(
        result.loc[tested, "sensitivity_p"]
    )
    result.loc[tested, "nonparametric_fdr_q"] = _fdr_bh(
        result.loc[tested, "nonparametric_p"]
    )
    result["analysis_family"] = "three_demographic_group_comparisons"
    return result


def _pairwise_correlations(frame: pd.DataFrame, x: str, y: str) -> dict[str, object]:
    complete = frame[[x, y]].dropna()
    pearson = stats.pearsonr(complete[x], complete[y])
    spearman = stats.spearmanr(complete[x], complete[y])
    regression = stats.linregress(complete[x], complete[y])
    return {
        "x": x,
        "y": y,
        "n": int(len(complete)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "linear_slope": float(regression.slope),
        "linear_intercept": float(regression.intercept),
        "linear_slope_stderr": float(regression.stderr),
    }


def cll_outcome_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Relate continuous marrow involvement to the eight microCT outcomes."""

    rows = []
    for outcome in OUTCOMES:
        row = _pairwise_correlations(frame, CLL_COLUMN, outcome)
        row.update({"outcome": outcome, "label": OUTCOME_LABELS[outcome]})
        rows.append(row)
    result = pd.DataFrame(rows)
    result["pearson_fdr_q"] = _fdr_bh(result["pearson_p"])
    result["spearman_fdr_q"] = _fdr_bh(result["spearman_p"])
    result["analysis_family"] = "eight_cll_outcome_correlations"
    return result


def full_correlation_matrix(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return 55 unique pairwise results and Pearson matrices for plotting."""

    rows = [
        _pairwise_correlations(frame, first, second)
        for first, second in combinations(CORRELATION_VARIABLES, 2)
    ]
    result = pd.DataFrame(rows)
    result["pearson_fdr_q"] = _fdr_bh(result["pearson_p"])
    result["spearman_fdr_q"] = _fdr_bh(result["spearman_p"])
    result["analysis_family"] = "all_55_unique_pairwise_correlations"

    pearson_matrix = pd.DataFrame(
        np.eye(len(CORRELATION_VARIABLES)),
        index=CORRELATION_VARIABLES,
        columns=CORRELATION_VARIABLES,
    )
    p_matrix = pd.DataFrame(
        np.zeros((len(CORRELATION_VARIABLES), len(CORRELATION_VARIABLES))),
        index=CORRELATION_VARIABLES,
        columns=CORRELATION_VARIABLES,
    )
    for row in result.itertuples(index=False):
        pearson_matrix.loc[row.x, row.y] = row.pearson_r
        pearson_matrix.loc[row.y, row.x] = row.pearson_r
        p_matrix.loc[row.x, row.y] = row.pearson_p
        p_matrix.loc[row.y, row.x] = row.pearson_p

    return result, pearson_matrix, p_matrix


def tissue_associations(frame: pd.DataFrame) -> pd.DataFrame:
    """Test age, BMI, and gender against the two highlighted tissue outcomes."""

    rows: list[dict[str, object]] = []
    for outcome in HIGHLIGHT_OUTCOMES:
        for covariate in (AGE_COLUMN, BMI_COLUMN):
            correlations = _pairwise_correlations(frame, covariate, outcome)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    "covariate": covariate,
                    "association_type": "continuous",
                    "n": correlations["n"],
                    "primary_test": "two-sided Pearson correlation",
                    "primary_statistic": correlations["pearson_r"],
                    "primary_p": correlations["pearson_p"],
                    "sensitivity_test": "two-sided Spearman correlation",
                    "sensitivity_statistic": correlations["spearman_rho"],
                    "sensitivity_p": correlations["spearman_p"],
                }
            )

        male = frame.loc[frame[GENDER_COLUMN] == "M", outcome].dropna()
        female = frame.loc[frame[GENDER_COLUMN] == "F", outcome].dropna()
        student = stats.ttest_ind(male, female, equal_var=True, alternative="two-sided")
        welch = stats.ttest_ind(male, female, equal_var=False, alternative="two-sided")
        mann_whitney = stats.mannwhitneyu(male, female, alternative="two-sided")
        rows.append(
            {
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "covariate": GENDER_COLUMN,
                "association_type": "categorical",
                "n": int(len(male) + len(female)),
                "n_male": int(len(male)),
                "n_female": int(len(female)),
                "mean_male": float(male.mean()),
                "mean_female": float(female.mean()),
                "primary_test": "two-sided Student t-test",
                "primary_statistic": float(student.statistic),
                "primary_p": float(student.pvalue),
                "sensitivity_test": "two-sided Welch t-test",
                "sensitivity_statistic": float(welch.statistic),
                "sensitivity_p": float(welch.pvalue),
                "nonparametric_test": "two-sided Mann-Whitney U",
                "nonparametric_statistic": float(mann_whitney.statistic),
                "nonparametric_p": float(mann_whitney.pvalue),
            }
        )

    result = pd.DataFrame(rows)
    result["primary_fdr_q"] = _fdr_bh(result["primary_p"])
    result["sensitivity_fdr_q"] = _fdr_bh(result["sensitivity_p"])
    result["nonparametric_fdr_q"] = _fdr_bh(result["nonparametric_p"])
    result["analysis_family"] = "six_highlighted_tissue_associations"
    return result


def run_statistics(frame: pd.DataFrame) -> StatisticalResults:
    """Run every workbook-supported paper and sensitivity analysis."""

    full_long, pearson_matrix, pearson_p_matrix = full_correlation_matrix(frame)
    return StatisticalResults(
        cohort_characteristics=cohort_characteristics(frame),
        group_comparisons=group_comparisons(frame),
        cll_outcome_correlations=cll_outcome_correlations(frame),
        correlation_matrix_long=full_long,
        tissue_associations=tissue_associations(frame),
        pearson_matrix=pearson_matrix,
        pearson_p_matrix=pearson_p_matrix,
    )

