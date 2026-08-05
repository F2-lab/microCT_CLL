"""Deterministic publication-style figures for all supported analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .clustering import ClusteringResults
from .constants import (
    AGE_COLUMN,
    BMI_COLUMN,
    CLL_COLUMN,
    CORRELATION_LABELS,
    CORRELATION_VARIABLES,
    GENDER_COLUMN,
    GROUP_COLORS,
    GROUP_COLUMN,
    GROUP_ORDER,
    HIGHLIGHT_OUTCOMES,
    OUTCOME_LABELS,
    OUTCOMES,
)
from .statistics import StatisticalResults


def _style() -> None:
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _stars(p_value: float) -> str:
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return ""


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _group_boxplot(
    axis: plt.Axes,
    frame: pd.DataFrame,
    variable: str,
    p_value: float,
    rng: np.random.Generator,
    *,
    title: str | None = None,
) -> None:
    low = frame.loc[frame[GROUP_COLUMN] == GROUP_ORDER[0], variable].dropna()
    high = frame.loc[frame[GROUP_COLUMN] == GROUP_ORDER[1], variable].dropna()
    axis.boxplot(
        [low, high],
        patch_artist=True,
        widths=0.65,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "", "markersize": 0},
    )
    for position, (group, values) in enumerate(
        ((GROUP_ORDER[0], low), (GROUP_ORDER[1], high)), start=1
    ):
        jitter = rng.normal(position, 0.04, size=len(values))
        axis.scatter(
            jitter,
            values,
            color=GROUP_COLORS[group],
            alpha=0.75,
            s=32,
            zorder=3,
        )
    axis.set_xticks([1, 2], [f"Lo-CLL\n(n={len(low)})", f"Hi-CLL\n(n={len(high)})"])
    shown_title = title or OUTCOME_LABELS.get(variable, variable)
    suffix = _stars(float(p_value))
    axis.set_title(f"{shown_title}\n{suffix}" if suffix else shown_title)
    axis.set_ylabel(OUTCOME_LABELS.get(variable, variable))
    _clean_axis(axis)


def plot_clustering(
    frame: pd.DataFrame,
    results: ClusteringResults,
    output_dir: Path,
    *,
    seed: int,
) -> None:
    _style()
    rng = np.random.default_rng(seed + 101)
    values = frame[CLL_COLUMN].to_numpy(dtype=float)
    labels = results.labels
    summary = dict(zip(results.summary["metric"], results.summary["value"], strict=True))

    figure, (distribution, silhouette_axis) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.1, 1.2]}
    )
    for label, group in enumerate(GROUP_ORDER):
        mask = labels == label
        distribution.scatter(
            values[mask],
            rng.uniform(-0.08, 0.08, size=int(mask.sum())),
            color=GROUP_COLORS[group],
            label=f"{group} (n={int(mask.sum())})",
            s=65,
            alpha=0.8,
        )
    distribution.axvline(
        float(summary["centroid_midpoint"]), color="#666666", linestyle="--", linewidth=1.2
    )
    distribution.set_xlabel("CLL involvement (% of marrow)")
    distribution.set_yticks([])
    distribution.set_ylim(-0.22, 0.22)
    distribution.set_title("K-means distribution (k=2)")
    distribution.legend(frameon=False)
    _clean_axis(distribution)

    y_lower = 5
    for label, group in enumerate(GROUP_ORDER):
        cluster_values = np.sort(results.silhouette_values[labels == label])
        y_upper = y_lower + len(cluster_values)
        silhouette_axis.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            color=GROUP_COLORS[group],
            alpha=0.75,
            label=group,
        )
        y_lower = y_upper + 5
    silhouette_axis.axvline(
        float(summary["silhouette_score"]),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {float(summary['silhouette_score']):.3f}",
    )
    silhouette_axis.set_xlim(0, 1)
    silhouette_axis.set_yticks([])
    silhouette_axis.set_xlabel("Silhouette coefficient")
    silhouette_axis.set_title("Silhouette analysis")
    silhouette_axis.legend(frameon=False, loc="lower right")
    _clean_axis(silhouette_axis)

    figure.tight_layout(w_pad=3)
    _save_figure(figure, output_dir, "clustering_summary")


def plot_group_comparisons(
    frame: pd.DataFrame,
    statistics: StatisticalResults,
    output_dir: Path,
    *,
    seed: int,
) -> None:
    _style()
    rng = np.random.default_rng(seed + 202)
    p_values = statistics.group_comparisons.set_index("variable")["student_p"]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, variable in zip(axes.ravel(), OUTCOMES, strict=True):
        _group_boxplot(axis, frame, variable, float(p_values.loc[variable]), rng)
    figure.suptitle("Lo-CLL versus Hi-CLL microCT outcomes", fontsize=16, y=1.01)
    figure.tight_layout(h_pad=2.5, w_pad=2)
    _save_figure(figure, output_dir, "microct_group_comparisons")


def plot_highlighted_outcome(
    frame: pd.DataFrame,
    statistics: StatisticalResults,
    output_dir: Path,
    outcome: str,
    *,
    seed: int,
) -> None:
    _style()
    offset = HIGHLIGHT_OUTCOMES.index(outcome)
    rng = np.random.default_rng(seed + 303 + offset)
    group_row = statistics.group_comparisons.set_index("variable").loc[outcome]
    correlation_row = statistics.cll_outcome_correlations.set_index("outcome").loc[outcome]

    figure, (group_axis, correlation_axis) = plt.subplots(1, 2, figsize=(12, 5))
    _group_boxplot(
        group_axis,
        frame,
        outcome,
        float(group_row["student_p"]),
        rng,
        title=OUTCOME_LABELS[outcome],
    )

    for group in GROUP_ORDER:
        subset = frame.loc[frame[GROUP_COLUMN] == group, [CLL_COLUMN, outcome]].dropna()
        correlation_axis.scatter(
            subset[CLL_COLUMN],
            subset[outcome],
            color=GROUP_COLORS[group],
            label=group,
            s=55,
            alpha=0.8,
        )
    x_line = np.linspace(frame[CLL_COLUMN].min(), frame[CLL_COLUMN].max(), 100)
    y_line = correlation_row["linear_intercept"] + correlation_row["linear_slope"] * x_line
    correlation_axis.plot(x_line, y_line, color="#666666", linestyle="--", linewidth=1.4)
    correlation_axis.text(
        0.04,
        0.96,
        f"r = {correlation_row['pearson_r']:.2f}\np = {correlation_row['pearson_p']:.3f}",
        transform=correlation_axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    correlation_axis.set_xlabel("CLL involvement (% of marrow)")
    correlation_axis.set_ylabel(OUTCOME_LABELS[outcome])
    correlation_axis.set_title("Continuous CLL association")
    correlation_axis.legend(frameon=False)
    _clean_axis(correlation_axis)
    figure.tight_layout(w_pad=3)

    stem = {
        "lacuna_density_per_mm3": "lacuna_density_analysis",
        "intertrabecular_marrow_sa_over_volume_normalized": "intertrabecular_marrow_sav_analysis",
    }[outcome]
    _save_figure(figure, output_dir, stem)


def plot_correlation_matrix(
    statistics: StatisticalResults,
    output_dir: Path,
) -> None:
    _style()
    correlation = statistics.pearson_matrix.rename(
        index=CORRELATION_LABELS, columns=CORRELATION_LABELS
    )
    p_values = statistics.pearson_p_matrix.rename(
        index=CORRELATION_LABELS, columns=CORRELATION_LABELS
    )
    annotations = correlation.copy().astype(object)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            if row == column:
                annotations.iat[row, column] = "1.00"
            else:
                p_value = float(p_values.iat[row, column])
                annotations.iat[row, column] = (
                    f"{correlation.iat[row, column]:.2f}{_stars(p_value)}"
                )

    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    figure, axis = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        correlation,
        mask=mask,
        annot=annotations,
        fmt="",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.25,
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Pearson correlation coefficient"},
        ax=axis,
    )
    axis.set_title("Pairwise Pearson correlations\n* p≤0.05, ** p≤0.01, *** p≤0.001")
    axis.tick_params(axis="x", rotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    _save_figure(figure, output_dir, "correlation_matrix")


def _continuous_association_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    result: pd.Series,
) -> None:
    complete = frame[[x, y, GROUP_COLUMN]].dropna()
    for group in GROUP_ORDER:
        subset = complete[complete[GROUP_COLUMN] == group]
        axis.scatter(
            subset[x],
            subset[y],
            color=GROUP_COLORS[group],
            alpha=0.8,
            s=42,
            label=group,
        )
    coefficients = np.polyfit(complete[x], complete[y], 1)
    x_line = np.linspace(complete[x].min(), complete[x].max(), 100)
    axis.plot(x_line, np.polyval(coefficients, x_line), color="#666666", linestyle="--")
    axis.set_xlabel(CORRELATION_LABELS.get(x, x))
    axis.set_ylabel(OUTCOME_LABELS[y])
    axis.set_title(
        f"r={result['primary_statistic']:.2f}, p={result['primary_p']:.3f}"
    )
    _clean_axis(axis)


def _gender_outcome_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    outcome: str,
    result: pd.Series,
    rng: np.random.Generator,
) -> None:
    male = frame.loc[frame[GENDER_COLUMN] == "M", outcome].dropna()
    female = frame.loc[frame[GENDER_COLUMN] == "F", outcome].dropna()
    means = [male.mean(), female.mean()]
    sems = [male.sem(), female.sem()]
    axis.bar(
        [1, 2],
        means,
        yerr=sems,
        color="white",
        edgecolor="black",
        capsize=4,
        width=0.65,
    )
    for position, values, color in (
        (1, male, "#87CEEB"),
        (2, female, "#F4A6C1"),
    ):
        axis.scatter(
            rng.normal(position, 0.04, size=len(values)),
            values,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=35,
            zorder=3,
        )
    axis.set_xticks([1, 2], [f"Male\n(n={len(male)})", f"Female\n(n={len(female)})"])
    axis.set_ylabel(OUTCOME_LABELS[outcome])
    axis.set_title(f"Student t-test, p={result['primary_p']:.3f}")
    _clean_axis(axis)


def plot_tissue_associations(
    frame: pd.DataFrame,
    statistics: StatisticalResults,
    output_dir: Path,
    *,
    seed: int,
) -> None:
    _style()
    rng = np.random.default_rng(seed + 404)
    association_lookup = statistics.tissue_associations.set_index(["outcome", "covariate"])
    cohort_lookup = statistics.cohort_characteristics.set_index("characteristic")
    figure, axes = plt.subplots(3, 3, figsize=(15, 13))

    for row, outcome in enumerate(HIGHLIGHT_OUTCOMES):
        for column, covariate in enumerate((AGE_COLUMN, BMI_COLUMN)):
            result = association_lookup.loc[(outcome, covariate)]
            _continuous_association_panel(
                axes[row, column], frame, covariate, outcome, result
            )
        _gender_outcome_panel(
            axes[row, 2],
            frame,
            outcome,
            association_lookup.loc[(outcome, GENDER_COLUMN)],
            rng,
        )

    for column, variable in enumerate((AGE_COLUMN, BMI_COLUMN)):
        row = cohort_lookup.loc[variable]
        _group_boxplot(
            axes[2, column],
            frame,
            variable,
            float(row["primary_p"]),
            rng,
            title=CORRELATION_LABELS[variable],
        )

    contingency = (
        pd.crosstab(frame[GROUP_COLUMN], frame[GENDER_COLUMN])
        .reindex(index=GROUP_ORDER, columns=["M", "F"], fill_value=0)
        .astype(int)
    )
    positions = np.arange(2)
    width = 0.36
    axes[2, 2].bar(
        positions - width / 2,
        contingency["M"],
        width,
        label="Male",
        color=[GROUP_COLORS[group] for group in GROUP_ORDER],
        edgecolor="black",
    )
    axes[2, 2].bar(
        positions + width / 2,
        contingency["F"],
        width,
        label="Female",
        color=[GROUP_COLORS[group] for group in GROUP_ORDER],
        edgecolor="black",
        hatch="///",
    )
    axes[2, 2].set_xticks(positions, GROUP_ORDER)
    axes[2, 2].set_ylabel("Count")
    axes[2, 2].set_title(
        f"Fisher exact test, p={cohort_lookup.loc[GENDER_COLUMN, 'primary_p']:.3f}"
    )
    axes[2, 2].legend(frameon=False)
    _clean_axis(axes[2, 2])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=2,
            frameon=False,
        )
    figure.suptitle("Tissue associations and cohort characteristics", fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.5, w_pad=2)
    _save_figure(figure, output_dir, "tissue_associations")


def create_all_figures(
    frame: pd.DataFrame,
    clustering: ClusteringResults,
    statistics: StatisticalResults,
    output_dir: Path,
    *,
    seed: int,
) -> None:
    """Write all reproducible PNG and PDF analysis figures."""

    plot_clustering(frame, clustering, output_dir, seed=seed)
    plot_group_comparisons(frame, statistics, output_dir, seed=seed)
    for outcome in HIGHLIGHT_OUTCOMES:
        plot_highlighted_outcome(
            frame, statistics, output_dir, outcome, seed=seed
        )
    plot_correlation_matrix(statistics, output_dir)
    plot_tissue_associations(frame, statistics, output_dir, seed=seed)
