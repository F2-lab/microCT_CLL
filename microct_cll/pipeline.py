"""End-to-end orchestration for tables and publication figures."""

from __future__ import annotations

from pathlib import Path

from .clustering import run_clustering
from .data import load_data
from .plotting import create_all_figures
from .statistics import run_statistics


TABLE_FILENAMES = (
    "clustering_summary.csv",
    "clustering_metrics.csv",
    "cluster_assignments.csv",
    "bootstrap_stability.csv",
    "cohort_characteristics.csv",
    "group_comparisons.csv",
    "cll_outcome_correlations.csv",
    "correlation_matrix_long.csv",
    "tissue_associations.csv",
)

FIGURE_STEMS = (
    "clustering_summary",
    "microct_group_comparisons",
    "lacuna_density_analysis",
    "intertrabecular_marrow_sav_analysis",
    "correlation_matrix",
    "tissue_associations",
)


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    bootstrap_iterations: int = 1000,
) -> dict[str, object]:
    """Validate the workbook, run every analysis, and write generated artifacts."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(input_path)
    clustering = run_clustering(
        frame,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    statistical = run_statistics(frame)

    tables = {
        "clustering_summary.csv": clustering.summary,
        "clustering_metrics.csv": clustering.metrics,
        "cluster_assignments.csv": clustering.assignments,
        "bootstrap_stability.csv": clustering.bootstrap_stability,
        "cohort_characteristics.csv": statistical.cohort_characteristics,
        "group_comparisons.csv": statistical.group_comparisons,
        "cll_outcome_correlations.csv": statistical.cll_outcome_correlations,
        "correlation_matrix_long.csv": statistical.correlation_matrix_long,
        "tissue_associations.csv": statistical.tissue_associations,
    }
    for filename, table in tables.items():
        table.to_csv(table_dir / filename, index=False, float_format="%.10g")

    create_all_figures(frame, clustering, statistical, figure_dir, seed=seed)

    summary_values = clustering.summary.set_index("metric")["value"]
    return {
        "input": str(input_path),
        "output": str(output_dir),
        "n_samples": int(len(frame)),
        "n_lo_cll": int((frame["group"] == "Lo-CLL").sum()),
        "n_hi_cll": int((frame["group"] == "Hi-CLL").sum()),
        "silhouette_score": float(summary_values["silhouette_score"]),
        "bootstrap_stability_mean": float(summary_values["bootstrap_stability_mean"]),
        "tables_written": len(tables),
        "figures_written": len(FIGURE_STEMS) * 2,
    }

