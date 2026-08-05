"""Deterministic K-means clustering and corrected bootstrap stability."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_samples, silhouette_score

from .constants import CLL_COLUMN, GROUP_COLUMN, GROUP_ORDER, ID_COLUMN


@dataclass(frozen=True)
class ClusteringResults:
    """All tabular and array outputs from the clustering analysis."""

    summary: pd.DataFrame
    metrics: pd.DataFrame
    assignments: pd.DataFrame
    bootstrap_stability: pd.DataFrame
    labels: np.ndarray
    centroids: np.ndarray
    silhouette_values: np.ndarray


def _normalize_labels(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary K-means labels to ascending centroid order."""

    order = np.argsort(centers.ravel())
    remap = {int(original): int(normalized) for normalized, original in enumerate(order)}
    normalized = np.asarray([remap[int(label)] for label in labels], dtype=int)
    return normalized, centers.ravel()[order]


def _fit_kmeans(values: np.ndarray, k: int, seed: int, n_init: int = 50) -> KMeans:
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(values)


def _bootstrap_stability(
    values: np.ndarray,
    reference_labels: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Refit on bootstrap samples, then predict and align every original subject."""

    if iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")

    rng = np.random.default_rng(seed)
    agreements = np.zeros((iterations, len(values)), dtype=bool)

    for iteration in range(iterations):
        sample_indices = rng.integers(0, len(values), size=len(values))
        model = _fit_kmeans(values[sample_indices], k=2, seed=seed)
        predicted = model.predict(values)
        predicted, _ = _normalize_labels(predicted, model.cluster_centers_)
        agreements[iteration] = predicted == reference_labels

    return agreements.mean(axis=0)


def run_clustering(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    bootstrap_iterations: int = 1000,
) -> ClusteringResults:
    """Run the primary two-cluster model and its supporting diagnostics."""

    values = frame[[CLL_COLUMN]].to_numpy(dtype=float)
    primary_model = _fit_kmeans(values, k=2, seed=seed)
    labels, centroids = _normalize_labels(primary_model.labels_, primary_model.cluster_centers_)

    expected_labels = frame[GROUP_COLUMN].map({GROUP_ORDER[0]: 0, GROUP_ORDER[1]: 1}).to_numpy()
    if not np.array_equal(labels, expected_labels):
        mismatches = frame.loc[labels != expected_labels, [ID_COLUMN, GROUP_COLUMN, CLL_COLUMN]]
        raise ValueError(
            "The supplied group labels do not match the ascending two-cluster solution:\n"
            + mismatches.to_string(index=False)
        )

    sample_silhouettes = silhouette_samples(values, labels)
    primary_silhouette = float(silhouette_score(values, labels))
    primary_ch = float(calinski_harabasz_score(values, labels))
    boundary = float(centroids.mean())

    metric_rows: list[dict[str, object]] = []
    for k in range(2, 6):
        model = _fit_kmeans(values, k=k, seed=seed)
        normalized_labels, normalized_centers = _normalize_labels(
            model.labels_, model.cluster_centers_
        )
        sizes = [int(np.sum(normalized_labels == label)) for label in range(k)]
        metric_rows.append(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(values, normalized_labels)),
                "calinski_harabasz_score": float(
                    calinski_harabasz_score(values, normalized_labels)
                ),
                "centroids": json.dumps([float(value) for value in normalized_centers]),
                "cluster_sizes": json.dumps(sizes),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    stability = _bootstrap_stability(
        values,
        labels,
        iterations=bootstrap_iterations,
        seed=seed,
    )

    assignments = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].to_numpy(),
            CLL_COLUMN: values.ravel(),
            "workbook_group": frame[GROUP_COLUMN].to_numpy(),
            "kmeans_group": np.where(labels == 0, GROUP_ORDER[0], GROUP_ORDER[1]),
            "normalized_cluster_label": labels,
            "silhouette_value": sample_silhouettes,
            "assignment_matches_workbook": labels == expected_labels,
        }
    )

    bootstrap = assignments[[ID_COLUMN, CLL_COLUMN, "workbook_group"]].copy()
    bootstrap["assignment_stability"] = stability

    low_values = values[labels == 0].ravel()
    high_values = values[labels == 1].ravel()
    summary = pd.DataFrame(
        {
            "metric": [
                "n_samples",
                "n_lo_cll",
                "n_hi_cll",
                "lo_centroid",
                "hi_centroid",
                "centroid_midpoint",
                "lo_min",
                "lo_max",
                "hi_min",
                "hi_max",
                "silhouette_score",
                "calinski_harabasz_score",
                "bootstrap_iterations",
                "bootstrap_stability_mean",
                "bootstrap_stability_min",
                "bootstrap_stability_max",
                "group_assignment_matches",
            ],
            "value": [
                len(frame),
                len(low_values),
                len(high_values),
                centroids[0],
                centroids[1],
                boundary,
                low_values.min(),
                low_values.max(),
                high_values.min(),
                high_values.max(),
                primary_silhouette,
                primary_ch,
                bootstrap_iterations,
                stability.mean(),
                stability.min(),
                stability.max(),
                int(np.sum(labels == expected_labels)),
            ],
        }
    )

    return ClusteringResults(
        summary=summary,
        metrics=metrics,
        assignments=assignments,
        bootstrap_stability=bootstrap,
        labels=labels,
        centroids=centroids,
        silhouette_values=sample_silhouettes,
    )

