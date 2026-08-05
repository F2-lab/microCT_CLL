from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

import numpy as np

from microct_cll.clustering import run_clustering
from microct_cll.constants import (
    BMI_COLUMN,
    GROUP_COLUMN,
    HIGHLIGHT_OUTCOMES,
    REQUIRED_COLUMNS,
)
from microct_cll.data import load_data
from microct_cll.pipeline import FIGURE_STEMS, TABLE_FILENAMES, run_pipeline
from microct_cll.statistics import run_statistics


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPOSITORY_ROOT / "Supplementary Data 2.xlsx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AnalysisRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_data(WORKBOOK)
        cls.clustering = run_clustering(
            cls.frame, seed=42, bootstrap_iterations=1000
        )
        cls.statistics = run_statistics(cls.frame)

    def test_workbook_shape_and_missingness(self) -> None:
        self.assertEqual(self.frame.shape, (18, 21))
        self.assertEqual(tuple(self.frame.columns), REQUIRED_COLUMNS)
        self.assertEqual(self.frame[GROUP_COLUMN].value_counts().to_dict(), {"Lo-CLL": 11, "Hi-CLL": 7})
        self.assertEqual(int(self.frame[BMI_COLUMN].isna().sum()), 1)
        self.assertEqual(int(self.frame["CC#   Seq"].duplicated().sum()), 0)

    def test_clustering_regression_values(self) -> None:
        summary = self.clustering.summary.set_index("metric")["value"]
        np.testing.assert_allclose(
            self.clustering.centroids,
            np.array([0.3790909090909091, 71.28571428571429]),
            rtol=0,
            atol=1e-10,
        )
        self.assertAlmostEqual(float(summary["silhouette_score"]), 0.831838, places=6)
        self.assertAlmostEqual(
            float(summary["calinski_harabasz_score"]), 128.042987, places=6
        )
        self.assertAlmostEqual(
            float(summary["bootstrap_stability_mean"]), 0.984388888888889, places=12
        )
        self.assertEqual(int(summary["group_assignment_matches"]), 18)
        self.assertTrue(self.clustering.assignments["assignment_matches_workbook"].all())

    def test_primary_statistical_regression_values(self) -> None:
        comparisons = self.statistics.group_comparisons.set_index("variable")
        correlations = self.statistics.cll_outcome_correlations.set_index("outcome")
        density, sav = HIGHLIGHT_OUTCOMES

        self.assertAlmostEqual(
            float(comparisons.loc[density, "student_p"]), 0.0354982, places=7
        )
        self.assertAlmostEqual(
            float(comparisons.loc[sav, "student_p"]), 0.0308848, places=7
        )
        self.assertAlmostEqual(
            float(correlations.loc[density, "pearson_p"]), 0.0386513, places=7
        )
        self.assertAlmostEqual(
            float(correlations.loc[sav, "pearson_p"]), 0.0171691, places=7
        )

    def test_workbook_release_safety(self) -> None:
        with zipfile.ZipFile(WORKBOOK) as archive:
            names = set(archive.namelist())
            xml_text = "\n".join(
                archive.read(name).decode("utf-8", errors="ignore")
                for name in names
                if name.endswith((".xml", ".rels"))
            )

        lowered = xml_text.lower()
        self.assertNotIn("sharepoint", lowered)
        self.assertNotIn("lee, in kyu", lowered)
        self.assertNotIn("revisionptr", lowered)
        self.assertNotIn("coauthversion", lowered)
        self.assertNotIn("externalink", lowered)
        self.assertNotIn("threadedcomment", lowered)
        self.assertNotIn("docProps/core.xml", names)
        self.assertIsNone(re.search(r"<f(?:\s|>)", xml_text))
        self.assertNotIn('hidden="1"', xml_text)
        self.assertNotIn("<hyperlink", lowered)

    def test_end_to_end_outputs_and_read_only_input(self) -> None:
        before = sha256(WORKBOOK)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "results"
            summary = run_pipeline(
                WORKBOOK,
                output,
                seed=42,
                bootstrap_iterations=25,
            )
            table_names = {path.name for path in (output / "tables").glob("*.csv")}
            figure_names = {path.name for path in (output / "figures").iterdir()}
            expected_figures = {
                f"{stem}.{extension}"
                for stem in FIGURE_STEMS
                for extension in ("png", "pdf")
            }
            self.assertEqual(table_names, set(TABLE_FILENAMES))
            self.assertEqual(figure_names, expected_figures)
            self.assertTrue(all(path.stat().st_size > 0 for path in (output / "figures").iterdir()))
            self.assertEqual(summary["n_samples"], 18)

        self.assertEqual(before, sha256(WORKBOOK))


if __name__ == "__main__":
    unittest.main()

