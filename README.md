# CLL microCT clustering and statistical analysis

This repository reproduces the clustering and statistical analyses for the CLL microCT study using the single public input workbook, `Supplementary Data 2.xlsx`. It contains no segmentation, image-processing, model-training, or quantification code.

## Data

The workbook contains 18 unique bone-marrow samples and 21 columns:

- 11 Lo-CLL samples and 7 Hi-CLL samples.
- Complete values for all clustering and microCT outcome variables.
- One missing BMI value. BMI is never imputed; analyses use pairwise-complete observations and report the resulting sample size.

The analysis never writes to the input workbook. Its Lo-CLL/Hi-CLL labels are used for the paper statistics and are independently checked against the ascending two-cluster K-means solution. The K-means centroid midpoint is reported as a diagnostic; it is not used to redefine the supplied groups.

## Installation

Python 3.12 is recommended. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

## Run the analysis

The default command reads the workbook in the repository root and writes generated files under the ignored `results/` directory:

```bash
python run_analysis.py
```

The full explicit command is:

```bash
python run_analysis.py \
  --input "Supplementary Data 2.xlsx" \
  --output-dir results \
  --seed 42 \
  --bootstrap-iterations 1000
```

Generated tables are written to `results/tables/`; PNG and PDF figures are written to `results/figures/`.

## Analyses

### Clustering

- One-dimensional K-means clustering of CLL marrow involvement with `k=2`, `random_state=42`, and `n_init=50`.
- Cluster labels normalized by ascending centroid, then checked against every workbook group label.
- Silhouette and Calinski-Harabasz diagnostics for `k=2` through `k=5`.
- Per-subject silhouette coefficients and cluster assignments.
- Corrected bootstrap stability: each bootstrap sample is refit, its centroids are ordered, and the fitted model predicts every original subject before agreement is calculated. This avoids the subject-index misalignment in the exploratory source notebook.

### Paper-primary statistics

The eight prespecified outcomes are:

1. Median lacuna volume
2. Median lacuna aspect ratio
3. Lacuna density per mm³
4. Tissue mineral peak density
5. Intertrabecular marrow density
6. Normalized intertrabecular marrow surface-area-to-volume ratio
7. BV/TV
8. SV/TV (reported as IMV/TV in figures)

Primary results reproduce the paper-style analyses:

- Two-sided equal-variance Student t tests for continuous group comparisons.
- Two-sided Pearson correlations for continuous associations.
- Two-sided Fisher exact test for gender because of the small contingency table.
- Raw p-values for paper figure annotations.

### Sensitivity analyses and multiplicity

Sensitivity columns are labeled rather than substituted for the paper-primary results:

- Welch t tests and Mann-Whitney U tests for continuous group comparisons.
- Spearman correlations for continuous associations.
- Benjamini-Hochberg q-values calculated separately within each declared family:
  - eight microCT group comparisons;
  - eight CLL-outcome correlations;
  - 55 unique pairs in the full 11-variable correlation analysis;
  - six highlighted tissue associations; and
  - three demographic group comparisons.

No observation is removed globally because of a missing covariate, and no missing value is filled or hardcoded.

## Output tables

- `clustering_summary.csv`: primary centroids, ranges, diagnostics, concordance, and stability summary.
- `clustering_metrics.csv`: silhouette and Calinski-Harabasz results for `k=2..5`.
- `cluster_assignments.csv`: normalized assignments and per-subject silhouettes.
- `bootstrap_stability.csv`: per-subject corrected assignment stability.
- `cohort_characteristics.csv`: age, BMI, gender, and CLL involvement summaries/tests.
- `group_comparisons.csv`: primary and sensitivity results for the eight outcomes.
- `cll_outcome_correlations.csv`: Pearson, Spearman, regression, and FDR results.
- `correlation_matrix_long.csv`: all 55 unique pairwise correlations with pairwise sample sizes.
- `tissue_associations.csv`: age, BMI, and gender analyses for the two highlighted outcomes.

## Verification

Run the regression and end-to-end tests with:

```bash
python -m unittest discover -s tests -v
```

The tests verify the dataset shape and missingness, key clustering/statistical values, exact K-means/group concordance, workbook release safety, all expected outputs, and that an analysis run leaves the input workbook hash unchanged.

## Source audit

The implementation was refactored from the maintained clustering logic and the workbook-supported portions of the later journal-figure notebook in the private research workspace. The following exploratory material was intentionally excluded:

- The older statistical notebook, which used 19 cases, obsolete column names, and only one of the two final exclusions.
- Treatment-adjusted sensitivity analyses, because the public workbook has no treatment variables.
- Intertrabecular marrow component-count analyses, because that outcome is not present in the public workbook.
- The exploratory “longitudinal” cells, which do not represent repeated rows in the public workbook.
- Image histograms, segmentation, quantification, and neural-network code, which precede the supplied tabular data and are outside this repository's scope.

The refactor also corrects a misspelled group label in the original age test, removes inconsistent BMI mean/hardcoded filling, uses Fisher's exact test for the small gender table, updates the final intertrabecular marrow column names, fixes random-state handling, and replaces the invalid bootstrap subject mapping.

## License

Code in this repository is available under the BSD 3-Clause License. The supplementary workbook remains the study's published data artifact.

