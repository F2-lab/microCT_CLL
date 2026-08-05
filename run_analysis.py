#!/usr/bin/env python3
"""Command-line entry point for the public CLL microCT analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microct_cll.data import DataValidationError
from microct_cll.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clustering and statistical analyses for Supplementary Data 2.xlsx."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Supplementary Data 2.xlsx"),
        help="Sole input workbook (default: Supplementary Data 2.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated tables and figures (default: results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for K-means, bootstrap sampling, and plot jitter.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of corrected clustering bootstrap iterations (default: 1000).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_pipeline(
            args.input,
            args.output_dir,
            seed=args.seed,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    except (DataValidationError, ValueError) as error:
        raise SystemExit(f"Analysis failed: {error}") from error

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

