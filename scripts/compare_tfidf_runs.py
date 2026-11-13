#!/usr/bin/env python3
"""Compare the most common TF-IDF terms across saved analysis runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_run(path: Path, top_n: int) -> pd.DataFrame:
    columns = ["file_url_sha16", "term", "tfidf_score"]
    data = pd.read_csv(path, usecols=columns)
    document_count = data["file_url_sha16"].nunique()
    language_filter = "en" if "detected_language" in pd.read_csv(
        path,
        nrows=0,
    ).columns else "none"

    summary = (
        data.groupby("term", as_index=False)
        .agg(
            document_frequency=("file_url_sha16", "nunique"),
            mean_tfidf_score=("tfidf_score", "mean"),
            max_tfidf_score=("tfidf_score", "max"),
        )
        .sort_values(
            ["document_frequency", "mean_tfidf_score", "max_tfidf_score"],
            ascending=False,
        )
        .head(top_n)
    )
    summary.insert(0, "run_file", path.name)
    summary.insert(1, "language_filter", language_filter)
    summary.insert(2, "document_count", document_count)
    summary["document_frequency_ratio"] = (
        summary["document_frequency"] / document_count
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved TF-IDF result CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="TF-IDF result CSV files to compare",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="number of terms to retain from each run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tfidf_range_comparison.csv"),
        help="comparison CSV path",
    )
    arguments = parser.parse_args()

    if arguments.top_n < 1:
        parser.error("--top-n must be at least 1")

    for path in arguments.inputs:
        if not path.exists():
            parser.error(f"Input file does not exist: {path}")

    return arguments


def main() -> None:
    arguments = parse_args()
    comparison = pd.concat(
        [summarize_run(path, arguments.top_n) for path in arguments.inputs],
        ignore_index=True,
    )
    comparison.to_csv(arguments.output, index=False)
    print(f"Compared {len(arguments.inputs)} runs")
    print(f"Wrote {len(comparison)} term rows to {arguments.output}")


if __name__ == "__main__":
    main()
