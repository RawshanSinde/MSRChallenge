#!/usr/bin/env python3
"""Compute TF-IDF terms for a sample of SpecMine specification files."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


METADATA_COLUMNS = [
    "file_url_sha16",
    "repo_name",
    "file_path",
    "file_name",
    "spec_tool",
    "spec_role",
]


def load_first_rows(path: Path, limit: int) -> pd.DataFrame:
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        parquet_file = parquet.ParquetFile(path)
        batches = parquet_file.iter_batches(
            batch_size=limit,
            columns=METADATA_COLUMNS,
        )
        try:
            return next(batches).to_pandas()
        except StopIteration as error:
            raise ValueError(f"No rows found in {path}") from error

    return pd.read_csv(
        path,
        compression="infer",
        nrows=limit,
        usecols=METADATA_COLUMNS,
    )


def load_spec_texts(path: Path, file_ids: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}

    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

            file_id = record.get("file_url_sha16")
            if file_id not in file_ids:
                continue

            content = record.get("content")
            if content is None:
                content = record.get("file_content")
            texts[file_id] = content or ""

            if len(texts) == len(file_ids):
                break

    return texts


def detect_languages(
    texts: list[str],
    minimum_characters: int,
) -> list[tuple[str, float]]:
    try:
        from langdetect import DetectorFactory, LangDetectException, detect_langs
    except ImportError as error:
        raise RuntimeError(
            "Language filtering requires langdetect. Install it with "
            "python -m pip install langdetect."
        ) from error

    DetectorFactory.seed = 0
    detections: list[tuple[str, float]] = []

    for text in texts:
        if len(text.strip()) < minimum_characters:
            detections.append(("unknown", 0.0))
            continue

        try:
            candidates = detect_langs(text)
        except LangDetectException:
            detections.append(("unknown", 0.0))
            continue

        if not candidates:
            detections.append(("unknown", 0.0))
            continue

        best_match = candidates[0]
        detections.append((best_match.lang, float(best_match.prob)))

    return detections


def build_output(
    metadata: pd.DataFrame,
    texts: list[str],
    top_n: int,
    max_features: int | None,
) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as error:
        raise ValueError(
            "TF-IDF could not find any terms in the selected documents."
        ) from error

    terms = vectorizer.get_feature_names_out()
    output_rows: list[dict[str, object]] = []

    for document_index in range(matrix.shape[0]):
        document_scores = matrix.getrow(document_index)
        order = document_scores.data.argsort()[::-1][:top_n]

        for score_index in order:
            document_metadata = metadata.iloc[document_index]
            output_row = {
                "row_number": document_index + 1,
                "file_url_sha16": document_metadata["file_url_sha16"],
                "repo_name": document_metadata["repo_name"],
                "file_path": document_metadata["file_path"],
                "file_name": document_metadata["file_name"],
                "spec_tool": document_metadata["spec_tool"],
                "spec_role": document_metadata["spec_role"],
                "term": terms[document_scores.indices[score_index]],
                "tfidf_score": float(document_scores.data[score_index]),
            }
            if "detected_language" in metadata:
                output_row["detected_language"] = document_metadata[
                    "detected_language"
                ]
                output_row["language_probability"] = document_metadata[
                    "language_probability"
                ]
            output_rows.append(output_row)

    return pd.DataFrame(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute top TF-IDF terms for the first rows of spec_files "
            "using their text from specs.jsonl.gz."
        )
    )
    parser.add_argument(
        "--spec-files",
        type=Path,
        default=Path("data/spec_files.parquet"),
        help="spec_files Parquet or CSV/CSV.GZ file",
    )
    parser.add_argument(
        "--spec-text",
        type=Path,
        default=Path("specs.jsonl.gz"),
        help="gzipped JSONL file containing specification text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="number of spec_files rows to analyze",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="number of terms to write per document",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="keep only documents detected as this language, for example en",
    )
    parser.add_argument(
        "--min-language-probability",
        type=float,
        default=0.80,
        help="minimum langdetect confidence when --language is used",
    )
    parser.add_argument(
        "--min-language-chars",
        type=int,
        default=100,
        help="minimum non-whitespace characters required for detection",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="optional maximum vocabulary size",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tfidf_spec_files_top_terms.csv"),
        help="output CSV path",
    )
    arguments = parser.parse_args()

    if arguments.limit < 1:
        parser.error("--limit must be at least 1")
    if arguments.top_n < 1:
        parser.error("--top-n must be at least 1")
    if not 0 <= arguments.min_language_probability <= 1:
        parser.error("--min-language-probability must be between 0 and 1")
    if arguments.min_language_chars < 1:
        parser.error("--min-language-chars must be at least 1")
    if arguments.max_features is not None and arguments.max_features < 1:
        parser.error("--max-features must be at least 1")

    if arguments.language is not None:
        arguments.language = arguments.language.lower()

    return arguments


def main() -> None:
    arguments = parse_args()

    for input_path in (arguments.spec_files, arguments.spec_text):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

    metadata = load_first_rows(arguments.spec_files, arguments.limit)
    file_ids = set(metadata["file_url_sha16"].dropna().astype(str))
    text_by_id = load_spec_texts(arguments.spec_text, file_ids)
    texts = [text_by_id.get(str(file_id), "") for file_id in metadata["file_url_sha16"]]

    if arguments.language is not None:
        detections = detect_languages(texts, arguments.min_language_chars)
        metadata = metadata.copy()
        metadata["detected_language"] = [
            language for language, _ in detections
        ]
        metadata["language_probability"] = [
            probability for _, probability in detections
        ]
        keep_rows = [
            language == arguments.language
            and probability >= arguments.min_language_probability
            for language, probability in detections
        ]
        metadata = metadata.loc[keep_rows].reset_index(drop=True)
        texts = [text for text, keep in zip(texts, keep_rows) if keep]
        print(
            f"Language filter: kept {len(metadata)} of {len(keep_rows)} "
            f"rows as {arguments.language}"
        )

        if metadata.empty:
            raise ValueError(
                "No documents matched the requested language filter."
            )

    output = build_output(
        metadata=metadata,
        texts=texts,
        top_n=arguments.top_n,
        max_features=arguments.max_features,
    )
    output.to_csv(arguments.output, index=False)

    missing_count = len(file_ids - text_by_id.keys())
    print(f"Analyzed {len(metadata)} rows from {arguments.spec_files}")
    print(f"Missing text records: {missing_count}")
    print(f"Wrote {len(output)} term rows to {arguments.output}")


if __name__ == "__main__":
    main()
