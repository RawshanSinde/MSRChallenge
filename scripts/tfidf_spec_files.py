#!/usr/bin/env python3
"""Compute TF-IDF terms for a sample of SpecMine specification files."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


METADATA_COLUMNS = [
    "file_url_sha16",
    "repo_name",
    "file_path",
    "file_name",
    "spec_tool",
    "spec_role",
]

DOMAIN_STOPWORDS = {
    "shall",
    "must",
    "should",
    "may",
    "given",
    "when",
    "then",
    "following",
    "requirement",
    "requirements",
    "spec",
    "specification",
    "specifications",
    "file",
    "files",
    "fr",
    "md",
    "ts",
    "js",
    "jsx",
    "tsx",
    "py",
    "json",
    "yaml",
    "yml",
    "xml",
    "html",
    "css",
    "sql",
}


def load_rows(path: Path, start_row: int, end_row: int) -> pd.DataFrame:
    start_index = start_row - 1
    row_count = end_row - start_row + 1

    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        parquet_file = parquet.ParquetFile(path)
        batches = parquet_file.iter_batches(
            batch_size=end_row,
            columns=METADATA_COLUMNS,
        )
        try:
            metadata = next(batches).slice(start_index, row_count).to_pandas()
        except StopIteration as error:
            raise ValueError(f"No rows found in {path}") from error
    else:
        metadata = pd.read_csv(
            path,
            compression="infer",
            skiprows=range(1, start_row) if start_row > 1 else None,
            nrows=row_count,
            usecols=METADATA_COLUMNS,
        )

    metadata.insert(
        0,
        "source_row_number",
        range(start_row, start_row + len(metadata)),
    )
    return metadata


def iter_metadata_records(path: Path):
    source_row_number = 0

    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        parquet_file = parquet.ParquetFile(path)
        batches = parquet_file.iter_batches(
            batch_size=10_000,
            columns=METADATA_COLUMNS,
        )
        for batch in batches:
            metadata = batch.to_pandas()
            for values in metadata.itertuples(index=False, name=None):
                source_row_number += 1
                yield source_row_number, dict(zip(METADATA_COLUMNS, values))
        return

    for metadata in pd.read_csv(
        path,
        compression="infer",
        usecols=METADATA_COLUMNS,
        chunksize=10_000,
    ):
        for values in metadata.itertuples(index=False, name=None):
            source_row_number += 1
            yield source_row_number, dict(zip(METADATA_COLUMNS, values))


def allocate_stratified_counts(
    strata_counts: Counter[str],
    sample_size: int,
) -> dict[str, int]:
    total_rows = sum(strata_counts.values())
    if sample_size > total_rows:
        raise ValueError(
            f"Requested {sample_size} rows, but the table has only {total_rows} rows"
        )

    strata = sorted(strata_counts)
    if sample_size < len(strata):
        selected_strata = sorted(
            strata,
            key=lambda stratum: strata_counts[stratum],
            reverse=True,
        )[:sample_size]
        return {
            stratum: 1 if stratum in selected_strata else 0
            for stratum in strata
        }

    targets = {stratum: 0 for stratum in strata}
    targets = {stratum: 1 for stratum in strata}

    remaining = sample_size - sum(targets.values())
    if remaining <= 0:
        return targets

    capacities = {
        stratum: strata_counts[stratum] - targets[stratum]
        for stratum in strata
    }
    capacity_total = sum(capacities.values())
    raw_allocations = {
        stratum: remaining * capacities[stratum] / capacity_total
        for stratum in strata
    }

    for stratum in strata:
        targets[stratum] += min(
            capacities[stratum], int(raw_allocations[stratum])
        )

    remaining = sample_size - sum(targets.values())
    remainders = sorted(
        strata,
        key=lambda stratum: raw_allocations[stratum]
        - int(raw_allocations[stratum]),
        reverse=True,
    )
    for stratum in remainders:
        if remaining == 0:
            break
        if targets[stratum] < strata_counts[stratum]:
            targets[stratum] += 1
            remaining -= 1

    return targets


def load_stratified_rows(
    path: Path,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    strata_counts: Counter[str] = Counter()
    for _, record in iter_metadata_records(path):
        stratum = record["spec_tool"] or "unknown"
        strata_counts[str(stratum)] += 1

    target_counts = allocate_stratified_counts(strata_counts, sample_size)
    generator = random.Random(random_state)
    reservoirs: dict[str, list[tuple[int, dict[str, object]]]] = {
        stratum: [] for stratum in target_counts
    }
    seen_counts: Counter[str] = Counter()

    for source_row_number, record in iter_metadata_records(path):
        stratum = str(record["spec_tool"] or "unknown")
        seen_counts[stratum] += 1
        reservoir = reservoirs[stratum]
        target_count = target_counts[stratum]

        if len(reservoir) < target_count:
            reservoir.append((source_row_number, record))
            continue

        replacement_index = generator.randrange(seen_counts[stratum])
        if replacement_index < target_count:
            reservoir[replacement_index] = (source_row_number, record)

    selected_rows = [
        (source_row_number, record)
        for reservoir in reservoirs.values()
        for source_row_number, record in reservoir
    ]
    selected_rows.sort(key=lambda item: item[0])
    metadata = pd.DataFrame(
        [record for _, record in selected_rows],
        columns=METADATA_COLUMNS,
    )
    metadata.insert(
        0,
        "source_row_number",
        [source_row_number for source_row_number, _ in selected_rows],
    )
    return metadata


def load_random_rows(path: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    generator = random.Random(random_state)

    if path.suffix == ".parquet":
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        parquet_file = parquet.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        if sample_size > total_rows:
            raise ValueError(
                f"Requested {sample_size} rows, but {path} has only "
                f"{total_rows} rows"
            )

        selected_rows = sorted(
            generator.sample(range(1, total_rows + 1), sample_size)
        )
        selected_frames: list[pd.DataFrame] = []
        row_group_start = 1

        for row_group_index in range(parquet_file.num_row_groups):
            row_group_rows = parquet_file.metadata.row_group(
                row_group_index
            ).num_rows
            row_group_end = row_group_start + row_group_rows - 1
            group_rows = [
                row_number
                for row_number in selected_rows
                if row_group_start <= row_number <= row_group_end
            ]

            if group_rows:
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=METADATA_COLUMNS,
                )
                local_indices = [
                    row_number - row_group_start for row_number in group_rows
                ]
                metadata = table.take(arrow.array(local_indices)).to_pandas()
                metadata.insert(0, "source_row_number", group_rows)
                selected_frames.append(metadata)

            row_group_start = row_group_end + 1

        return pd.concat(selected_frames, ignore_index=True)

    reservoir: list[dict[str, object]] = []
    row_number = 0
    for chunk in pd.read_csv(
        path,
        compression="infer",
        usecols=METADATA_COLUMNS,
        chunksize=10_000,
    ):
        for values in chunk.itertuples(index=False, name=None):
            row_number += 1
            record = dict(zip(METADATA_COLUMNS, values))
            record["source_row_number"] = row_number

            if len(reservoir) < sample_size:
                reservoir.append(record)
                continue

            replacement_index = generator.randrange(row_number)
            if replacement_index < sample_size:
                reservoir[replacement_index] = record

    if len(reservoir) < sample_size:
        raise ValueError(
            f"Requested {sample_size} rows, but {path} has only {len(reservoir)} rows"
        )

    return pd.DataFrame(reservoir).sort_values("source_row_number").reset_index(
        drop=True
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
        stop_words=sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS),
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
                "row_number": document_metadata["source_row_number"],
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
            "Compute top TF-IDF terms for spec_files rows or samples "
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
        help="number of rows to analyze when --end-row is not provided",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="first 1-based spec_files row to analyze",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="last 1-based spec_files row to analyze, inclusive",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="randomly select this many rows from the full table",
    )
    sampling_group.add_argument(
        "--stratified-sample",
        type=int,
        default=None,
        help="sample this many rows proportionally across spec_tool groups",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="seed used with --random-sample",
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

    if arguments.start_row < 1:
        parser.error("--start-row must be at least 1")
    if arguments.limit < 1:
        parser.error("--limit must be at least 1")
    if arguments.end_row is not None and arguments.end_row < arguments.start_row:
        parser.error("--end-row must be greater than or equal to --start-row")
    if arguments.random_sample is not None and arguments.random_sample < 1:
        parser.error("--random-sample must be at least 1")
    if arguments.stratified_sample is not None and arguments.stratified_sample < 1:
        parser.error("--stratified-sample must be at least 1")
    if (
        arguments.random_sample is not None
        or arguments.stratified_sample is not None
    ) and (
        arguments.start_row != 1 or arguments.end_row is not None
    ):
        parser.error("sampling cannot be combined with a row range")
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

    if arguments.end_row is None:
        arguments.end_row = arguments.start_row + arguments.limit - 1

    return arguments


def main() -> None:
    arguments = parse_args()

    for input_path in (arguments.spec_files, arguments.spec_text):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if arguments.random_sample is not None:
        metadata = load_random_rows(
            arguments.spec_files,
            arguments.random_sample,
            arguments.random_state,
        )
    elif arguments.stratified_sample is not None:
        metadata = load_stratified_rows(
            arguments.spec_files,
            arguments.stratified_sample,
            arguments.random_state,
        )
    else:
        metadata = load_rows(
            arguments.spec_files,
            arguments.start_row,
            arguments.end_row,
        )
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
