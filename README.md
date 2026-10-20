# SpecMine v1.0 — full dataset (July 2026 snapshot)

SpecMine is a large-scale corpus of **Spec-Driven Development (SDD)** artifacts
mined from public GitHub repositories: the Markdown specifications that AI coding
agents turn into code, attributed to the tool that produced them, enriched with
full repository metadata and commit history, and, for a subsample of
repositories, followed into the pull requests that change them.

This Zenodo record is the **complete v1.0 dataset**. A 500-repository sample, the
loader scripts, and an example notebook are in the GitHub mirror:
<https://github.com/shyamagarwal13/specmine-official>. The dataset is the basis of
the MSR 2027 Mining Challenge.

## Contents

| Layer | Count |
|---|---:|
| Broad census (`spec.md`/`specs.md`, 17 named tools) | 470,795 specs / 73,030 repos |
| Kiro census (`.kiro/specs` requirements/design/tasks) | 98,574 artifacts / 12,910 repos |
| PR layer (spec-touching PRs, ≥10-star repos, 11 tools swept) | 5,992 PRs / 581 repos |
| OpenSpec change artifacts | 266,230 |
| Traceability index (typed references) | 2,421,323 |
| Spec-file commits | 780,335 |

```
data/                             Parquet, one file per released table (zstd)
  spec_files.parquet              broad-census spine (repo object, provenance, attribution)
  spec_file_commits.parquet       per-file commit history
  spec_content_features.parquet   39 structural / requirement-template features per spec
  spec_links.parquet              traceability index (typed references)
  kiro_files.parquet              Kiro requirements / design / tasks (metadata)
  openspec_artifact_files.parquet OpenSpec proposals / designs / task lists (+ content)
  openspec_code_refs.parquet      OpenSpec task-to-code refs resolved on the git tree
  pull_requests.parquet           PR layer: the 11 <tool>_prs tables unioned (10 non-empty), with a `tool` column
  pr_files.parquet                PR layer: per-file changesets, each file flagged spec/code
  repo_trees.parquet              git-tree metadata references resolve against
data_csv/*.csv.gz                 the same ten tables as gzipped CSV
specmine-mysql.sql.gz             MySQL dump (DROP/CREATE + INSERTs; restore with: gunzip -c specmine-mysql.sql.gz | mysql <db>)
specs.jsonl.gz                    raw Markdown of every spec (one JSON object per line)
schema.sql                        CREATE TABLE for each shipped table (MySQL types)
DATA_DICTIONARY.md                every shipped column, typed and described
README.md                         this file
LICENSE.txt                       license terms
```

The dataset is provided in three interchangeable forms so you can start however
you like: **Parquet** (`data/`, read directly with pandas/polars/DuckDB),
**CSV** (`data_csv/`), and a **MySQL dump** (`specmine-mysql.sql.gz`). Spec text
is separate in `specs.jsonl.gz` and joins back on `file_url_sha16`. A per-table
Parquet mirror is also on Hugging Face (see the paper for the link).

Every identifier (`repo_name`, `file_url`, `pr_number`) is a live GitHub value, so
any row can be re-fetched or joined to external sources.

## Quick start

```bash
pip install pandas pyarrow scikit-learn langdetect  # duckdb optional
```

```python
import pandas as pd
prs = pd.read_parquet("data/pull_requests.parquet")
prf = pd.read_parquet("data/pr_files.parquet")

# "which specs change in the same PR as code?" — the flagship spec->code query
cochange = prs[prs.touches_code == 1]
files = prf.merge(cochange[["tool", "repo_name", "pr_number"]],
                  on=["tool", "repo_name", "pr_number"])
print(files[files.is_spec == 1].shape, "spec files changed alongside code")
```

DuckDB (query the Parquet directly):

```sql
SELECT tool, COUNT(*) AS cochange_prs
FROM 'data/pull_requests.parquet'
WHERE touches_code = 1
GROUP BY tool ORDER BY cochange_prs DESC;
```

Sample TF-IDF analysis of the first 50 `spec_files` rows:

```bash
python scripts/tfidf_spec_files.py --limit 50 --top-n 20
```

The script reads metadata from `data/spec_files.parquet`, joins the matching
Markdown from `specs.jsonl.gz`, and writes the top terms for each document to
`tfidf_spec_files_top_terms.csv`.

To keep only confidently detected English documents in the sample:

```bash
python scripts/tfidf_spec_files.py \
  --limit 50 \
  --language en \
  --min-language-probability 0.80
```

Language filtering applies to the selected sample and does not modify the raw
Parquet or JSONL files. The output includes the detected language and its
confidence score.

Spec text joins to `spec_files` on `file_url_sha16`:

```python
import gzip, json
with gzip.open("specs.jsonl.gz", "rt") as f:
    specs = [json.loads(l) for l in f]   # {file_url_sha16, repo_name, file_path, spec_tool, spec_role, content}
```

MySQL: create the tables with `schema.sql`, then load each Parquet (e.g. with
`pandas.DataFrame.to_sql`, or convert to CSV and `LOAD DATA`).

## Notes on the release

- **PR layer.** The internal schema stores the PR layer as 11 per-tool table
  pairs (`<tool>_prs`/`<tool>_pr_files`). This release unions them into
  `pull_requests.parquet` / `pr_files.parquet` with a `tool` column. The sweep
  targeted 11 tools; `specweave` yielded no spec-touching PRs, so 10 tools appear.
  A PR enters the sweep when a commit that modified a spec file is associated with
  it, while the changeset is the PR's net diff; the two can disagree, so 5 of the
  5,992 PRs carry `n_spec_files = 0` (3 truncated at GitHub's 3,000-file-per-PR
  cap, 2 whose spec commit sits in shared history). Filter on `n_spec_files > 0`
  if you need every PR to contain a visible spec file.
- **Kept set.** This release ships the kept specs (`not_spec_filter = 0`); package
  caches, vendored trees, and pre-2024 files are excluded.
- **Dropped fields.** Commit-email addresses, avatar URLs, and GPG-signature
  blobs are removed; raw `file_content` ships in `specs.jsonl.gz`. Author and
  committer logins and numeric IDs are kept (public, and needed for authorship
  analysis).

## Licensing, ethics, and privacy

All artifacts come from public GitHub repositories, collected within the API
terms of service. Every repository row keeps its SPDX license (`repo_license_spdx_id`)
so you can filter to license-compatible subsets; redistribution of any file
follows its repository's own terms. The SpecMine compilation is released under
**CC BY 4.0** (see `LICENSE.txt`). Public handles are kept; private contact
details are not. Please avoid deanonymization and keep personal data out of
published results beyond aggregate analysis.

## Citation

```bibtex
@inproceedings{specmine2027,
  author    = {Agarwal, Shyam and Vasilescu, Bogdan},
  title     = {SpecMine: A Large-Scale Corpus of Spec-Driven Development Artifacts},
  year      = {2027},
  booktitle = {Proceedings of the 24th International Conference on Mining Software Repositories},
  series    = {MSR '27}
}
```
