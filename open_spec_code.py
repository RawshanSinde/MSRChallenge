import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

proposal_df = pd.read_parquet("english_proposals.parquet")
print("Number of English proposals:", len(proposal_df))

# Folder to store analysis results
Path("sample_2000_results").mkdir(exist_ok=True)

# Get English stop words for text preprocessing
nltk.download("stopwords")
stop_words = stopwords.words("english")

def process_text(text):
    # Convert text to lowercase and tokenize it
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())

    # Remove common English stop words
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Remove single-character tokens and numbers
    words = [
        word for word in words
        if len(word) > 1 and not word.isdigit()
    ]

    return words

# Randomly select a sample of proposals
sample = proposal_df.sample(n=2000, random_state=42)
print("Number of repositories in sample:", sample["repo_name"].nunique())

# Check for proposals with identical content
duplicate_count = sample["content_sha"].duplicated().sum()
print("\nNumber of duplicate proposals:", duplicate_count)

# Count unique proposal contents
unique_count = sample["content_sha"].nunique()
print("Number of unique proposal contents:", unique_count)

# Count how many proposals come from each repository
proposals_per_repo = sample["repo_name"].value_counts()
print("\nRepositories with multiple proposals:")
print(proposals_per_repo[proposals_per_repo > 1].head(20))

# Bag of words:
# Preprocess each proposal and convert tokens back into text
processed_text = []

for text in sample["content"]:
    words = process_text(text)
    processed_text.append(" ".join(words))

vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(processed_text)

# Count the total number of times each word appears
word_counts = X.sum(axis=0)
words = vectorizer.get_feature_names_out()
counts = []

for i in range(len(words)):
    counts.append([words[i], word_counts[0, i]])

# Store and sort total word frequencies
count_df = pd.DataFrame(
    counts,
    columns=["word", "count"]
)

count_df = count_df.sort_values(
    by="count",
    ascending=False
)

print("\nTop words by total frequency:")
print(count_df.head(30))

# Count how many proposals contain each word
document_counts = (X > 0).sum(axis=0)
doc_counts = []

for i in range(len(words)):
    doc_counts.append([words[i], document_counts[0, i]])

# Store and sort document frequencies
doc_count_df = pd.DataFrame(
    doc_counts,
    columns=["word", "number_of_proposals"]
)

doc_count_df = doc_count_df.sort_values(
    by="number_of_proposals",
    ascending=False
)

print("\nTop words by number of proposals:")
print(doc_count_df.head(30))

# Markdown heading analysis part:
heading_counts = {}

for text in sample["content"]:
    headings = []

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()

            # Keep headings containing only ASCII characters
            if heading.isascii():
                headings.append(heading)

    # Count each heading once per proposal
    for heading in set(headings):
        if heading in heading_counts:
            heading_counts[heading] += 1
        else:
            heading_counts[heading] = 1

heading_df = pd.DataFrame(
    heading_counts.items(),
    columns=["heading", "number_of_proposals"]
)

heading_df = heading_df.sort_values(
    by="number_of_proposals",
    ascending=False
)

print("\nMost common English proposal headings:")
print(heading_df.head(30))

# Calculate the percentage of words in Markdown headings
heading_percentages = []

for text in sample["content"]:
    total_words = len(text.split())
    heading_words = 0

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            heading_words += len(heading.split())

    if total_words > 0:
        percentage = (heading_words / total_words) * 100
        heading_percentages.append(percentage)

# Calculate the average percentage across proposals
average_percentage = sum(heading_percentages) / len(heading_percentages)

print("\nAverage percentage of words in headings:",
      round(average_percentage, 2), "%")

# Repeated lines of text analysis part:
line_counts = {}

for text in sample["content"]:
    lines = []

    for line in text.split("\n"):
        line = line.strip().lower()

        # Ignore headings and lines with fewer than 4 words
        if len(line.split()) >= 4 and not line.startswith("#"):
            lines.append(line)

    # Count each line once per proposal
    for line in set(lines):
        if line in line_counts:
            line_counts[line] += 1
        else:
            line_counts[line] = 1

# Store and sort repeated lines
line_df = pd.DataFrame(
    line_counts.items(),
    columns=["line", "number_of_proposals"]
)

line_df = line_df.sort_values(
    by="number_of_proposals",
    ascending=False
)

# Keep lines appearing in at least 2 proposals
repeated_df = line_df[
    line_df["number_of_proposals"] >= 2
]

print("\nRepeated lines with at least 4 words:")
print(repeated_df.head(30))
print("\nNumber of unique repeated lines:", repeated_df.shape[0])

# Count how many different repositories contain each line
line_repos = {}

for index, row in sample.iterrows():
    repo = row["repo_name"]
    text = row["content"]

    for line in text.split("\n"):
        line = line.strip().lower()

        # Ignore headings and lines with fewer than 4 words
        if len(line.split()) >= 4 and not line.startswith("#"):

            if line not in line_repos:
                line_repos[line] = set()

            line_repos[line].add(repo)

# Store the number of repositories containing each line
line_repo_counts = []

for line in line_repos:
    line_repo_counts.append([
        line,
        len(line_repos[line])
    ])

repo_df = pd.DataFrame(
    line_repo_counts,
    columns=["line", "number_of_repositories"]
)

# Keep lines appearing in at least 2 repositories
repo_df = repo_df[
    repo_df["number_of_repositories"] >= 2
]

repo_df = repo_df.sort_values(
    by="number_of_repositories",
    ascending=False
)

print("\nRepeated lines across different repositories:")
print(repo_df.head(30))
print("\nNumber of lines repeated across repositories:",
      repo_df.shape[0])

# Compare repeated lines within and across repositories
within_repo_lines = 0
across_repo_lines = 0

for line in repeated_df["line"]:
    if len(line_repos[line]) >= 2:
        across_repo_lines += 1
    else:
        within_repo_lines += 1

print("\nRepeated line breakdown:")
print("Lines repeated only within repositories:", within_repo_lines)
print("Lines repeated across repositories:", across_repo_lines)

# Identify potential boilerplate lines shared across repositories
minimum_repos = 2

potential_boilerplate_df = repo_df[
    repo_df["number_of_repositories"] >= minimum_repos
].copy()

print("\nPotential textual boilerplate candidates:")
print(potential_boilerplate_df.head(30))

# Save candidates for manual review
potential_boilerplate_df.to_csv(
    "sample_2000_results/potential_boilerplate_lines.csv",
    index=False
)

# Create a file for reviewing potential boilerplate lines
review_file = Path("sample_2000_results/boilerplate_review.csv")

if not review_file.exists():
    review_df = potential_boilerplate_df.copy()
    review_df["classification"] = "unreviewed"
    review_df.to_csv(review_file, index=False)

# Load manually reviewed boilerplate classifications
review_df = pd.read_csv(review_file)

# Keep lines classified as template formatting or instructions
reviewed_boilerplate = review_df[
    review_df["classification"].isin([
        "template_formatting",
        "template_instruction"
    ])
]

reviewed_lines = set(reviewed_boilerplate["line"])

print("\nNumber of reviewed boilerplate lines:", len(reviewed_lines))

# Count the number of lines in each classification
classification_counts = review_df["classification"].value_counts()

print("\nManual classification breakdown:")
print(classification_counts)

# Calculate percentage of candidates in each category
classification_percentages = (
    classification_counts / len(review_df)
) * 100

print("\nClassification percentages:")
print(classification_percentages.round(2))

# Structural boilerplate part:
# Identify headings appearing in at least 50% of proposals (percentage can be tweaked)
common_headings = []

for index, row in heading_df.iterrows():
    if row["number_of_proposals"] >= len(sample) * 0.5:
        common_headings.append(row["heading"])

print("\nPotential structural boilerplate headings:")
print(common_headings)

# Calculate percentage of words in common template headings
structural_percentages = []

for text in sample["content"]:
    total_words = len(text.split())
    structural_words = 0

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()

            if heading in common_headings:
                structural_words += len(heading.split())

    if total_words > 0:
        percentage = (structural_words / total_words) * 100
        structural_percentages.append(percentage)

average_structural = sum(structural_percentages) / len(structural_percentages)

print("\nAverage potential structural boilerplate:",
      round(average_structural, 2), "%")

# Calculate percentage of words matching reviewed boilerplate lines
reviewed_percentages = []

for text in sample["content"]:
    total_words = len(text.split())
    boilerplate_words = 0

    for line in text.split("\n"):
        line = line.strip().lower()

        if line in reviewed_lines:
            boilerplate_words += len(line.split())

    if total_words > 0:
        percentage = (boilerplate_words / total_words) * 100
        reviewed_percentages.append(percentage)

average_reviewed = sum(reviewed_percentages) / len(reviewed_percentages)

print("\nAverage reviewed boilerplate percentage:",
      round(average_reviewed, 2), "%")

# Textual boilerplate part:
# Known template instructions and placeholders
template_lines = [
    "describe the background and motivation.",
    "list the proposed changes at a high level.",
    "- goal 1: ...",
    "- goal 2: ..."
]

# Calculate percentage of words in known template lines
textual_percentages = []

for text in sample["content"]:
    total_words = len(text.split())
    textual_words = 0

    for line in text.split("\n"):
        line = line.strip().lower()

        if line in template_lines:
            textual_words += len(line.split())

    if total_words > 0:
        percentage = (textual_words / total_words) * 100
        textual_percentages.append(percentage)

average_textual = sum(textual_percentages) / len(textual_percentages)

print("\nAverage potential textual boilerplate:",
      round(average_textual, 2), "%")

# Calculate percentage not classified as boilerplate
remaining_percentage = 100 - average_structural - average_textual

print("\nPreliminary content breakdown:")
print("Structural boilerplate:", round(average_structural, 2), "%")
print("Textual boilerplate:", round(average_textual, 2), "%")
print("Unclassified content:", round(remaining_percentage, 2), "%")

# Save analysis sample_2000_results to CSV files:
# Bag-of-Words sample_2000_results
count_df.to_csv("sample_2000_results/word_frequencies.csv", index=False)
doc_count_df.to_csv("sample_2000_results/document_frequencies.csv", index=False)

# Markdown heading sample_2000_results
heading_df.to_csv("sample_2000_results/heading_frequencies.csv", index=False)

# Repeated line sample_2000_results
repeated_df.to_csv("sample_2000_results/repeated_lines.csv", index=False)
repo_df.to_csv("sample_2000_results/cross_repository_repeated_lines.csv", index=False)

# Store summary statistics
summary_df = pd.DataFrame([
    ["Total English proposals", len(proposal_df)],
    ["Sample size", len(sample)],
    ["Repositories in sample", sample["repo_name"].nunique()],
    ["Duplicate proposals", duplicate_count],
    ["Unique proposal contents", unique_count],
    ["Repeated lines", repeated_df.shape[0]],
    ["Repeated lines across repositories", repo_df.shape[0]],
    ["Average heading percentage", round(average_percentage, 2)],
    ["Structural boilerplate percentage", round(average_structural, 2)],
    ["Textual boilerplate percentage", round(average_textual, 2)],
    ["Reviewed boilerplate percentage", round(average_reviewed, 2)],
    ["Unclassified content percentage", round(remaining_percentage, 2)]
], columns=["metric", "value"])

# Save summary statistics
summary_df.to_csv("sample_2000_results/summary.csv", index=False)

# Save potential structural boilerplate headings
common_headings_df = heading_df[
    heading_df["heading"].isin(common_headings)
]

common_headings_df.to_csv(
    "sample_2000_results/common_template_headings.csv",
    index=False
)

# Save the sampled proposal ids
sample[["id", "repo_name", "file_path", "content_sha"]].to_csv(
    "sample_2000_results/sample_proposals.csv",
    index=False
)

# Save manual classification breakdown
classification_df = pd.DataFrame({
    "count": classification_counts,
    "percentage": classification_percentages.round(2)
})

classification_df.to_csv(
    "sample_2000_results/classification_breakdown.csv"
)
