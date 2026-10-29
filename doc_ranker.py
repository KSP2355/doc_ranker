from __future__ import annotations

import html
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
except ImportError:  # pragma: no cover - handled at runtime in helper function
    TfidfVectorizer = None
    accuracy_score = None
    classification_report = None
    train_test_split = None
    MultinomialNB = None
    KNeighborsClassifier = None
    make_pipeline = None
    SVC = None

TOKEN_PATTERN = re.compile(r"\b\w+\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    return HTML_TAG_PATTERN.sub(" ", html.unescape(text))


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(clean_html(text).lower())


def rank_datasets(query: str, datasets: list[str], top_k: int | None = None) -> list[dict]:
    documents = [tokenize(document) for document in datasets]
    if not documents:
        return []

    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    total_documents = len(documents)

    def tfidf(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        length = len(tokens)
        if length == 0:
            return {}

        return {
            term: (count / length)
            * (math.log((total_documents + 1) / (document_frequency[term] + 1)) + 1)
            for term, count in counts.items()
        }

    query_vector = tfidf(tokenize(query))
    query_norm = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))

    results = []
    for original_document, tokens in zip(datasets, documents):
        document_vector = tfidf(tokens)
        document_norm = math.sqrt(
            sum(weight ** 2 for weight in document_vector.values())
        )

        dot_product = sum(
            query_weight * document_vector.get(term, 0)
            for term, query_weight in query_vector.items()
        )
        score = (
            dot_product / (query_norm * document_norm)
            if query_norm and document_norm else 0.0
        )
        results.append(
            {
                "dataset": original_document,
                "document": original_document,
                "score": score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k] if top_k else results


def rank_dataset_directory(directory: str | Path, query: str, top_k: int | None = None):
    paths = sorted(Path(directory).glob("*.txt"))
    documents = [path.read_text(encoding="utf-8", errors="replace") for path in paths]

    ranked = rank_datasets(query, documents, top_k)
    for result in ranked:
        dataset_value = result.get("dataset", result.get("document"))
        index = documents.index(dataset_value)
        result["filename"] = paths[index].name

    return ranked


def _require_sklearn() -> None:
    if not all(
        [
            TfidfVectorizer,
            accuracy_score,
            classification_report,
            train_test_split,
            MultinomialNB,
            KNeighborsClassifier,
            make_pipeline,
            SVC,
        ]
    ):
        raise ImportError(
            "scikit-learn is required for classifier comparison. Install it with "
            "`pip install scikit-learn`."
        )


def load_labeled_dataset(directory: str | Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    for path in sorted(Path(directory).glob("*.txt")):
        texts.append(path.read_text(encoding="utf-8", errors="replace"))
        stem_parts = path.stem.rsplit("_", maxsplit=1)
        label = stem_parts[-1] if len(stem_parts) == 2 else path.stem
        labels.append(label)

    return texts, labels


def compare_classifiers(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    _require_sklearn()

    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if len(texts) < 2:
        raise ValueError("At least two labeled documents are required.")
    if len(set(labels)) < 2:
        raise ValueError("At least two classes are required for comparison.")

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    knn_neighbors = max(1, min(5, len(y_train)))
    models = {
        "knn": KNeighborsClassifier(n_neighbors=knn_neighbors),
        "svm": SVC(kernel="linear", random_state=random_state),
        "naive_bayes": MultinomialNB(),
    }

    results: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        pipeline = make_pipeline(TfidfVectorizer(), model)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        results[model_name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions, zero_division=0),
            "predictions": list(predictions),
        }

    return results


def compare_classifiers_in_directory(
    directory: str | Path,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    texts, labels = load_labeled_dataset(directory)
    return compare_classifiers(
        texts=texts,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    query = input("Enter your search query: ").strip()
    results = rank_dataset_directory("IR_30dataset", query, top_k=10)

    for position, result in enumerate(results, start=1):
        print(f"{position}. {result['filename']} — score: {result['score']:.4f}")
