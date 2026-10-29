# doc_ranker
by using the TF-IDF algorithm the documents are ranked.

You can also compare text classification performance for:
- KNN
- SVM
- Naive Bayes

using `compare_classifiers_in_directory("IR_30dataset")` where labels are read from
the filename suffix after the last `_` (for example, `356_9.txt` -> label `9`).

## Usage

```python
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
```
