import unittest

from doc_ranker import compare_classifiers, load_labeled_dataset, rank_datasets
from pathlib import Path
from tempfile import TemporaryDirectory

class RankDatasetsTests(unittest.TestCase):
    def test_ranks_most_relevant_dataset_first(self):
        data_dir = Path("IR_30dataset")
        datasets = []
        for file_path in sorted(data_dir.glob("*.txt")):
            datasets.append(file_path.read_text(encoding="utf-8").strip())

        results = rank_datasets("hitchcock", datasets)

        self.assertEqual(len(results), len(datasets))
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertIn("dataset", results[0])

    def test_top_k_limits_results(self):
        datasets = ["apple", "banana", "apple banana"]

        results = rank_datasets("apple", datasets, top_k=2)

        self.assertEqual(len(results), 2)

    def test_empty_query_returns_zero_scores(self):
        datasets = ["a", "b"]

        results = rank_datasets("", datasets)

        self.assertEqual(results[0]["score"], 0.0)
        self.assertEqual(results[1]["score"], 0.0)


class ClassifierComparisonTests(unittest.TestCase):
    def test_compare_classifiers_returns_metrics_for_all_models(self):
        texts = [
            "apple red fruit",
            "banana yellow fruit",
            "grape sweet fruit",
            "orange citrus fruit",
            "car road engine",
            "truck diesel road",
            "bus city vehicle",
            "train rail vehicle",
        ]
        labels = ["fruit", "fruit", "fruit", "fruit", "vehicle", "vehicle", "vehicle", "vehicle"]

        results = compare_classifiers(texts, labels, test_size=0.25, random_state=1)

        self.assertEqual(set(results.keys()), {"knn", "svm", "naive_bayes"})
        for model_name in results:
            self.assertIn("accuracy", results[model_name])
            self.assertIn("report", results[model_name])
            self.assertIn("predictions", results[model_name])
            self.assertGreaterEqual(results[model_name]["accuracy"], 0.0)
            self.assertLessEqual(results[model_name]["accuracy"], 1.0)

    def test_load_labeled_dataset_extracts_label_from_filename_suffix(self):
        with TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "doc_a_class1.txt"
            second = Path(temp_dir) / "doc_b_class2.txt"
            first.write_text("alpha", encoding="utf-8")
            second.write_text("beta", encoding="utf-8")

            texts, labels = load_labeled_dataset(temp_dir)

            self.assertEqual(texts, ["alpha", "beta"])
            self.assertEqual(labels, ["class1", "class2"])


if __name__ == "__main__":
    unittest.main()
