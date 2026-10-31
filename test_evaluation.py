import json
import tempfile
import unittest
from pathlib import Path

from evaluate_photobench import evaluate
from make_submission import convert


class EvaluationToolsTest(unittest.TestCase):
    def test_submission_formats_and_photobench_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cirr_pool = [f"image{i}.png" for i in range(50)]
            cirr_data = root / "cirr.json"
            cirr_run = root / "cirr.jsonl"
            cirr_data.write_text(json.dumps({"cases": [{"query_id": "q1", "retrieved_top50": cirr_pool}]}))
            cirr_run.write_text(json.dumps({"query_id": "q1", "final_ranking": list(reversed(cirr_pool))}) + "\n")
            submission = convert("cirr", str(cirr_data), str(cirr_run), 50)
            self.assertEqual(submission["q1"][0], "image49")
            self.assertEqual(submission["version"], "rc2")

            circo_pool = [f"{i}.jpg" for i in range(50)]
            circo_data = root / "circo.json"
            circo_run = root / "circo.jsonl"
            circo_data.write_text(json.dumps({"cases": [{"query_id": 7, "retrieved_top100": circo_pool}]}))
            circo_run.write_text(json.dumps({"query_id": 7, "final_ranking": list(reversed(circo_pool))}) + "\n")
            self.assertEqual(convert("circo", str(circo_data), str(circo_run), 50)["7"][0], 49)

            photo_data = root / "photo.json"
            photo_run = root / "photo.jsonl"
            photo_data.write_text(json.dumps({"cases": [
                {"query_id": "a", "retrieved_top50": cirr_pool, "target": ["image49.png"]},
                {"query_id": "b", "retrieved_top50": cirr_pool, "target": []},
            ]}))
            photo_run.write_text("\n".join(json.dumps({"query_id": qid, "final_ranking": list(reversed(cirr_pool))})
                                               for qid in ("a", "b")) + "\n")
            result = evaluate([str(photo_data)], [str(photo_run)])
            self.assertEqual(result["queries"], 2)
            self.assertEqual(result["zero_ground_truth_queries"], 1)
            self.assertEqual(result["metrics_percent"]["recall@1"], 50.0)
            self.assertEqual(result["metrics_percent"]["ndcg@1"], 50.0)


if __name__ == "__main__":
    unittest.main()
