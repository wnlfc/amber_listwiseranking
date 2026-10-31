"""Convert AMBER JSONL into the CIRR or CIRCO submission format."""

import argparse
import json
from pathlib import Path


def load_inputs(path):
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    by_id = {}
    for case in cases:
        qid = str(case["query_id"])
        if qid in by_id:
            raise ValueError(f"duplicate retrieval query_id: {qid}")
        pool = case.get("retrieved_top50") or case.get("retrieved_top100")
        if not isinstance(pool, list) or len(pool) < 50:
            raise ValueError(f"query {qid} has fewer than 50 candidates")
        by_id[qid] = pool[:50]
    return by_id


def load_rerank(path):
    by_id = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record["query_id"])
            if qid in by_id:
                raise ValueError(f"duplicate rerank query_id: {qid}")
            if record.get("error"):
                raise ValueError(f"query {qid} failed: {record['error']}")
            by_id[qid] = record["final_ranking"]
    return by_id


def merge(pool, ranking, top_k):
    if len(pool) != len(set(pool)):
        raise ValueError("retrieval pool contains duplicate image names")
    if len(ranking) != len(pool) or set(ranking) != set(pool):
        raise ValueError("rerank result is not a permutation of the retrieval pool")
    ordered = ranking[:top_k] + [image for image in pool if image not in ranking[:top_k]]
    if len(ordered) != 50 or len(set(ordered)) != 50:
        raise ValueError("merged result is not a unique top-50 list")
    return ordered


def convert(dataset, retrieval, rerank, top_k):
    original = load_inputs(retrieval)
    ranked = load_rerank(rerank)
    if set(original) != set(ranked):
        missing = len(set(original) - set(ranked))
        extra = len(set(ranked) - set(original))
        raise ValueError(f"query IDs differ: {missing} missing, {extra} unexpected")
    if dataset == "cirr":
        submission = {"version": "rc2", "metric": "recall"}
        for qid, pool in original.items():
            submission[qid] = [Path(name).stem for name in merge(pool, ranked[qid], top_k)]
    else:
        submission = {}
        for qid, pool in original.items():
            submission[qid] = [int(Path(name).stem) for name in merge(pool, ranked[qid], top_k)]
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cirr", "circo"))
    parser.add_argument("--retrieval", required=True, help="original retrieval JSON")
    parser.add_argument("--rerank", required=True, help="AMBER output JSONL")
    parser.add_argument("--output", required=True, help="submission JSON")
    parser.add_argument("--top_k", type=int, default=50,
                        help="number of reranked candidates to keep before appending retrieval order")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50:
        parser.error("--top_k must be in 1..50")
    submission = convert(args.dataset, args.retrieval, args.rerank, args.top_k)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(submission, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(submission) - (2 if args.dataset == 'cirr' else 0)} queries to {destination}")


if __name__ == "__main__":
    main()
