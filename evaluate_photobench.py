"""Evaluate AMBER PhotoBench outputs against locally supplied retrieval JSONs."""

import argparse
import json
import math


K_VALUES = (1, 5, 10, 20, 50)


def ground_truth(case):
    value = case.get("target") or case.get("ground_truth") or []
    if isinstance(value, str):
        value = [value]
    return set(value)


def scores(ranking, relevant):
    result = {}
    for k in K_VALUES:
        top = ranking[:k]
        hits = [1 if name in relevant else 0 for name in top]
        result[f"recall@{k}"] = sum(hits) / len(relevant) if relevant else 0.0
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        result[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return result


def evaluate(retrieval_files, rerank_files):
    totals = {name: 0.0 for name in scores([], set())}
    count = 0
    empty_gt = 0
    for retrieval_file, rerank_file in zip(retrieval_files, rerank_files):
        with open(retrieval_file, encoding="utf-8") as handle:
            cases = json.load(handle)["cases"]
        by_id = {str(case["query_id"]): case for case in cases}
        if len(by_id) != len(cases):
            raise ValueError(f"duplicate query IDs in {retrieval_file}")
        seen = set()
        with open(rerank_file, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                qid = str(record["query_id"])
                if qid in seen or qid not in by_id:
                    raise ValueError(f"duplicate or unknown query ID: {qid}")
                seen.add(qid)
                if record.get("error"):
                    raise ValueError(f"query {qid} failed: {record['error']}")
                case = by_id[qid]
                pool = (case.get("retrieved_top50") or case.get("retrieved_top100") or [])[:50]
                ranking = record["final_ranking"]
                if (len(pool) != len(set(pool)) or len(ranking) != len(pool)
                        or set(ranking) != set(pool)):
                    raise ValueError(f"query {qid} ranking does not match its retrieval pool")
                gt = ground_truth(case)
                empty_gt += not gt
                for name, value in scores(ranking, gt).items():
                    totals[name] += value
                count += 1
        if seen != set(by_id):
            raise ValueError(f"missing {len(set(by_id) - seen)} queries in {rerank_file}")
    if count == 0:
        raise ValueError("no queries to evaluate")
    return {
        "queries": count,
        "zero_ground_truth_queries": empty_gt,
        "metrics_percent": {name: 100 * value / count for name, value in totals.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True, nargs="+", help="PhotoBench retrieval JSON files")
    parser.add_argument("--rerank", required=True, nargs="+", help="matching AMBER JSONL files")
    args = parser.parse_args()
    if len(args.retrieval) != len(args.rerank):
        parser.error("provide one --rerank file for every --retrieval file, in the same order")
    print(json.dumps(evaluate(args.retrieval, args.rerank), indent=2))


if __name__ == "__main__":
    main()
