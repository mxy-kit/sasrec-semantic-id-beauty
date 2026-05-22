import argparse
import json
import pickle

import numpy as np


def bootstrap_ci(values, n_bootstrap=1000, seed=42, alpha=0.05):
    values = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)

    means = []
    n = len(values)

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        means.append(values[sample_idx].mean())

    low = np.percentile(means, 100 * alpha / 2)
    high = np.percentile(means, 100 * (1 - alpha / 2))

    return float(values.mean()), float(low), float(high)


def load_metric_values(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        raise ValueError("metric file must be .pkl, .json, or .jsonl")

    if isinstance(data, dict):
        return data

    metric_values = {}

    for row in data:
        for key, value in row.items():
            if key not in metric_values:
                metric_values[key] = []
            metric_values[key].append(value)

    return metric_values


def find_metric_key(metric_values, candidates):
    lower_to_key = {key.lower(): key for key in metric_values.keys()}

    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return key

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_path", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_bootstrap", default=1000, type=int)
    args = parser.parse_args()

    metric_values = load_metric_values(args.metric_path)

    metrics = [
        ("Recall@5", ["Recall@5", "recall@5"]),
        ("NDCG@5", ["NDCG@5", "ndcg@5"]),
        ("Recall@10", ["Recall@10", "recall@10"]),
        ("NDCG@10", ["NDCG@10", "ndcg@10"]),
        ("Recall@100", ["Recall@100", "recall@100"]),
        ("NDCG@100", ["NDCG@100", "ndcg@100"]),
    ]

    print("Final metrics with 95% bootstrap CI")
    print("=" * 50)

    used_key = None

    for display_name, candidates in metrics:
        key = find_metric_key(metric_values, candidates)

        if key is None:
            continue

        values = metric_values[key]
        mean, low, high = bootstrap_ci(
            values,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=0.05,
        )

        print(f"{display_name} = {mean:.4f} [{low:.4f}, {high:.4f}]")
        used_key = key

    if used_key is not None:
        print()
        print("users:", len(metric_values[used_key]))
    else:
        print()
        print("No matching metrics found.")
        print("Available keys:", list(metric_values.keys()))


if __name__ == "__main__":
    main()