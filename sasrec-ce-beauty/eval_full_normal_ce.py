import argparse
import pickle

import numpy as np
import torch

from model_sasrec_ce import SASRecCE


def metrics_from_ranks(ranks, ks=(5, 10, 20, 50, 100)):
    ranks = np.array(ranks, dtype=np.int64)
    res = {}

    for k in ks:
        hits = (ranks < k).astype(np.float64)
        ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
        mrr = np.where(ranks < k, 1.0 / (ranks + 1), 0.0)

        res[f"Recall@{k}"] = hits
        res[f"NDCG@{k}"] = ndcg
        res[f"MRR@{k}"] = mrr

    return res


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


@torch.no_grad()
def collect_ranks(model, dataset, args, split="test"):
    train, valid, test, usernum, itemnum = dataset

    ranks = []

    for u in range(1, usernum + 1):
        if split == "valid":
            if len(train[u]) < 1 or len(valid[u]) != 1:
                continue

            history = list(train[u])
            target = valid[u][0]

        elif split == "test":
            if len(train[u]) < 1 or len(test[u]) != 1:
                continue

            history = list(train[u])

            if len(valid[u]) > 0:
                history += list(valid[u])

            target = test[u][0]

        else:
            raise ValueError("split must be valid or test")

        if target < 1 or target > itemnum:
            continue

        seq = np.zeros(args.maxlen, dtype=np.int64)
        hist = history[-args.maxlen:]

        if len(hist) == 0:
            continue

        seq[-len(hist):] = hist

        seq_t = torch.LongTensor(seq).unsqueeze(0).to(args.device)

        logits = model.predict(seq_t).detach().cpu().numpy().reshape(-1)

        scores = logits[1:itemnum + 1].copy()

        seen = set(history)
        seen.discard(target)

        for item in seen:
            if 1 <= item <= itemnum:
                scores[item - 1] = -np.inf

        target_score = scores[target - 1]
        rank = int((scores > target_score).sum())

        ranks.append(rank)

    return np.array(ranks, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--state_dict_path", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])

    parser.add_argument("--maxlen", default=50, type=int)
    parser.add_argument("--hidden_units", default=100, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm_first", action="store_true")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_bootstrap", default=1000, type=int)

    args = parser.parse_args()

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    train, valid, test, usernum, itemnum = dataset

    print("Dataset:")
    print("users:", usernum)
    print("items:", itemnum)
    print("split:", args.split)
    print("dataset_path:", args.dataset_path)
    print("state_dict_path:", args.state_dict_path)

    model = SASRecCE(usernum, itemnum, args).to(args.device)

    state = torch.load(args.state_dict_path, map_location=torch.device(args.device))
    model.load_state_dict(state)
    model.eval()

    ranks = collect_ranks(
        model=model,
        dataset=dataset,
        args=args,
        split=args.split,
    )

    print()
    print("Evaluated users:", len(ranks))

    if len(ranks) == 0:
        raise RuntimeError("No users were evaluated. Check split and dataset_path.")

    metric_values = metrics_from_ranks(
        ranks,
        ks=(5, 10, 20, 50, 100),
    )

    print()
    print("Final metrics with 95% bootstrap CI")
    print("=" * 50)

    for name, values in metric_values.items():
        mean, low, high = bootstrap_ci(
            values,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=0.05,
        )

        print(f"{name} = {mean:.4f} [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
