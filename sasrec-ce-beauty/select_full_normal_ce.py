import argparse
import glob
import os
import pickle
import re

import numpy as np
import torch

from model_sasrec_ce import SASRecCE


def get_epoch(path):
    m = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def metrics_from_ranks(ranks, ks=(5, 10, 20, 50, 100)):
    res = {}
    ranks = np.array(ranks, dtype=np.int64)

    for k in ks:
        hits = (ranks < k).astype(np.float64)
        ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
        mrr = np.where(ranks < k, 1.0 / (ranks + 1), 0.0)

        res[f"Recall@{k}"] = float(hits.mean())
        res[f"NDCG@{k}"] = float(ndcg.mean())
        res[f"MRR@{k}"] = float(mrr.mean())

    res["users"] = len(ranks)
    return res


@torch.no_grad()
def evaluate_full_normal(model, dataset, args, split="valid", ks=(5, 10, 20, 50, 100)):
    train, valid, test, usernum, itemnum = dataset

    ranks = []

    all_items = np.arange(1, itemnum + 1)

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

        seq = np.zeros(args.maxlen, dtype=np.int64)
        hist = history[-args.maxlen:]
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

    return metrics_from_ranks(ranks, ks=ks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--split", default="valid", choices=["valid", "test"])

    parser.add_argument("--maxlen", default=50, type=int)
    parser.add_argument("--hidden_units", default=100, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm_first", action="store_true")
    args = parser.parse_args()

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    train, valid, test, usernum, itemnum = dataset

    ckpts = sorted(
        glob.glob(os.path.join(args.checkpoint_dir, "*.pth")),
        key=get_epoch,
    )

    print("checkpoints:", len(ckpts))

    summary = []
    best = None

    for ckpt in ckpts:
        epoch = get_epoch(ckpt)
        print()
        print("Evaluating epoch", epoch)

        model = SASRecCE(usernum, itemnum, args).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device(args.device)))
        model.eval()

        res = evaluate_full_normal(
            model=model,
            dataset=dataset,
            args=args,
            split=args.split,
            ks=(5, 10, 20, 50, 100),
        )
        res["epoch"] = epoch
        summary.append(res)

        print(
            f"epoch={epoch:3d} | "
            f"NDCG@5={res['NDCG@5']:.5f} | Recall@5={res['Recall@5']:.5f} | "
            f"NDCG@10={res['NDCG@10']:.5f} | Recall@10={res['Recall@10']:.5f} | "
            f"NDCG@100={res['NDCG@100']:.5f} | Recall@100={res['Recall@100']:.5f}"
        )

        if best is None or res["NDCG@10"] > best["NDCG@10"]:
            best = res

    print()
    print("Summary:")
    for res in summary:
        print(
            f"epoch={res['epoch']:3d} | "
            f"NDCG@5={res['NDCG@5']:.5f} | Recall@5={res['Recall@5']:.5f} | "
            f"NDCG@10={res['NDCG@10']:.5f} | Recall@10={res['Recall@10']:.5f} | "
            f"NDCG@100={res['NDCG@100']:.5f} | Recall@100={res['Recall@100']:.5f}"
        )

    print()
    print("Best by NDCG@10:")
    print(best)


if __name__ == "__main__":
    main()