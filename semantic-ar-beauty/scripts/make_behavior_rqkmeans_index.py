import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def build_behavior_embeddings(train, usernum, itemnum, output_dir, window=3, dim=128):
    rows, cols, vals = [], [], []

    for u in range(1, usernum + 1):
        seq = train[u]
        n = len(seq)

        for i in range(n):
            a = seq[i]
            if a <= 0:
                continue

            left = max(0, i - window)
            right = min(n, i + window + 1)

            for j in range(left, right):
                if i == j:
                    continue

                b = seq[j]
                if b <= 0:
                    continue

                # SASRec item ids are 1-based, while TIGER index ids are 0-based.
                rows.append(a - 1)
                cols.append(b - 1)
                vals.append(1.0 / abs(i - j))

    X = coo_matrix((vals, (rows, cols)), shape=(itemnum, itemnum)).tocsr()
    print("cooc:", X.shape, "nnz:", X.nnz)

    svd = TruncatedSVD(n_components=dim, random_state=42)
    emb = svd.fit_transform(X)
    emb = normalize(emb).astype(np.float32)

    print("emb:", emb.shape)
    print("explained variance:", svd.explained_variance_ratio_.sum())

    emb_path = os.path.join(output_dir, "behavior_svd128_embeddings.npy")
    np.save(emb_path, emb)
    print("saved:", emb_path)

    return emb


def make_rqkmeans_codes(emb, itemnum, codebook_size=256, num_codebooks=4):
    codes = np.zeros((itemnum, num_codebooks), dtype=np.int64)
    residual = emb.copy()

    for q in range(num_codebooks):
        print("\nRQ level", q)

        km = MiniBatchKMeans(
            n_clusters=codebook_size,
            random_state=42 + q,
            batch_size=2048,
            n_init="auto",
            max_iter=300
        )

        labels = km.fit_predict(residual)
        centers = km.cluster_centers_

        codes[:, q] = labels
        residual = residual - centers[labels]

        cnt = Counter(labels.tolist())
        print("unique:", len(cnt))
        print("top 10:", cnt.most_common(10))

    codes4 = np.zeros((itemnum + 1, num_codebooks), dtype=np.int64)
    codes4[1:] = codes

    print("\nBefore repair:")
    print("max per col:", codes4.max(axis=0))
    print("unique per col:", [len(set(codes4[1:, i].tolist())) for i in range(num_codebooks)])
    print("unique full codes:", len(set(tuple(codes4[i]) for i in range(1, itemnum + 1))))

    # Collision repair:
    # Keep the original RQ-KMeans codes as much as possible.
    # Only duplicated full semantic IDs are repaired by changing the 4th token.
    used = set()
    repaired = 0

    for item_id in range(1, itemnum + 1):
        code = tuple(codes4[item_id])

        if code not in used:
            used.add(code)
            continue

        base = tuple(codes4[item_id, :3])
        old_c4 = int(codes4[item_id, 3])

        found = False
        for delta in range(1, codebook_size + 1):
            new_c4 = (old_c4 + delta) % codebook_size
            new_code = base + (new_c4,)

            if new_code not in used:
                codes4[item_id, 3] = new_c4
                used.add(new_code)
                repaired += 1
                found = True
                break

        if not found:
            print("WARNING: could not repair item", item_id, "base", base)

    print("\nAfter repair:")
    print("repaired:", repaired)
    print("max per col:", codes4.max(axis=0))
    print("unique per col:", [len(set(codes4[1:, i].tolist())) for i in range(num_codebooks)])
    print("unique full codes:", len(set(tuple(codes4[i]) for i in range(1, itemnum + 1))))

    for col in range(num_codebooks):
        cnt = Counter(codes4[1:, col].tolist())
        print(f"\ncol {col}")
        print("num unique:", len(cnt))
        print("top 10:", cnt.most_common(10))

    return codes4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_pkl", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split_pkl, "rb") as f:
        train, valid, test, usernum, itemnum = pickle.load(f)

    emb = build_behavior_embeddings(
        train=train,
        usernum=usernum,
        itemnum=itemnum,
        output_dir=args.output_dir
    )

    codes4 = make_rqkmeans_codes(
        emb=emb,
        itemnum=itemnum,
        codebook_size=256,
        num_codebooks=4
    )

    codes_path = os.path.join(args.output_dir, "beauty_item_codes_behavior_rqkmeans_better_ours.npy")
    np.save(codes_path, codes4)
    print("saved:", codes_path)

    index = {}
    for item_id in range(1, itemnum + 1):
        # TIGER item ids are 0-based.
        index[str(item_id - 1)] = [int(x) for x in codes4[item_id]]

    out_path = os.path.join(args.output_dir, "index_behavior_rqkmeans_better_ours.json")
    with open(out_path, "w") as f:
        json.dump(index, f)

    print("saved:", out_path)
    print("first 3:", list(index.items())[:3])


if __name__ == "__main__":
    main()
