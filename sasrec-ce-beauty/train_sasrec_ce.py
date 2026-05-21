import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn

from model_sasrec_ce import SASRecCE


def sample_batch(train, usernum, maxlen, batch_size):
    seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
    labels = np.zeros((batch_size, maxlen), dtype=np.int64)

    users = []
    while len(users) < batch_size:
        u = random.randint(1, usernum)
        if len(train[u]) >= 2:
            users.append(u)

    for i, u in enumerate(users):
        items = train[u]
        idx = maxlen - 1
        nxt = items[-1]

        for item in reversed(items[:-1]):
            seqs[i, idx] = item
            labels[i, idx] = nxt
            nxt = item
            idx -= 1

            if idx < 0:
                break

    return seqs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_dir", required=True)

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=50, type=int)
    parser.add_argument("--hidden_units", default=100, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_every", default=10, type=int)
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_path = f"data/{args.dataset}.pkl"

    with open(data_path, "rb") as f:
        train, valid, test, usernum, itemnum = pickle.load(f)

    print("usernum:", usernum)
    print("itemnum:", itemnum)
    seq_lens = [len(train[u]) for u in range(1, usernum + 1) if len(train[u]) > 0]
    print("average sequence length:", np.mean(seq_lens))

    out_dir = f"{args.dataset}_{args.train_dir}"
    os.makedirs(out_dir, exist_ok=True)

    model = SASRecCE(usernum, itemnum, args).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.l2_emb,
    )
    criterion = nn.CrossEntropyLoss()

    trainable_users = [u for u in range(1, usernum + 1) if len(train[u]) >= 2]
    steps_per_epoch = max(1, sum(max(0, len(train[u]) - 1) for u in trainable_users) // args.batch_size)

    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        losses = []

        for _ in range(steps_per_epoch):
            seq_np, labels_np = sample_batch(
                train=train,
                usernum=usernum,
                maxlen=args.maxlen,
                batch_size=args.batch_size,
            )

            seq = torch.LongTensor(seq_np).to(args.device)
            labels = torch.LongTensor(labels_np).to(args.device)

            logits = model(seq)

            mask = labels > 0
            if mask.sum() == 0:
                continue

            loss = criterion(logits[mask], labels[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"mean CE loss in epoch {epoch}: {mean_loss:.6f}")

        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            ckpt_name = (
                f"SASRecCE.epoch={epoch}.lr={args.lr}."
                f"layer={args.num_blocks}.head={args.num_heads}."
                f"hidden={args.hidden_units}.maxlen={args.maxlen}.pth"
            )
            ckpt_path = os.path.join(out_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print("saved checkpoint:", ckpt_name)

    print("Done")
    print("elapsed seconds:", round(time.time() - start_time, 2))


if __name__ == "__main__":
    main()