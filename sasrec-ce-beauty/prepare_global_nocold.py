import argparse
import copy
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="beauty_global")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_pkl, "rb") as f:
        train, valid, test, usernum, itemnum = pickle.load(f)

    print("Original split")
    print("users:", usernum)
    print("items:", itemnum)
    print("valid users:", sum(len(valid[u]) > 0 for u in range(1, usernum + 1)))
    print("test users:", sum(len(test[u]) > 0 for u in range(1, usernum + 1)))

    train_items = set()
    for u in range(1, usernum + 1):
        train_items.update(train[u])

    valid_nocold = copy.deepcopy(valid)
    removed_valid = 0

    for u in range(1, usernum + 1):
        if len(valid_nocold[u]) > 0:
            item = valid_nocold[u][0]
            if item not in train_items:
                valid_nocold[u] = []
                removed_valid += 1

    train_valid_items = set(train_items)
    for u in range(1, usernum + 1):
        train_valid_items.update(valid_nocold[u])

    test_nocold = copy.deepcopy(test)
    removed_test = 0

    for u in range(1, usernum + 1):
        if len(test_nocold[u]) > 0:
            item = test_nocold[u][0]
            if item not in train_valid_items:
                test_nocold[u] = []
                removed_test += 1

    print("\nAfter no-cold filtering")
    print("removed valid cold users:", removed_valid)
    print("removed test cold users:", removed_test)
    print("valid users after:", sum(len(valid_nocold[u]) > 0 for u in range(1, usernum + 1)))
    print("test users after:", sum(len(test_nocold[u]) > 0 for u in range(1, usernum + 1)))
    print("max valid len:", max(len(valid_nocold[u]) for u in range(1, usernum + 1)))
    print("max test len:", max(len(test_nocold[u]) for u in range(1, usernum + 1)))

    out_nocold = os.path.join(args.output_dir, f"{args.prefix}_nocold.pkl")
    with open(out_nocold, "wb") as f:
        pickle.dump([train, valid_nocold, test_nocold, usernum, itemnum], f)

    trainval = copy.deepcopy(train)
    valid_empty = {u: [] for u in range(1, usernum + 1)}

    for u in range(1, usernum + 1):
        if len(valid_nocold[u]) > 0:
            trainval[u] = trainval[u] + valid_nocold[u]

    out_trainval = os.path.join(args.output_dir, f"{args.prefix}_nocold_trainval.pkl")
    with open(out_trainval, "wb") as f:
        pickle.dump([trainval, valid_empty, test_nocold, usernum, itemnum], f)

    print("\nSaved files")
    print("saved:", out_nocold)
    print("saved:", out_trainval)


if __name__ == "__main__":
    main()
