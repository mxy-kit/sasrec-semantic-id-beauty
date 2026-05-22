import argparse
import copy
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_pkl, "rb") as f:
        train, valid, test, usernum, itemnum = pickle.load(f)

    # 1. Items observed in train
    train_items = set()
    for u in range(1, usernum + 1):
        train_items.update(train[u])

    # 2. Remove validation cold items with respect to train
    valid2 = copy.deepcopy(valid)
    removed_valid = 0

    for u in range(1, usernum + 1):
        if len(valid2[u]) > 0 and valid2[u][0] not in train_items:
            valid2[u] = []
            removed_valid += 1

    # 3. Items observed in train + filtered validation
    train_valid_items = set(train_items)
    for u in range(1, usernum + 1):
        train_valid_items.update(valid2[u])

    # 4. Remove test cold items with respect to train + validation
    test2 = copy.deepcopy(test)
    removed_test = 0

    for u in range(1, usernum + 1):
        if len(test2[u]) > 0 and test2[u][0] not in train_valid_items:
            test2[u] = []
            removed_test += 1

    print("users:", usernum)
    print("items:", itemnum)
    print("removed valid cold users:", removed_valid)
    print("removed test cold users:", removed_test)
    print("valid users after:", sum(len(valid2[u]) > 0 for u in range(1, usernum + 1)))
    print("test users after:", sum(len(test2[u]) > 0 for u in range(1, usernum + 1)))
    print("max valid len:", max(len(valid2[u]) for u in range(1, usernum + 1)))
    print("max test len:", max(len(test2[u]) for u in range(1, usernum + 1)))

    # Save train / validation / test split after no-cold filtering
    out_nocold = os.path.join(args.output_dir, "beauty_global_nocold.pkl")
    with open(out_nocold, "wb") as f:
        pickle.dump([train, valid2, test2, usernum, itemnum], f)

    # Build final train+validation split
    trainval = copy.deepcopy(train)
    valid_empty = {u: [] for u in range(1, usernum + 1)}

    for u in range(1, usernum + 1):
        if len(valid2[u]) > 0:
            trainval[u] = trainval[u] + valid2[u]

    out_trainval = os.path.join(args.output_dir, "beauty_global_nocold_trainval.pkl")
    with open(out_trainval, "wb") as f:
        pickle.dump([trainval, valid_empty, test2, usernum, itemnum], f)

    print("saved:", out_nocold)
    print("saved:", out_trainval)


if __name__ == "__main__":
    main()
