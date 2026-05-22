import pickle

from modeling.dataset.samplers import TrainSampler, EvalSampler
from modeling.dataset.base import Dataset


def _convert_seq_to_zero_based(seq):
    # SASRec.pytorch item ids are 1..itemnum.
    # TIGER repo item ids are 0..itemnum-1.
    return [int(x) - 1 for x in seq if int(x) > 0]


def create_dataset_from_sasrec_pkl(
    pkl_path,
    max_sequence_length,
    sampler_type,
    is_extended=True,
    mode="tune"
):
    with open(pkl_path, "rb") as f:
        train, valid, test, usernum, itemnum = pickle.load(f)

    train_dataset = []
    validation_dataset = []
    test_dataset = []

    for u in range(1, usernum + 1):
        user_id = int(u) - 1

        train_seq = _convert_seq_to_zero_based(train[u])
        valid_seq = _convert_seq_to_zero_based(valid[u])
        test_seq = _convert_seq_to_zero_based(test[u])

        # Train sample format: history + target.
        # TrainSampler uses the last item as the target.
        if len(train_seq) >= 2:
            if is_extended:
                for prefix_len in range(2, len(train_seq) + 1):
                    train_dataset.append({
                        "user.ids": [user_id],
                        "item.ids": train_seq[:prefix_len],
                    })
            else:
                train_dataset.append({
                    "user.ids": [user_id],
                    "item.ids": train_seq,
                })

        # Validation sample: train history + one validation holdout target.
        if len(valid_seq) == 1 and len(train_seq) >= 1:
            validation_dataset.append({
                "user.ids": [user_id],
                "item.ids": train_seq + valid_seq,
            })

        # Test sample:
        # tune pkl: train + valid + test
        # trainval pkl: train already equals train + valid, valid is empty
        if len(test_seq) == 1 and len(train_seq) >= 1:
            test_dataset.append({
                "user.ids": [user_id],
                "item.ids": train_seq + valid_seq + test_seq,
            })

    print("Created global split dataset from:", pkl_path)
    print("mode:", mode)
    print("train samples:", len(train_dataset))
    print("validation samples:", len(validation_dataset))
    print("test samples:", len(test_dataset))
    print("itemnum:", itemnum)

    train_sampler = TrainSampler(
        train_dataset,
        sampler_type,
        max_sequence_length=max_sequence_length
    )
    validation_sampler = EvalSampler(
        validation_dataset,
        max_sequence_length=max_sequence_length
    )
    test_sampler = EvalSampler(
        test_dataset,
        max_sequence_length=max_sequence_length
    )

    return Dataset(
        train_sampler=train_sampler,
        validation_sampler=validation_sampler,
        test_sampler=test_sampler,
        num_items=itemnum,
        max_sequence_length=max_sequence_length
    )
