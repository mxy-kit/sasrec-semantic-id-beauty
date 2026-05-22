#  SASRecCE with Ordinary Item IDs and SASRec-style Model with Semantic IDs on Amazon Beauty 2014

This repository contains the ordinary item-ID SASRec baseline used for next-item prediction on Amazon Beauty 2014, and also a TIGER-style generative recommendation model under the same global temporal split protocol as the ordinary SASRec baseline.

The setup follows:

- global temporal split 90/5/5;
- no-cold filtering:
  - validation targets must appear in train;
  - test targets must appear in train + validation;
- one holdout item per evaluated user;
- full catalog ranking;
- metrics with 95% bootstrap confidence intervals.
  
## Compared Methods

### 1. SASRecCE with Ordinary Item IDs - `sasrec-ce-beauty` 

### 2. SASRec-style Model with Semantic IDs - `semantic-ar-beauty` 

## Files of `sasrec-ce-beauty` 

- `prepare_global_nocold.py`  
  Creates no-cold validation/test splits and the train+validation final split.

- `model_sasrec_ce.py`  
  SASRec model trained with full softmax cross-entropy over the item catalog.

- `train_sasrec_ce.py`  
  Trains SASRecCE and saves checkpoints.

- `select_full_normal_ce.py`  
  Selects the best checkpoint on validation by NDCG@10.

- `eval_full_normal_ce.py`  
  Evaluates a final checkpoint on test using full catalog ranking and bootstrap confidence intervals.

## Prepare data

The input file should be a pickle file with the following format:

```python
[user_train, user_valid, user_test, usernum, itemnum]
```

Run:

```bash
python prepare_global_nocold.py \
  --input_path data/beauty_global.pkl \
  --output_dir data \
  --prefix beauty_global
```

This creates:

```text
data/beauty_global_nocold.pkl
data/beauty_global_nocold_trainval.pkl
```

## Tune SASRecCE on train / validation

```bash
python train_sasrec_ce.py \
  --dataset=beauty_global_nocold \
  --train_dir=normal_ce_h100_h2_nocold_tune \
  --maxlen=50 \
  --hidden_units=100 \
  --num_blocks=2 \
  --num_epochs=100 \
  --num_heads=2 \
  --dropout_rate=0.2 \
  --batch_size=128 \
  --lr=0.001 \
  --device=cuda \
  --save_every=10
```

Select the best checkpoint by validation NDCG@10:

```bash
python select_full_normal_ce.py \
  --dataset_path=data/beauty_global_nocold.pkl \
  --checkpoint_dir=beauty_global_nocold_normal_ce_h100_h2_nocold_tune \
  --split=valid \
  --maxlen=50 \
  --hidden_units=100 \
  --num_blocks=2 \
  --num_heads=2 \
  --dropout_rate=0.2 \
  --device=cuda
```

## Final training on train + validation

If the best validation epoch is `10`, run:

```bash
python train_sasrec_ce.py \
  --dataset=beauty_global_nocold_trainval \
  --train_dir=normal_ce_h100_h2_nocold_final10 \
  --maxlen=50 \
  --hidden_units=100 \
  --num_blocks=2 \
  --num_epochs=10 \
  --num_heads=2 \
  --dropout_rate=0.2 \
  --batch_size=128 \
  --lr=0.001 \
  --device=cuda \
  --save_every=10
```

## Final test evaluation

```bash
python eval_full_normal_ce.py \
  --dataset_path=data/beauty_global_nocold_trainval.pkl \
  --state_dict_path=beauty_global_nocold_trainval_normal_ce_h100_h2_nocold_final10/SASRecCE.epoch=10.lr=0.001.layer=2.head=2.hidden=100.maxlen=50.pth \
  --maxlen=50 \
  --hidden_units=100 \
  --num_blocks=2 \
  --num_heads=2 \
  --dropout_rate=0.2 \
  --device=cuda \
  --k=10 \
  --seed=42
```

## Results from our run

SASRecCE ordinary item ID, `h100/h2`, final epoch `10`.

| Metric | Value | 95% CI |
|---|---:|---:|
| Recall@5 | 0.0316 | [0.0251, 0.0381] |
| NDCG@5 | 0.0193 | [0.0151, 0.0233] |
| Recall@10 | 0.0527 | [0.0442, 0.0616] |
| NDCG@10 | 0.0262 | [0.0217, 0.0306] |
| Recall@100 | 0.1727 | [0.1581, 0.1869] |
| NDCG@100 | 0.0499 | [0.0449, 0.0549] |
