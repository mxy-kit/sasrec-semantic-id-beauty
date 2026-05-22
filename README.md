#  SASRecCE with Ordinary Item IDs and SASRec-style Model with Semantic IDs on Amazon Beauty 2014

This repository contains the ordinary item-ID SASRec baseline used for next-item prediction on Amazon Beauty 2014, and also a TIGER-style generative recommendation model under the same global temporal split protocol as the ordinary SASRec baseline.

## Compared Methods

### 1. SASRecCE with Ordinary Item IDs - `sasrec-ce-beauty` 

### 2. SASRec-style Model with Semantic IDs - `semantic-ar-beauty` 

## 1.The setup follows:

- global temporal split 90/5/5;
- no-cold filtering:
  - validation targets must appear in train;
  - test targets must appear in train + validation;
- one holdout item per evaluated user;
- full catalog ranking;
- metrics with 95% bootstrap confidence intervals.
  

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




## 2.The implementation adapts the TIGER-style idea to the required experimental protocol:

- global temporal split `90/5/5`;
- no-cold filtering;
- our own item mapping;
- behavior-based semantic ID construction;
- validation-based checkpoint selection;
- final evaluation on the test split.

---

## Main idea

The original TIGER approach uses semantic IDs and a generative retrieval model.

In this repository, we adapt the TIGER-style architecture to the required global temporal split and to our own item mapping.

The semantic IDs are generated from behavior-based item embeddings:

```text
train sequences
→ item-item co-occurrence matrix
→ TruncatedSVD behavior item embeddings
→ Residual Quantization KMeans
→ 4-token semantic IDs
```

Each item is represented by a 4-token semantic ID:

```text
semantic_id(item) = [code_1, code_2, code_3, code_4]
```

The model then learns to generate the semantic ID of the next item autoregressively.

---

## Difference from the original TIGER repository

The original TIGER repository uses its own preprocessing pipeline, leave-one-out split, and item mapping.

This repository differs from the original TIGER setup in several important ways.

### 1. Global temporal split instead of leave-one-out

The original TIGER code uses a leave-one-out split:

```text
train = sequence[:-2]
validation target = sequence[-2]
test target = sequence[-1]
```

In this work, we use the required global temporal split:

```text
train / validation / test = 90% / 5% / 5%
```

### 2. No-cold filtering

Validation items that are unseen in train are removed.

Test items that are unseen in train + validation are also removed.

This ensures that validation and test targets can be mapped to known items.

### 3. Our own item mapping

The original TIGER `index_rqvae.json` cannot be directly used with our `beauty_global.pkl`, because the item mappings are different.

Therefore, semantic IDs are generated directly under our item mapping.

### 4. Behavior RQ-KMeans semantic IDs

Instead of using the original TIGER semantic ID file, this implementation builds semantic IDs from item behavior embeddings.

The pipeline is:

```text
train sequences
→ item-item co-occurrence matrix
→ TruncatedSVD item embeddings
→ Residual Quantization KMeans
→ collision repair
→ semantic ID index
```

### 5. Modified trainer

The trainer was modified to:

- respect `train_epochs_num` / `train_steps_num`;
- save the best checkpoint immediately when validation `NDCG@10` improves;
- save the last checkpoint at the end of training;
- allow disabling test evaluation during tuning.

This is important because the final test set should not be used during validation-based model selection.

---

## Repository structure

```text
semantic-ar-beauty/
├── configs/
│   └── tiger_global_behavior_rqkmeans_better_final250.json
├── modeling/
│   ├── dataloader/
│   ├── dataset/
│   │   ├── base.py
│   │   ├── samplers.py
│   │   └── global_split.py
│   ├── loss/
│   ├── metric/
│   ├── models/
│   ├── trainer/
│   │   ├── callbacks.py
│   │   └── trainer.py
│   └── utils/
├── scripts/
│   ├── make_global_nocold_split.py
│   └── make_behavior_rqkmeans_index.py
├── train_sasrec.py
├── train_tiger.py
├── train_tiger_global.py
└── README.md
```

---

## File descriptions

### `train_tiger_global.py`

Main training script for the modified TIGER-style model.

It loads:

- the global split dataset;
- the behavior RQ-KMeans semantic ID index;
- the TIGER encoder-decoder model;
- the semantic Recall / NDCG metrics.

It is used for:

- validation-based model selection;
- final train+validation training;
- test evaluation.

---

### `modeling/dataset/global_split.py`

This file implements the dataset loader for the required global temporal split.

It reads a SASRec-style pickle file:

```python
train, valid, test, usernum, itemnum
```

and converts it into TIGER-style training / validation / test samples.

Important details:

- SASRec item IDs are 1-based;
- TIGER internal item IDs are 0-based;
- each validation/test sample contains exactly one holdout target item.

---

### `modeling/trainer/trainer.py`

Modified trainer.

Compared with the original TIGER trainer, this version:

- supports explicit stopping by `train_epochs_num` or `train_steps_num`;
- saves the best checkpoint when validation `NDCG@10` improves;
- saves the last checkpoint after training;
- allows disabling test evaluation during tuning.

This is important because the final test set should not be used during validation-based model selection.

---

### `scripts/make_global_nocold_split.py`

Creates the no-cold split files from `beauty_global.pkl`.

It produces:

```text
beauty_global_nocold.pkl
beauty_global_nocold_trainval.pkl
```

The first file is used for tuning:

```text
train → validation
```

The second file is used for final training:

```text
train + validation → test
```

---

### `scripts/make_behavior_rqkmeans_index.py`

Generates behavior-based semantic IDs.

The script performs:

```text
train sequences
→ item-item co-occurrence matrix
→ TruncatedSVD item embeddings
→ RQ-KMeans semantic IDs
→ collision repair
```

It outputs:

```text
behavior_svd128_embeddings.npy
beauty_item_codes_behavior_rqkmeans_better_ours.npy
index_behavior_rqkmeans_better_ours.json
```

The final JSON file is used by TIGER as the semantic ID index.

---

### `configs/tiger_global_behavior_rqkmeans_better_final250.json`

Training configuration for the final TIGER-style generative model.

Main settings:

| Parameter | Value |
|---|---:|
| `num_codebooks` | 4 |
| `codebook_size` | 256 |
| `embedding_dim` | 128 |
| `num_encoder_layers` | 4 |
| `num_decoder_layers` | 4 |
| `num_heads` | 6 |
| `num_beams` | 20 |
| `top_k` | 20 |
| `train_epochs_num` | 250 |
| `valid_step` | 10000 |
| `best_metric` | `ndcg@10` |

---

## Running the experiment

### 1. Prepare the no-cold global split

Put the input file here:

```text
data/Beauty_global/beauty_global.pkl
```

Then run:

```bash
python scripts/make_global_nocold_split.py \
  --input_pkl data/Beauty_global/beauty_global.pkl \
  --output_dir data/Beauty_global
```

This creates:

```text
data/Beauty_global/beauty_global_nocold.pkl
data/Beauty_global/beauty_global_nocold_trainval.pkl
```

---

### 2. Generate behavior RQ-KMeans semantic IDs

```bash
python scripts/make_behavior_rqkmeans_index.py \
  --split_pkl data/Beauty_global/beauty_global_nocold.pkl \
  --output_dir data/Beauty_global
```

This creates:

```text
data/Beauty_global/behavior_svd128_embeddings.npy
data/Beauty_global/beauty_item_codes_behavior_rqkmeans_better_ours.npy
data/Beauty_global/index_behavior_rqkmeans_better_ours.json
```

---

### 3. Train TIGER on train split and select by validation

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_tiger_global.py \
  --params configs/tiger_global_behavior_rqkmeans_better_final250.json
```


### 4. Final training and test evaluation

After selecting the best validation step, the final model is trained on:

```text
train + validation
```

and evaluated on:

```text
test
```

The final metrics are computed for:

- `Recall@5`, `NDCG@5`;
- `Recall@10`, `NDCG@10`;


---



## Notes


The original TIGER implementation uses its own item mapping and leave-one-out split. This repository adapts the TIGER-style model to the required global temporal split protocol.

## Results from our run

SASRecCE ordinary item IDs .

| Metric | Value | 95% CI |
|---|---:|---:|
| Recall@5 | 0.0316 | [0.0251, 0.0381] |
| NDCG@5 | 0.0193 | [0.0151, 0.0233] |
| Recall@10 | 0.0527 | [0.0442, 0.0616] |
| NDCG@10 | 0.0262 | [0.0217, 0.0306] |


TIGER-like autoregressive prediction with semantic IDs.

| Metric | Value | 95% CI |
|---|---:|---:|
| Recall@5 | 0.0328 | [0.0263, 0.0393] |
| NDCG@5 | 0.0198 | [0.0156, 0.0238] |
| Recall@10 | 0.0556 | [0.0471, 0.0645] |
| NDCG@10 | 0.0270 | [0.0225, 0.0314] |


