# Enzyme Fitness Ranking

Predict enzyme fitness from protein sequence data using ESM2-based ranking models.

This repository experiments with ranking enzyme variants from limited experimental
fitness measurements. The data is stored as protein sequences with numeric fitness
values, and the models learn to order variants by expected fitness rather than
predicting an absolute value directly.

## Repository Layout

```text
.
├── binaryClassification/
│   ├── train.py        # Binary classification model
│   └── test.py         # Binary classification model
├── data/
│   ├── data.csv        # Full sequence/fitness dataset
│   ├── split_data.py   # Stratified train/val/test splitter
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── listMLE/
│   ├── listsize/
│   │   ├── train.py    # Ranking model with different listsizes 
│   │   └── test.py     # Ranking model with different listsizes 
│   ├── lora/
│   │   ├── train_lora.py # Ranking model with LoRA
│   │   └── test_lora.py # Ranking model with LoRA
│   └── utils/
│       ├── testmetrics.py # Compute test metrics
│       └── testplots.py # Create test plots
└── requirements.txt
```

## Data Format

CSV files are expected to contain:

```csv
Sequence,Fitness
MTPSDIPGY...,48498
MTPSDIPGY...,23712
```

`Sequence` is the protein sequence string. `Fitness` is the measured numeric
fitness value used for ranking and stratified splitting.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
conda create -n env python=3.10 
conda activate env
pip install -r requirements.txt
```


## Prepare Data Splits

The repository already includes `train.csv`, `val.csv`, and `test.csv`. To
regenerate them from `data/data.csv`:

```bash
cd data
python split_data.py --input data.csv --output-dir .
```

By default this creates an 80/10/10 train/validation/test split stratified by
quantile bins of the `Fitness` column.

## Training

Run commands from the repository root.

### Pairwise Binary Classification

This model ranks lists of size 2 (binary classification).

```bash
python binaryClassification/train.py
```

The best checkpoint is saved as:

```text
best_model_n2.pt
```

### ListMLE Ranking

This model ranks lists of configured size.
```bash
python listMLE/listsize/train.py \
  --list_size 4 \
  --batch_size 256 \
  --sample_size 10000
```

Optional arguments:

```text
--list_size        Number of sequences per ranked list
--batch_size       Training batch size
--sample_size      Number of sequence combinations to sample
--checkpoint_path  Custom checkpoint output path
```

If no checkpoint path is provided, the script saves:

```text
best_model_listmle_listsize_<list_size>.pt
```

### LoRA ListMLE

This model ranks lists of size 4 with LoRA (on query, key, and value modules).


```bash
python listMLE/lora/train_lora.py \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 256 \
  --num_epochs 100 \
  --sample_size 10000
```

If no checkpoint path is provided, the script saves:

```text
best_model_listmle_lora_r<r>_alpha<alpha>.pt
```

## Evaluation

### Pairwise Binary Classification

```bash
python binaryClassification/test.py
```

This loads `best_model_n2.pt`, reports test loss and accuracy, and writes:

```text
roc_auc_curve.png
confusion_matrix.png
```

### ListMLE

```bash
python listMLE/listsize/test.py
```

The test script expects a checkpoint name defined inside the file,
such as:

```python
name = "best_model_listmle_listsize_16"
```

It writes NumPy arrays containing targets and predictions:

```text
<name>_targets.npy
<name>_preds.npy
```

### LoRA ListMLE

```bash
python listMLE/lora/test_lora.py
```

The LoRA test script also expects the checkpoint base name to be set inside the
file, for example:

```python
name = "best_model_listmle_lora_r1_alpha2"
```

## Metrics and Plots

After generating ListMLE prediction arrays, use:

```bash
python listMLE/utils/testmetrics.py
python listMLE/utils/testplots.py
```

The utilities compute ranking metrics such as Spearman correlation, top-1
accuracy, and pairwise ranking accuracy, and can generate rank heatmaps and
pairwise accuracy plots.