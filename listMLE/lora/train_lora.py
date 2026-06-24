import argparse
import pandas as pd
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmForMaskedLM
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

from peft import LoraConfig, get_peft_model


# LoRA version of the ListMLE experiment. The base ESM2 model is wrapped with
# low-rank adapters so only a small number of attention parameters are trained.
# =========================
# Argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train ESM2 ListMLE model with LoRA")

    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank r"
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of sequence combinations to sample"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to save best checkpoint"
    )

    return parser.parse_args()


args = parse_args()


# =========================
# Load data
# =========================

train_df = pd.read_csv("./data/train.csv")
val_df = pd.read_csv("./data/val.csv")
test_df = pd.read_csv("./data/test.csv")

model_id = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_id)


# =========================
# Loss and metrics
# =========================

def listwise_ranking_loss_batch(predicts, targets):
    """
    Stable batched ListMLE loss.

    predicts: [B, N]
    targets:  [B, N]
    """
    # ListMLE compares the model scores against the target-induced ordering,
    # not against the absolute scale of the fitness values.
    indices = targets.sort(descending=True, dim=-1).indices
    predicts_sorted = torch.gather(predicts, dim=1, index=indices)

    # Compute suffix denominators in log-space to avoid exp overflow.
    log_cumsums = torch.logcumsumexp(
        predicts_sorted.flip(dims=[1]),
        dim=1
    ).flip(dims=[1])

    loss = log_cumsums - predicts_sorted
    return loss.sum(dim=1).mean()


def pairwise_ranking_accuracy_batch(preds, targets):
    """
    preds:   Tensor of shape [batch_size, list_size]
    targets: Tensor of shape [batch_size, list_size]

    Returns:
        scalar accuracy = fraction of item pairs ranked correctly
    """
    batch_size, list_size = preds.shape

    total_correct = 0
    total_pairs = 0

    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    for b in range(batch_size):
        p = preds[b]
        y = targets[b]

        for i, j in combinations(range(list_size), 2):
            if y[i] == y[j]:
                continue

            true_order = y[i] > y[j]
            pred_order = p[i] > p[j]

            total_correct += int(true_order == pred_order)
            total_pairs += 1

    if total_pairs == 0:
        return 0.0

    return total_correct / total_pairs


# =========================
# LoRA ranking model
# =========================

class rankingModel(nn.Module):
    def __init__(
        self,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()

        base_model = EsmForMaskedLM.from_pretrained(model_id)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            # ESM attention modules use these names; LoRA inserts adapters into
            # the attention projections while leaving most base weights frozen.
            target_modules=["query", "key", "value"],
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.esm = get_peft_model(base_model, lora_config)
        self.tokenizer = EsmTokenizer.from_pretrained(model_id)
        for name, param in self.esm.named_parameters():
            if 'contact_head.regression' in name:
                param.requires_grad = False

        self.esm.print_trainable_parameters()

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        B, N, L = input_ids.shape

        # Treat each protein independently for the ESM call, then regroup the
        # scalar sequence scores back into candidate lists.
        flat_input_ids = input_ids.reshape(B * N, L)
        flat_attention_mask = attention_mask.reshape(B * N, L)

        outputs = self.esm(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Score a sequence by summing the model log-probability of the observed
        # token at each non-padding position.
        token_log_probs = log_probs.gather(
            dim=-1,
            index=flat_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        token_log_probs = token_log_probs * flat_attention_mask

        seq_scores = token_log_probs.sum(dim=1)
        seq_scores = seq_scores.reshape(B, N)

        return seq_scores


# =========================
# Data preparation
# =========================

def loadData_batch(df, c, sample_size=10000):
    """Create random sampled groups of c sequences and their fitness values."""
    seqs = df["Sequence"].tolist()
    data_x = list(combinations(seqs, c))

    fitness = df["Fitness"].tolist()
    data_y = list(combinations(fitness, c))

    sample_size = min(sample_size, len(data_x))

    # Sample from all possible c-combinations to keep epochs bounded.
    indices = np.random.choice(
        len(data_x),
        size=sample_size,
        replace=False
    )

    data_x = [data_x[i] for i in indices]
    data_y = [data_y[i] for i in indices]

    return data_x, data_y


train_x, train_y = loadData_batch(
    train_df,
    c=4,
    sample_size=args.sample_size
)

val_x, val_y = loadData_batch(
    val_df,
    c=4,
    sample_size=args.sample_size
)

test_x, test_y = loadData_batch(
    test_df,
    c=4,
    sample_size=args.sample_size
)


class ProteinSequenceDataset(Dataset):
    """Thin dataset wrapper around listwise sequence groups and fitness lists."""

    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.sequences[idx]

        return self.sequences[idx], self.labels[idx]


def collate_fn_batch(batch):
    sequences_list, labels = zip(*batch)

    batch_size = len(sequences_list)
    num_sequences = len(sequences_list[0])

    # HuggingFace tokenizers operate on flat lists of strings, so flatten first
    # and restore [batch_size, list_size, seq_len] afterward.
    flat_sequences = [
        seq
        for sequences in sequences_list
        for seq in sequences
    ]

    encoded = tokenizer(
        flat_sequences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].reshape(batch_size, num_sequences, -1)
    attention_mask = encoded["attention_mask"].reshape(batch_size, num_sequences, -1)

    labels = torch.tensor(labels, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = ProteinSequenceDataset(train_x, train_y)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn_batch,
    drop_last=False
)

val_dataset = ProteinSequenceDataset(val_x, val_y)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn_batch,
    drop_last=False
)

test_dataset = ProteinSequenceDataset(test_x, test_y)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn_batch,
    drop_last=False
)


# =========================
# Training setup
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print(f"LoRA r: {args.lora_r}")
print(f"LoRA alpha: {args.lora_alpha}")
print(f"LoRA dropout: {args.lora_dropout}")
print(f"Learning rate: {args.lr}")
print(f"Batch size: {args.batch_size}")
print(f"Num epochs: {args.num_epochs}")

model = rankingModel(
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

best_val_loss = float("inf")

if args.checkpoint_path is None:
    checkpoint_path = (
        f"best_model_listmle_lora_"
        f"r{args.lora_r}_alpha{args.lora_alpha}.pt"
    )
else:
    checkpoint_path = args.checkpoint_path

print(f"Checkpoint path: {checkpoint_path}")


# =========================
# Training loop
# =========================

for epoch in range(args.num_epochs):
    model.train()
    running_train_loss = 0.0
    train_acc = 0.0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(batch)
        targets = batch["labels"].float()

        loss = listwise_ranking_loss_batch(outputs, targets)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        train_acc += pairwise_ranking_accuracy_batch(outputs, targets)

    avg_train_accuracy = train_acc / len(train_loader)
    avg_train_loss = running_train_loss / len(train_loader)

    model.eval()
    running_val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch)
            targets = batch["labels"].float()

            loss = listwise_ranking_loss_batch(outputs, targets)

            running_val_loss += loss.item()
            val_acc += pairwise_ranking_accuracy_batch(outputs, targets)

    avg_val_accuracy = val_acc / len(val_loader)
    avg_val_loss = running_val_loss / len(val_loader)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        # Persist the LoRA hyperparameters with the weights so eval scripts can
        # reconstruct the same adapter shape before loading state_dict.
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }, checkpoint_path)

        print(f"✓ Saved new best LoRA model checkpoint: val_loss={avg_val_loss:.4f}")

    print(
        f"Epoch [{epoch + 1}/{args.num_epochs}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Accuracy: {avg_train_accuracy:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Accuracy: {avg_val_accuracy:.4f}"
    )


# =========================
# Testing
# =========================

checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model.eval()

running_test_loss = 0.0
test_acc = 0.0

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch)
        targets = batch["labels"].float()

        loss = listwise_ranking_loss_batch(outputs, targets)

        running_test_loss += loss.item()
        test_acc += pairwise_ranking_accuracy_batch(outputs, targets)

avg_test_accuracy = test_acc / len(test_loader)
avg_test_loss = running_test_loss / len(test_loader)

print(
    f"Test Loss: {avg_test_loss:.4f} | "
    f"Test Accuracy: {avg_test_accuracy:.4f}"
)
