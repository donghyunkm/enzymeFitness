import argparse
import pandas as pd
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmForMaskedLM
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from itertools import combinations
import random
import numpy as np
from math import comb

# List-size experiment: train ESM2 with a listwise ranking objective on sampled
# groups of c sequences. Increase --list_size to ask the model to order longer
# candidate lists in one forward pass.
def parse_args():
    parser = argparse.ArgumentParser(description="Train ESM2 ListMLE model with LoRA")

    parser.add_argument(
        "--list_size",
        type=int,
        default=4,
        help="list size"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="list size"
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

train_df = pd.read_csv("./data/train.csv")
val_df = pd.read_csv("./data/val.csv")
test_df = pd.read_csv("./data/test.csv")

model_id = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_id)

def listwise_ranking_loss_batch(predicts, targets):
    """
    Stable batched ListMLE loss.

    predicts: [B, N]
    targets:  [B, N]
    """
    # Sort each list by true fitness. ListMLE then rewards high model scores
    # appearing early in that ground-truth ordering.
    indices = targets.sort(descending=True, dim=-1).indices
    predicts_sorted = torch.gather(predicts, dim=1, index=indices)

    # logcumsumexp computes each denominator in log-space for numerical
    # stability: log(sum(exp(score_k))) over the remaining suffix of the list.
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

    for b in range(batch_size):
        p = preds[b]
        y = targets[b]

        for i, j in combinations(range(list_size), 2):
            # Skip ties in ground truth
            if y[i] == y[j]:
                continue

            true_order = y[i] > y[j]
            pred_order = p[i] > p[j]

            total_correct += int(true_order == pred_order)
            total_pairs += 1

    if total_pairs == 0:
        return 0.0

    return total_correct / total_pairs

def listwise_ranking_loss(predicts, targets):
    """
    ListMLE loss for 1D inputs.

    predicts: shape [list_size]
    targets:  shape [list_size]
    """

    indices = torch.argsort(targets, descending=True, dim=0)
    predicts_sorted = predicts[indices]

    log_cumsums = torch.logcumsumexp(
        predicts_sorted.flip(dims=[0]),
        dim=0
    ).flip(dims=[0])

    loss = log_cumsums - predicts_sorted

    return loss.sum()

def pairwise_ranking_accuracy(preds, targets):
    """
    preds:   Tensor or list of shape [list_size]
    targets: Tensor or list of shape [list_size]

    Returns:
        Pairwise ranking accuracy.
    """

    preds = torch.tensor(preds, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    n = len(preds)

    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):

            # skip ties in ground truth
            if targets[i] == targets[j]:
                continue

            true_order = targets[i] > targets[j]
            pred_order = preds[i] > preds[j]

            if true_order == pred_order:
                correct += 1

            total += 1

    if total == 0:
        return 0.0

    return correct / total

class rankingModel(nn.Module):
    """Scores every sequence in a [batch, list_size] group with ESM2."""

    def __init__(self):
        super().__init__()
        self.esm = EsmForMaskedLM.from_pretrained(model_id)
        self.tokenizer = EsmTokenizer.from_pretrained(model_id)

        for name, param in self.esm.named_parameters():
            if 'contact_head.regression' in name:
                param.requires_grad = False



    def forward(self, x):
        input_ids = x["input_ids"]           
        attention_mask = x["attention_mask"]

        B, N, L = input_ids.shape

        # ESM expects a flat batch of sequences. Flatten list groups before the
        # transformer call, then reshape scores back to [batch, list_size].
        flat_input_ids = input_ids.view(B * N, L)
        flat_attention_mask = attention_mask.view(B * N, L)

        outputs = self.esm(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )

        logits = outputs.logits                      # [B * N, L, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Use the model's log-probability of the observed amino-acid tokens as
        # an unsupervised sequence score, masking padding before summing.
        token_log_probs = log_probs.gather(
            dim=-1,
            index=flat_input_ids.unsqueeze(-1)
        ).squeeze(-1)                                # [B * N, L]

        # Ignore padding tokens
        token_log_probs = token_log_probs * flat_attention_mask

        seq_scores = token_log_probs.sum(dim=1)      # [B * N]
        seq_scores = seq_scores.view(B, N)           # [B, N]

        return seq_scores


def loadData_batch(df, c, sample_size=10000, seed=42):
    """Sample fixed-size sequence groups without enumerating all combinations."""
    seqs = df["Sequence"].to_numpy()
    fitness = df["Fitness"].to_numpy()

    n = len(df)

    if c > n:
        raise ValueError(f"c={c} cannot be larger than number of rows n={n}")

    rng = np.random.default_rng(seed)

    data_x = []
    data_y = []

    for _ in range(sample_size):
        # Random subsets avoid the combinatorial explosion from n choose c.
        idx = rng.choice(n, size=c, replace=False)

        data_x.append(tuple(seqs[idx]))
        data_y.append(tuple(fitness[idx]))

    return data_x, data_y


train_x, train_y = loadData_batch(train_df, args.list_size, args.sample_size)
val_x, val_y = loadData_batch(val_df, args.list_size, args.sample_size)
test_x, test_y = loadData_batch(test_df, args.list_size, args.sample_size)

class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        """
        sequences: list[str]
            Protein sequences, e.g. ["MKT...", "GAVL..."]

        labels: list or tensor, optional
            Regression or classification labels.
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.sequences[idx]

        return self.sequences[idx], self.labels[idx]


def collate_fn_batch(batch):
    """
    Tokenizes batches where each x is a list of protein sequences.

    Example batch item:
        (["SEQ1", "SEQ2"], label)

    Returns tensors with shape:
        input_ids:      [batch_size, num_sequences, seq_len]
        attention_mask: [batch_size, num_sequences, seq_len]
        labels:         [batch_size]
    """

    sequences_list, labels = zip(*batch)
    # sequences_list is something like:
    # (
    #   ["AAA", "BBB"],
    #   ["CCC", "DDD"],
    #   ...
    # )

    batch_size = len(sequences_list)
    num_sequences = len(sequences_list[0])

    # Flatten from [batch_size, num_sequences] to [batch_size * num_sequences]
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

    # Reshape back to [batch_size, num_sequences, seq_len]
    input_ids = encoded["input_ids"].view(batch_size, num_sequences, -1)
    attention_mask = encoded["attention_mask"].view(batch_size, num_sequences, -1)

    labels = torch.tensor(labels)

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


epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print(f"List size: {args.list_size}")
print(f"Sample size: {args.sample_size}")


model = rankingModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
best_val_loss = float("inf")

if args.checkpoint_path is None:
    checkpoint_path = (
        f"best_model_listmle_"
        f"listsize_{args.list_size}.pt"
    )
else:
    checkpoint_path = args.checkpoint_path

print(f"Checkpoint path: {checkpoint_path}")


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    train_acc = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch)
        targets = batch["labels"].float()

        # targets remain raw fitness values; ListMLE uses only their ordering.
        loss = listwise_ranking_loss_batch(outputs, targets)
        
        loss.backward()
        
        optimizer.step()
        
        running_train_loss += loss.item()
        
        train_acc += pairwise_ranking_accuracy_batch(outputs, targets)

    avg_train_accuracy = train_acc / len(train_loader)
    avg_train_loss = running_train_loss / len(train_loader)
    
    model.eval()  
    running_val_loss = 0.0
    val_acc=0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            targets = batch["labels"].float()

            loss = listwise_ranking_loss_batch(outputs, targets)
        
            running_val_loss += loss.item()

            val_acc += pairwise_ranking_accuracy_batch(outputs, targets)


    avg_val_accuracy = val_acc/len(val_loader)

    avg_val_loss = running_val_loss / len(val_loader)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the best validation checkpoint so the final test pass below
        # evaluates the strongest epoch, not necessarily the last one.
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "list_size": args.list_size
        }, checkpoint_path)

        print(f"✓ Saved new best model (val_loss={avg_val_loss:.4f})")


    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Accuracy: {avg_train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Accuracy: {avg_val_accuracy:.4f}")


# Testing
    
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 


model.eval()  
running_test_loss = 0.0
test_acc=0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        targets = batch["labels"].float()        

        loss = listwise_ranking_loss_batch(outputs, targets)
    
        running_test_loss += loss.item()

        test_acc += pairwise_ranking_accuracy_batch(outputs, targets)


avg_test_accuracy = test_acc/len(test_loader)

avg_test_loss = running_test_loss / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f} | "
        f"Test Accuracy: {avg_test_accuracy:.4f}")
