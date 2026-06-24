import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmForMaskedLM
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Evaluate a ListMLE list-size checkpoint and save raw target/prediction arrays
# for downstream metrics/plotting utilities.
test_df = pd.read_csv("./data/test.csv")

model_id = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_id)

def listwise_ranking_loss_batch(predicts, targets):
    """
    Stable batched ListMLE loss.

    predicts: [B, N]
    targets:  [B, N]
    """
    # Sort predictions into the ground-truth fitness order before applying the
    # Plackett-Luce/ListMLE negative log-likelihood.
    indices = targets.sort(descending=True, dim=-1).indices
    predicts_sorted = torch.gather(predicts, dim=1, index=indices)

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

class rankingModel(nn.Module):
    """Scores each sequence in every candidate list with the frozen ESM2 head."""

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

        # Flatten [batch, list_size, seq_len] for ESM, then reshape the summed
        # sequence scores back to one row per candidate list.
        flat_input_ids = input_ids.view(B * N, L)
        flat_attention_mask = attention_mask.view(B * N, L)

        outputs = self.esm(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )

        logits = outputs.logits                      # [B * N, L, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather log-probabilities for the actual input tokens. Summing these
        # token scores gives each protein sequence one scalar ranking score.
        token_log_probs = log_probs.gather(
            dim=-1,
            index=flat_input_ids.unsqueeze(-1)
        ).squeeze(-1)                                # [B * N, L]

        # Ignore padding tokens
        token_log_probs = token_log_probs * flat_attention_mask

        seq_scores = token_log_probs.sum(dim=1)      # [B * N]
        seq_scores = seq_scores.view(B, N)           # [B, N]

        return seq_scores



def loadData_batch(df, c, sample_size=10000):
    """Enumerate c-sized groups and sample a subset for evaluation."""
    seqs = df["Sequence"].tolist()
    data_x = list(combinations(seqs, c))

    fitness = df["Fitness"].tolist()
    data_y = list(combinations(fitness, c))

    sample_size = min(sample_size, len(data_x))

    # Sampling keeps evaluation tractable when n choose c is large.
    indices = np.random.choice(
        len(data_x),
        size=sample_size,
        replace=False
    )

    data_x = [data_x[i] for i in indices]
    data_y = [data_y[i] for i in indices]

    return data_x, data_y


test_x, test_y = loadData_batch(test_df, 4, 10000)

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


test_dataset = ProteinSequenceDataset(test_x, test_y)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    collate_fn=collate_fn_batch,
    drop_last=True
)








device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = rankingModel().to(device)

name = "best_model_listmle_listsize_16"

checkpoint = torch.load(name + ".pt")
model.load_state_dict(checkpoint["model_state_dict"])


model.eval()  
running_test_loss = 0.0
test_acc = 0.0
all_targets = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        targets = batch["labels"].float()        

        # Evaluate both the ListMLE objective and the easier-to-interpret
        # pairwise ordering accuracy for each sampled list.
        loss = listwise_ranking_loss_batch(outputs, targets)
    
        running_test_loss += loss.item()

        test_acc += pairwise_ranking_accuracy_batch(outputs, targets)

        all_targets.append(targets.cpu().numpy())
        all_preds.append(outputs.cpu().numpy())

avg_test_accuracy = test_acc / len(test_loader)
avg_test_loss = running_test_loss / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f} | "
      f"Test Accuracy: {avg_test_accuracy:.4f}")        


    

all_targets = np.array(all_targets)
all_preds = np.array(all_preds)



# These .npy files are consumed by listMLE/utils/testmetrics.py and
# listMLE/utils/testplots.py.
np.save(name + "_targets.npy", all_targets)
np.save(name + "_preds.npy", all_preds)
