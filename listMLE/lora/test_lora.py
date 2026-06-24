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

from peft import LoraConfig, get_peft_model

# Evaluate a LoRA ListMLE checkpoint. The LoRA hyperparameters below must match
# the adapter configuration used when the checkpoint was trained.
test_df = pd.read_csv("./data/test.csv")

model_id = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_id)

def listwise_ranking_loss_batch(predicts, targets):
    """
    Stable batched ListMLE loss.

    predicts: [B, N]
    targets:  [B, N]
    """
    # Targets are raw fitnesses; sorting them gives the ground-truth ranking
    # that ListMLE uses to order the model scores.
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
            # These adapter locations must match listMLE/lora/train_lora.py.
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

        # Flatten the grouped candidate lists for ESM, then restore one score
        # vector per list for the ListMLE loss/metrics.
        flat_input_ids = input_ids.reshape(B * N, L)
        flat_attention_mask = attention_mask.reshape(B * N, L)

        outputs = self.esm(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather each observed token's log-probability and sum over sequence
        # length to obtain a scalar score per protein.
        token_log_probs = log_probs.gather(
            dim=-1,
            index=flat_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        token_log_probs = token_log_probs * flat_attention_mask

        seq_scores = token_log_probs.sum(dim=1)
        seq_scores = seq_scores.reshape(B, N)

        return seq_scores



def loadData_batch(df, c, sample_size=10000):
    """Enumerate c-sized lists from the test split and sample a subset."""
    seqs = df["Sequence"].tolist()
    data_x = list(combinations(seqs, c))

    fitness = df["Fitness"].tolist()
    data_y = list(combinations(fitness, c))

    sample_size = min(sample_size, len(data_x))

    # Avoid evaluating every combination when the test set is large.
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


model = rankingModel(
    lora_r=1,
    lora_alpha=2,
    lora_dropout=0.05,
).to(device)


# Keep this checkpoint stem synchronized with the LoRA args above.
name = "best_model_listmle_lora_r1_alpha2"

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

        # Store both loss/accuracy summaries and raw arrays for later plotting.
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



# Downstream scripts can compute additional rank metrics from these arrays.
np.save(name + "_targets.npy", all_targets)
np.save(name + "_preds.npy", all_preds)
