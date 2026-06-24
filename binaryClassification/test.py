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

# Evaluation script for the binary pairwise checkpoint produced by train.py.
# It reports test accuracy and writes ROC/confusion-matrix figures.
test_df = pd.read_csv("./data/test.csv")

model_id = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_id)

class rankingModel(nn.Module):
    """Same pairwise ESM2 scorer used during binaryClassification/train.py."""

    def __init__(self):
        super().__init__()
        self.esm = EsmForMaskedLM.from_pretrained(model_id)
        self.tokenizer = EsmTokenizer.from_pretrained(model_id)

        for name, param in self.esm.named_parameters():
            if 'contact_head.regression' in name:
                param.requires_grad = False


    def forward(self, x):
        # The DataLoader emits flattened adjacent pairs; this forward pass
        # scores each adjacent pair and returns one logit per comparison.
        outputs = self.esm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"]
        )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        pairs = [(i, i + 1) for i in range(0, len(log_probs), 2)]

        result = []
        for i,j in pairs:
            # Positive score means the first sequence in the pair is predicted
            # to outrank the second sequence.
            diff = log_probs[i] - log_probs[j]
            diff = diff.sum()
            result.append(diff)

        result = torch.stack(result) 

        return result
        # using BCEWithLogitsLoss so no need for sigmoid
        # return self.sigmoid(log_probs1 - log_probs2) # P(fitness(x_1) > fitness(x_2))




def loadData(df):
    seqs = df["Sequence"].tolist()
    data_x = list(combinations(seqs, 2))
    # Flatten pairs because rankingModel reconstructs comparisons from
    # adjacent rows: [seq_i, seq_j, seq_i, seq_k, ...].
    data_x = [x for pair in data_x for x in pair] # flattens n pairs into 2n list
    fitness = df["Fitness"].tolist()
    data_y = list(combinations(fitness, 2))
    data_y = [y for pair in data_y for y in pair]

    return data_x, data_y

test_x, test_y = loadData(test_df)

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


def collate_fn(batch):
    """
    Tokenizes protein strings into tensors usable by HuggingFace ESM models.
    """

    if isinstance(batch[0], tuple):
        sequences, labels = zip(*batch)

        encoded = tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        labels = torch.tensor(labels)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    else:
        sequences = batch

        encoded = tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }



test_dataset = ProteinSequenceDataset(test_x, test_y)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False
)






criterion = nn.BCEWithLogitsLoss() 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = rankingModel().to(device)
checkpoint = torch.load("best_model_n2.pt")

model.load_state_dict(checkpoint["model_state_dict"])


model.eval()  
running_test_loss = 0.0
test_correct = 0
test_total = 0

all_targets = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch)

        # Recreate binary labels from the paired fitness values: 1 if the
        # first sequence is fitter than the second, else 0.
        pairs = [(i, i + 1) for i in range(0, len(batch['labels']), 2)]

        targets = []
        for i, j in pairs:
            targets.append(int(batch['labels'][i] > batch['labels'][j]))

        targets = torch.tensor(targets).float().to(device)

        loss = criterion(outputs, targets)
    
        running_test_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

        test_correct += (preds == targets).sum().item()
        test_total += targets.numel()

        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

avg_test_accuracy = test_correct / test_total
avg_test_loss = running_test_loss / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f} | "
      f"Test Accuracy: {avg_test_accuracy:.4f}")        


    

all_targets = np.array(all_targets)
all_probs = np.array(all_probs)
all_preds = (all_probs >= 0.5).astype(int)


# ROC uses the sigmoid probabilities from the pairwise logits.
fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    "roc_auc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


cm = confusion_matrix(all_targets, all_preds)

# Confusion-matrix labels describe the underlying pairwise fitness comparison.
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["label_i <= label_j", "label_i > label_j"]
)

disp.plot(values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
