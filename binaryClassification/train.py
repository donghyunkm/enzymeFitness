import pandas as pd
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmForMaskedLM
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from itertools import combinations

# Binary baseline: train ESM2 to answer "is sequence_i fitter than sequence_j?"
# Each training example is an unordered pair of sequences from the same split.
train_df = pd.read_csv("./data/train.csv")
val_df = pd.read_csv("./data/val.csv")
test_df = pd.read_csv("./data/test.csv")

model_id = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_id)
class rankingModel(nn.Module):
    """ESM2 scorer that converts each adjacent sequence pair into one comparison logit."""

    def __init__(self):
        super().__init__()
        self.esm = EsmForMaskedLM.from_pretrained(model_id)
        self.tokenizer = EsmTokenizer.from_pretrained(model_id)

        for name, param in self.esm.named_parameters():
            if 'contact_head.regression' in name:
                param.requires_grad = False


    def forward(self, x):
        # Input batches are laid out as [seq_a, seq_b, seq_c, seq_d, ...].
        # Adjacent rows form the comparison pairs produced by loadData().
        outputs = self.esm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"]
        )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        pairs = [(i, i + 1) for i in range(0, len(log_probs), 2)]

        result = []
        for i,j in pairs:
            # A positive summed log-probability difference means the first
            # sequence in the pair is scored as more likely/fit than the second.
            diff = log_probs[i] - log_probs[j]
            diff = diff.sum()
            result.append(diff)

        result = torch.stack(result) 

        return result
        # using BCEWithLogitsLoss so no need for sigmoid
        # return self.sigmoid(log_probs1 - log_probs2) # P(fitness(x_1) > fitness(x_2))


def loadData(df):
    """Build flattened pairwise sequence/fitness lists from one split."""
    seqs = df["Sequence"].tolist()
    data_x = list(combinations(seqs, 2))
    # Flatten [(a, b), (a, c)] into [a, b, a, c] so each DataLoader batch can
    # be re-grouped into adjacent pairs inside the model/training loop.
    data_x = [x for pair in data_x for x in pair]

    fitness = df["Fitness"].tolist()
    data_y = list(combinations(fitness, 2))
    data_y = [y for pair in data_y for y in pair]

    return data_x, data_y



train_x, train_y = loadData(train_df)
val_x, val_y = loadData(val_df)
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



train_dataset = ProteinSequenceDataset(train_x, train_y)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=False
)

val_dataset = ProteinSequenceDataset(val_x, val_y)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False
)

test_dataset = ProteinSequenceDataset(test_x, test_y)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False
)



model = rankingModel()
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = rankingModel().to(device)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01)
best_val_loss = float("inf")


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Targets are generated on the fly from the original fitness values:
        # 1 means the first sequence in the adjacent pair has higher fitness.
        pairs = [(i, i + 1) for i in range(0, len(batch['labels']), 2)]

        targets = []
        for i,j in pairs:
            targets.append(int(batch['labels'][i] > batch['labels'][j]))
        targets = torch.tensor(targets).float().to(device)

        loss = criterion(outputs, targets)
        
        loss.backward()
        
        optimizer.step()
        
        running_train_loss += loss.item()
        
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        train_correct += (preds == targets.float()).sum().item()
        train_total += targets.numel()

    avg_train_accuracy = train_correct/train_total
    avg_train_loss = running_train_loss / len(train_loader)
    
    model.eval()  
    running_val_loss = 0.0
    val_correct=0
    val_total=0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            # Keep validation target construction identical to training so the
            # reported accuracy is pairwise classification accuracy.
            pairs = [(i, i + 1) for i in range(0, len(batch['labels']), 2)]

            targets = []
            for i,j in pairs:
                targets.append(int(batch['labels'][i] > batch['labels'][j]))
            targets = torch.tensor(targets).float().to(device)

            loss = criterion(outputs, targets)
        
            running_val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            val_correct += (preds == targets.float()).sum().item()
            val_total += targets.numel()

    avg_val_accuracy = val_correct/val_total

    avg_val_loss = running_val_loss / len(val_loader)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        }, "best_model_n2.pt")

        print(f"✓ Saved new best model (val_loss={avg_val_loss:.4f})")


    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Accuracy: {avg_train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Accuracy: {avg_val_accuracy:.4f}")


# Testing
    
checkpoint = torch.load("best_model_n2.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 


model.eval()  
running_test_loss = 0.0
test_correct=0
test_total=0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        # Final test pass uses the best validation checkpoint saved above.
        pairs = [(i, i + 1) for i in range(0, len(batch['labels']), 2)]

        targets = []
        for i,j in pairs:
            targets.append(int(batch['labels'][i] > batch['labels'][j]))
        targets = torch.tensor(targets).float().to(device)

        loss = criterion(outputs, targets)
    
        running_test_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        test_correct += (preds == targets.float()).sum().item()
        test_total += targets.numel()

avg_test_accuracy = test_correct/test_total

avg_test_loss = running_test_loss / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f} | "
        f"Test Accuracy: {avg_test_accuracy:.4f}")
