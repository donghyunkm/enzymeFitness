import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from itertools import combinations

import numpy as np
from itertools import combinations


# Utility script for summarizing saved ListMLE test outputs. It expects the
# *_targets.npy and *_preds.npy files written by the listMLE test scripts.
def evaluate_listmle_predictions(preds, targets):
    """
    preds:   array-like, shape [num_batches, batch_size, list_size]
    targets: array-like, shape [num_batches, batch_size, list_size]

    Returns ranking and regression-style metrics.
    """

    # Convert to numpy
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    assert preds.shape == targets.shape
    assert preds.ndim == 3

    num_batches, batch_size, list_size = preds.shape

    # Flatten batch dimension:
    # (197, 64, 4) -> (197 * 64, 4)
    preds_flat = preds.reshape(-1, list_size)
    targets_flat = targets.reshape(-1, list_size)

    spearman_scores = []
    top1_correct = []
    pairwise_correct = []

    for p, y in zip(preds_flat, targets_flat):
        # Spearman correlation within each list
        sp = spearmanr(p, y).correlation
        if not np.isnan(sp):
            spearman_scores.append(sp)


        # Top-1 accuracy: did model pick the true best item?
        pred_top = np.argmax(p)
        true_top = np.argmax(y)
        top1_correct.append(pred_top == true_top)

        # Pairwise ranking accuracy: average over all non-tied item pairs in
        # each candidate list, then average those per-list accuracies.
        correct = 0
        total = 0
        for i in range(list_size):
            for j in range(i + 1, list_size):
                true_order = y[i] > y[j]
                pred_order = p[i] > p[j]

                # ignore exact ties in target
                if y[i] == y[j]:
                    continue

                correct += int(true_order == pred_order)
                total += 1

        if total > 0:
            pairwise_correct.append(correct / total)

    # Global regression metrics after flattening all scores
    preds_all = preds.reshape(-1)
    targets_all = targets.reshape(-1)

    metrics = {
        "spearman_mean_per_list": np.mean(spearman_scores),
        "spearman_std_per_list": np.std(spearman_scores),


        "top1_accuracy": np.mean(top1_correct),
        "pairwise_accuracy": np.mean(pairwise_correct),

    }

    return metrics

# Change this stem to evaluate a different checkpoint output pair.
name = "best_model_listmle_listsize_16"
targets = np.load(name + "_targets.npy")
preds = np.load(name + "_preds.npy")

metrics = evaluate_listmle_predictions(preds, targets)
print(metrics)
