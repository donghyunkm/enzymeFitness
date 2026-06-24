#!/usr/bin/env python3
"""Create stratified train/val/test CSV splits from data.csv."""

import argparse
import csv
import math
from pathlib import Path

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a CSV into train/val/test files stratified by fitness."
    )
    parser.add_argument("--input", default="data.csv", help="Input CSV path.")
    parser.add_argument(
        "--fitness-column", default="Fitness", help="Column used for stratification."
    )
    parser.add_argument("--train-size", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction.")
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of quantile bins to use for stratifying fitness values.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default=".", help="Directory for split CSV files.")
    return parser.parse_args()


def read_csv(path, fitness_column):
    """Read the source CSV and verify the requested fitness column exists."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError(f"{path} has no header row.")
    if fitness_column not in fieldnames:
        raise ValueError(f"Missing fitness column: {fitness_column}")

    return rows, fieldnames


def fitness_bins(rows, fitness_column, bins):
    """Assign rows to quantile-like bins so splits preserve fitness coverage."""
    if bins < 1:
        raise ValueError("--bins must be at least 1.")

    # Rank by fitness and then divide the ordered rows into approximately equal
    # bins. These bin labels are used only for stratification, not modeling.
    sorted_indexes = sorted(
        range(len(rows)), key=lambda index: float(rows[index][fitness_column])
    )
    bin_count = min(bins, len(rows))
    labels = [0] * len(rows)

    for rank, row_index in enumerate(sorted_indexes):
        labels[row_index] = min(math.floor(rank * bin_count / len(rows)), bin_count - 1)

    return labels


def split_data(rows, fitness_column, train_size, val_size, test_size, bins, seed):
    total_size = train_size + val_size + test_size
    if not math.isclose(total_size, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--train-size, --val-size, and --test-size must sum to 1.0.")

    stratify = fitness_bins(rows, fitness_column, bins)
    # First split off train, then split the remainder into validation/test while
    # preserving the same fitness-bin distribution in each split.
    train_rows, temp_rows, _, temp_bins = train_test_split(
        rows,
        stratify,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )

    relative_val_size = val_size / (val_size + test_size)
    val_rows, test_rows = train_test_split(
        temp_rows,
        train_size=relative_val_size,
        random_state=seed,
        stratify=temp_bins,
    )

    return train_rows, val_rows, test_rows


def write_csv(path, rows, fieldnames):
    """Write one split with the same columns as the source CSV."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = read_csv(args.input, args.fitness_column)
    train_rows, val_rows, test_rows = split_data(
        rows,
        args.fitness_column,
        args.train_size,
        args.val_size,
        args.test_size,
        args.bins,
        args.seed,
    )

    splits = {
        "train.csv": train_rows,
        "val.csv": val_rows,
        "test.csv": test_rows,
    }
    for filename, split_rows in splits.items():
        path = output_dir / filename
        write_csv(path, split_rows, fieldnames)
        print(f"Wrote {len(split_rows)} rows to {path}")


if __name__ == "__main__":
    main()
