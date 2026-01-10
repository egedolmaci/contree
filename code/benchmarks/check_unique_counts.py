#!/usr/bin/env python3
import os
import subprocess
import sys

import pandas as pd


EPSILON = 1e-7
REPO_ROOT = "/home/mertrodop/406_project"
DATASET_DIR = os.path.join(REPO_ROOT, "contree", "datasets")
BINARY = os.path.join(REPO_ROOT, "contree", "build", "ConTree")


def smart_load_dataset(path):
    delimiters = [",", r"\s+", ";", "\t"]
    for sep in delimiters:
        try:
            df = pd.read_csv(path, header=None, sep=sep, engine="python")
            if df.shape[1] <= 1:
                continue
            first_col = df.iloc[:, 0]
            last_col = df.iloc[:, -1]
            is_first_integer = pd.api.types.is_integer_dtype(first_col) or first_col.nunique() < 20
            is_last_integer = pd.api.types.is_integer_dtype(last_col) or last_col.nunique() < 20
            label_loc = "LAST"
            if is_first_integer and not is_last_integer:
                label_loc = "FIRST"
            elif is_first_integer and is_last_integer:
                label_loc = "FIRST"
            return df, label_loc
        except Exception:
            continue
    return None, None


def extract_features(df, label_loc):
    if label_loc == "FIRST":
        X = df.iloc[:, 1:]
    else:
        X = df.iloc[:, :-1]
    X.columns = range(X.shape[1])
    return X


def unique_counts_eps(x):
    counts = []
    for col in x.columns:
        vals = sorted(x[col].astype(float).tolist())
        unique_vals = []
        for v in vals:
            if not unique_vals or (v - unique_vals[-1]) >= EPSILON:
                unique_vals.append(v)
        counts.append(len(unique_vals))
    return counts


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_unique_counts.py <dataset_file>")
        return 1

    dataset = sys.argv[1]
    path = dataset
    if not os.path.isabs(path):
        path = os.path.join(DATASET_DIR, dataset)
    if not os.path.isfile(path):
        print(f"Dataset not found: {path}")
        return 1

    df, label_loc = smart_load_dataset(path)
    if df is None:
        print("Failed to parse dataset.")
        return 1

    X = extract_features(df, label_loc)
    counts = unique_counts_eps(X)
    print("CPU unique counts (EPSILON):")
    for i, c in enumerate(counts):
        print(f"feature {i}: {c}")

    if not os.path.isfile(BINARY):
        print(f"Binary not found: {BINARY}")
        return 1

    print("\nGPU unique counts (from logs):")
    cmd = [
        BINARY,
        "-file", path,
        "-max-depth", "2",
        "-use-gpu-bruteforce", "1",
        "-print-logs", "1",
    ]
    subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
