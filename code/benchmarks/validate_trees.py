#!/usr/bin/env python3
import ast
import os
import sys

import pandas as pd


DATASET_DIR = "/home/mertrodop/406_project/contree/datasets"


def smart_load_dataset(dataset_name):
    filepath = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(filepath):
        return None, None, "File Not Found"

    delimiters = [",", r"\s+", ";", "\t"]
    for sep in delimiters:
        try:
            df = pd.read_csv(filepath, header=None, sep=sep, engine="python")
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
                label_loc = "TRY_BOTH"

            return df, label_loc, f"Shape {df.shape}"
        except Exception:
            continue

    return None, None, "Parse Failed"


def prepare_data(df, label_position):
    if label_position == "FIRST":
        y_raw = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    else:
        y_raw = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    X.columns = range(X.shape[1])
    y_normalized, _ = pd.factorize(y_raw, sort=True)
    y = pd.Series(y_normalized)
    return X, y


def predict_and_count_errors(node, X_subset, y_subset):
    if len(y_subset) == 0:
        return 0

    if isinstance(node, (int, float)):
        return (y_subset != node).sum()

    if isinstance(node, list):
        if len(node) < 3 or (node[2] == -1 and node[3] == -1):
            majority = y_subset.mode()
            if len(majority) > 0:
                return (y_subset != majority[0]).sum()
            return 0

        feature_idx = int(node[0])
        threshold = float(node[1])

        if feature_idx >= X_subset.shape[1]:
            raise IndexError

        mask_left = X_subset.iloc[:, feature_idx] <= threshold
        err_l = predict_and_count_errors(node[2], X_subset[mask_left], y_subset[mask_left])
        err_r = predict_and_count_errors(node[3], X_subset[~mask_left], y_subset[~mask_left])
        return err_l + err_r

    return 0


def parse_tree(tree_str):
    try:
        if pd.isna(tree_str) or tree_str == "ERROR":
            return None
        return ast.literal_eval(tree_str)
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: validate_trees.py <benchmark_csv>")
        return 1

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return 1

    results = pd.read_csv(csv_path)
    required = {"Dataset", "Depth", "Mode", "Threads", "Misclassification", "TreeStructure"}
    missing = required - set(results.columns)
    if missing:
        print(f"Missing columns: {', '.join(sorted(missing))}")
        return 1

    header = f"{'DATASET':<12} | {'D':<2} | {'MODE':<3} | {'THR':<3} | {'REPORT':<6} | {'CALC':<6} | {'STATUS'} | {'NOTE'}"
    print(header)
    print("-" * len(header))

    for _, row in results.iterrows():
        dataset = row["Dataset"]
        rep_score = row["Misclassification"]
        tree_str = row["TreeStructure"]

        tree = parse_tree(tree_str)
        if tree is None:
            continue

        df, label_loc, _msg = smart_load_dataset(dataset)
        if df is None:
            continue

        possible_locs = ["FIRST", "LAST"] if label_loc == "TRY_BOTH" else [label_loc]
        best_score = -1
        best_diff = 999999
        final_loc = "?"

        for loc in possible_locs:
            try:
                X, y = prepare_data(df, loc)
                calc = predict_and_count_errors(tree, X, y)
                diff = abs(calc - rep_score)
                if diff < best_diff:
                    best_diff = diff
                    best_score = calc
                    final_loc = loc
                if diff == 0:
                    break
            except Exception:
                continue

        status = "❌ DIFF"
        if best_diff == 0:
            status = "✅ MATCH"
        elif best_diff < 10:
            status = "⚠️ CLOSE"

        threads = row["Threads"]
        print(f"{dataset:<12} | {int(row['Depth']):<2} | {row['Mode']:<3} | {int(threads):<3} | {int(rep_score):<6} | {best_score:<6} | {status} | Loc:{final_loc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
