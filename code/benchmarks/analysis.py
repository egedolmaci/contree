#!/usr/bin/env python3
"""
Analyze benchmark_full.csv and generate summary reports.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "Dataset",
    "Depth",
    "Mode",
    "Threads",
    "Time(s)",
    "Accuracy",
    "Misclassification",
    "TreeStructure",
}


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_empty(values: list[float]) -> str:
    if not values:
        return ""
    return f"{statistics.fmean(values):.6f}"


def read_rows(input_path: Path) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    rows: list[dict] = []

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            errors.append("Input file has no header row.")
            return rows, errors

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            errors.append(f"Missing required columns: {missing_list}")
            return rows, errors

        for idx, row in enumerate(reader, start=2):
            dataset = (row.get("Dataset") or "").strip()
            depth = parse_int((row.get("Depth") or "").strip())
            mode = (row.get("Mode") or "").strip()
            threads = parse_int((row.get("Threads") or "").strip())
            time_s = parse_float((row.get("Time(s)") or "").strip())
            accuracy = parse_float((row.get("Accuracy") or "").strip())
            misclassification = parse_int((row.get("Misclassification") or "").strip())
            tree = (row.get("TreeStructure") or "").strip()

            if not dataset or depth is None or not mode or threads is None:
                errors.append(f"Row {idx}: missing required fields.")
                continue

            rows.append(
                {
                    "Dataset": dataset,
                    "Depth": depth,
                    "Mode": mode,
                    "Threads": threads,
                    "Time(s)": time_s,
                    "Accuracy": accuracy,
                    "Misclassification": misclassification,
                    "TreeStructure": tree,
                }
            )

    return rows, errors


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze benchmark_full.csv and write summaries.")
    parser.add_argument(
        "--input",
        default="benchmark_full.csv",
        help="Path to benchmark_full.csv",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_out",
        help="Directory to write analysis outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, errors = read_rows(input_path)
    if errors:
        for message in errors:
            print(f"Warning: {message}", file=sys.stderr)

    if not rows:
        print("No valid rows found. No outputs created.", file=sys.stderr)
        return 1

    summary_keyed: dict[tuple, dict[str, list[float]]] = {}
    mode_times: dict[tuple, list[float]] = {}
    best_by_dataset_depth: dict[tuple, dict] = {}

    valid_time_rows = 0
    for row in rows:
        dataset = row["Dataset"]
        depth = row["Depth"]
        mode = row["Mode"]
        threads = row["Threads"]
        time_s = row["Time(s)"]
        accuracy = row["Accuracy"]
        misclassification = row["Misclassification"]

        summary_key = (dataset, depth, mode, threads)
        summary_entry = summary_keyed.setdefault(
            summary_key,
            {"time": [], "accuracy": [], "misclassification": []},
        )

        if time_s is not None:
            summary_entry["time"].append(time_s)
            valid_time_rows += 1
            mode_times.setdefault((dataset, depth, threads, mode), []).append(time_s)

            best_key = (dataset, depth)
            current_best = best_by_dataset_depth.get(best_key)
            if current_best is None or time_s < current_best["Time(s)"]:
                best_by_dataset_depth[best_key] = row

        if accuracy is not None:
            summary_entry["accuracy"].append(accuracy)
        if misclassification is not None:
            summary_entry["misclassification"].append(float(misclassification))

    summary_rows: list[list[str]] = []
    for (dataset, depth, mode, threads), metrics in sorted(summary_keyed.items()):
        summary_rows.append(
            [
                dataset,
                str(depth),
                mode,
                str(threads),
                mean_or_empty(metrics["time"]),
                mean_or_empty(metrics["accuracy"]),
                mean_or_empty(metrics["misclassification"]),
                str(len(metrics["time"])),
            ]
        )

    summary_path = outdir / "summary_by_dataset_depth.csv"
    write_csv(
        summary_path,
        [
            "Dataset",
            "Depth",
            "Mode",
            "Threads",
            "AvgTime(s)",
            "AvgAccuracy",
            "AvgMisclassification",
            "TimeSamples",
        ],
        summary_rows,
    )

    speedup_rows: list[list[str]] = []
    speedup_keys = {(d, dep, t) for (d, dep, t, _) in mode_times.keys()}
    for dataset, depth, threads in sorted(speedup_keys):
        cpu_times = mode_times.get((dataset, depth, threads, "CPU"), [])
        gpu_times = mode_times.get((dataset, depth, threads, "GPU"), [])
        cpu_avg = statistics.fmean(cpu_times) if cpu_times else None
        gpu_avg = statistics.fmean(gpu_times) if gpu_times else None
        if cpu_avg is not None and gpu_avg is not None and gpu_avg > 0:
            speedup = cpu_avg / gpu_avg
            speedup_str = f"{speedup:.6f}"
        else:
            speedup_str = ""

        speedup_rows.append(
            [
                dataset,
                str(depth),
                str(threads),
                f"{cpu_avg:.6f}" if cpu_avg is not None else "",
                f"{gpu_avg:.6f}" if gpu_avg is not None else "",
                speedup_str,
            ]
        )

    speedup_path = outdir / "speedup_cpu_vs_gpu.csv"
    write_csv(
        speedup_path,
        ["Dataset", "Depth", "Threads", "AvgCPUTime(s)", "AvgGPUTime(s)", "Speedup(CPU/GPU)"],
        speedup_rows,
    )

    best_rows: list[list[str]] = []
    for (dataset, depth), row in sorted(best_by_dataset_depth.items()):
        best_rows.append(
            [
                dataset,
                str(depth),
                row["Mode"],
                str(row["Threads"]),
                f"{row['Time(s)']:.6f}" if row["Time(s)"] is not None else "",
                f"{row['Accuracy']:.6f}" if row["Accuracy"] is not None else "",
                str(row["Misclassification"]) if row["Misclassification"] is not None else "",
            ]
        )

    best_path = outdir / "best_config_per_dataset_depth.csv"
    write_csv(
        best_path,
        ["Dataset", "Depth", "Mode", "Threads", "Time(s)", "Accuracy", "Misclassification"],
        best_rows,
    )

    print(f"Loaded {len(rows)} rows ({valid_time_rows} with valid time).")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {speedup_path}")
    print(f"Wrote: {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
