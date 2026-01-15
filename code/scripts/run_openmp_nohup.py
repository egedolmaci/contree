#!/usr/bin/env python3
import subprocess
import csv
import os
import re

DATASETS_DIR = "../datasets"
BINARY = "./build/ConTree"
DEPTHS = [2, 3, 4]
THREADS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,60]

MIS_RE = re.compile(r"Misclassification score:\s*([0-9]+)")
ACC_RE = re.compile(r"Accuracy:\s*([0-9.]+)")
TIME_RE = re.compile(r"Average time taken.*?:\s*([0-9.]+)\s*seconds")

def run(dataset, depth, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"

    cmd = [BINARY, "-file", f"{DATASETS_DIR}/{dataset}.txt", "-max-depth", str(depth)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)

    out = p.stdout + p.stderr

    mis = MIS_RE.search(out)
    acc = ACC_RE.search(out)
    tme = TIME_RE.search(out)

    if not (mis and acc and tme):
        print(out)
        raise RuntimeError(f"Parse failed: {dataset}, depth={depth}, threads={threads}")

    return float(tme.group(1)), float(acc.group(1)), int(mis.group(1))

def main():
    print("SCRIPT STARTED", flush=True)
    datasets = [f.replace(".txt","") for f in os.listdir(DATASETS_DIR) if f.endswith(".txt")]

    with open("final_openmp_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","depth","threads","time_sec","accuracy","misclassification"])

        for d in DEPTHS:
            for ds in datasets:
                for t in THREADS:
                    print(f"RUN: dataset={ds} depth={d} threads={t}", flush=True)
                    tsec, acc, mis = run(ds, d, t)
                    w.writerow([ds, d, t, tsec, acc, mis])
                    f.flush()

if __name__ == "__main__":
    main()
