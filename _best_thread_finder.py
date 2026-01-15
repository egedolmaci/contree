import pandas as pd

# Load CSV
df = pd.read_csv("openmp.csv")

# Ensure numeric types
df["Threads"] = df["Threads"].astype(int)
df["Time(s)"] = df["Time(s)"].astype(float)

results = []

# Group by Dataset and Depth
for (dataset, depth), group in df.groupby(["Dataset", "Depth"]):
    
    # 1-thread time
    one_thread_row = group[group["Threads"] == 1]
    if one_thread_row.empty:
        continue  # skip if no 1-thread run
    
    one_thread_time = one_thread_row["Time(s)"].iloc[0]
    
    # Best time (minimum)
    best_row = group.loc[group["Time(s)"].idxmin()]
    best_time = best_row["Time(s)"]
    best_threads = best_row["Threads"]
    
    results.append({
        "Dataset": dataset,
        "Depth": depth,
        "1_thread_time": one_thread_time,
        "best_time": best_time,
        "best_threads": best_threads,
        "speedup": one_thread_time / best_time
    })

# Create result dataframe
result_df = pd.DataFrame(results)

# Sort for readability
result_df = result_df.sort_values(["Dataset", "Depth"])

# Print
print(result_df.to_string(index=False))

# Optional: save to CSV
result_df.to_csv("best_thread_summary.csv", index=False)