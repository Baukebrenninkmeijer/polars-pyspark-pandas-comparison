import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("Script started")

# Remove seaborn-specific styling
plt.style.use("default")

# Read the parquet file
parquet_file = "results/results_polars_gpu_100M.parquet"
print(f"Attempting to read parquet file: {parquet_file}")

if not Path(parquet_file).exists():
    print(f"Error: Parquet file not found at {parquet_file}")
    exit(1)

try:
    df = pl.read_parquet(parquet_file)
    print("Parquet file read successfully")
except Exception as e:
    print(f"Error reading parquet file: {e}")
    exit(1)

# Convert duration to milliseconds for better readability
df = df.with_columns(pl.col("duration") * 1000)

print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns)
print("\nData types:")
print(df.dtypes)

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Performance analysis

try:
    # 1. GPU vs CPU performance comparison
    print("\nCalculating GPU vs CPU performance comparison")
    gpu_vs_cpu = df.group_by(["func", "gpu", "lazy"]).agg(
        pl.col("duration").mean().alias("mean_duration"),
        pl.col("duration").std().alias("std_duration"),
    )

    print("\nGPU vs CPU performance comparison:")
    print(gpu_vs_cpu)

    # Visualize GPU vs CPU performance
    print("Creating GPU vs CPU performance plot")
    plt.figure(figsize=(12, 6))
    gpu_vs_cpu_pd = gpu_vs_cpu.to_pandas()
    plt.bar(
        gpu_vs_cpu_pd["func"] + " " + gpu_vs_cpu_pd["gpu"].astype(str),
        gpu_vs_cpu_pd["mean_duration"],
    )
    plt.title("GPU vs CPU Performance Comparison")
    plt.xlabel("Function and GPU usage")
    plt.ylabel("Mean Duration (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/gpu_vs_cpu_performance.png")
    print("GPU vs CPU performance plot saved")

    # 2. Effect of lazy execution
    print("\nCalculating effect of lazy execution")
    lazy_effect = df.group_by(["func", "lazy"]).agg(
        pl.col("duration").mean().alias("mean_duration"),
        pl.col("duration").std().alias("std_duration"),
    )

    print("\nEffect of lazy execution:")
    print(lazy_effect)

    # Visualize lazy execution effect
    print("Creating lazy execution effect plot")
    plt.figure(figsize=(12, 6))
    lazy_effect_pd = lazy_effect.to_pandas()
    plt.bar(
        lazy_effect_pd["func"] + " " + lazy_effect_pd["lazy"].astype(str),
        lazy_effect_pd["mean_duration"],
    )
    plt.title("Effect of Lazy Execution on Performance")
    plt.xlabel("Function and Lazy Execution")
    plt.ylabel("Mean Duration (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/lazy_execution_effect.png")
    print("Lazy execution effect plot saved")

    # 3. Impact of streaming
    print("\nCalculating impact of streaming")
    streaming_effect = df.group_by(["func", "streaming"]).agg(
        pl.col("duration").mean().alias("mean_duration"),
        pl.col("duration").std().alias("std_duration"),
    )

    print("\nImpact of streaming:")
    print(streaming_effect)

    # Visualize streaming effect
    print("Creating streaming effect plot")
    plt.figure(figsize=(12, 6))
    streaming_effect_pd = streaming_effect.to_pandas()
    plt.bar(
        streaming_effect_pd["func"]
        + " "
        + streaming_effect_pd["streaming"].astype(str),
        streaming_effect_pd["mean_duration"],
    )
    plt.title("Effect of Streaming on Performance")
    plt.xlabel("Function and Streaming")
    plt.ylabel("Mean Duration (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/streaming_effect.png")
    print("Streaming effect plot saved")

    # 4. Performance distribution
    print("\nCreating performance distribution plot")
    plt.figure(figsize=(12, 6))
    for func in df["func"].unique():
        plt.hist(
            df.filter(pl.col("func") == func)["duration"],
            bins=30,
            alpha=0.5,
            label=func,
        )
    plt.title("Performance Distribution by Function")
    plt.xlabel("Duration (ms)")
    plt.ylabel("Count")
    plt.legend(title="Function")
    plt.tight_layout()
    plt.savefig("results/performance_distribution.png")
    print("Performance distribution plot saved")

    # 5. Correlation analysis
    print("\nPerforming correlation analysis")
    correlation_matrix = df.select(
        pl.col(["duration", "limit", "gpu", "lazy", "streaming"])
    ).corr()
    print("\nCorrelation matrix:")
    print(correlation_matrix)

    # Visualize correlation matrix
    print("Creating correlation matrix plot")
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix.to_pandas(), cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Correlation Matrix of Performance Factors")
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig("results/correlation_matrix.png")
    print("Correlation matrix plot saved")

    # 6. Performance by limit
    print("\nCalculating performance by limit")
    limit_performance = df.group_by("limit").agg(
        pl.col("duration").mean().alias("mean_duration"),
        pl.col("duration").std().alias("std_duration"),
    )

    print("\nPerformance by limit:")
    print(limit_performance)

    # Visualize performance by limit
    print("Creating performance by limit plot")
    plt.figure(figsize=(12, 6))
    for func in df["func"].unique():
        func_data = df.filter(pl.col("func") == func)
        plt.plot(func_data["limit"], func_data["duration"], label=func)
    plt.title("Performance by Limit")
    plt.xlabel("Limit")
    plt.ylabel("Duration (ms)")
    plt.legend(title="Function")
    plt.tight_layout()
    plt.savefig("results/performance_by_limit.png")
    print("Performance by limit plot saved")

    # Save insights to a text file
    print("\nSaving analysis insights")
    with open("results/analysis_insights.txt", "w") as f:
        f.write("Analysis Insights for 100M Dataset\n")
        f.write("==================================\n\n")

        f.write("1. GPU vs CPU Performance:\n")
        f.write("   - [Insert observations about GPU vs CPU performance]\n\n")

        f.write("2. Effect of Lazy Execution:\n")
        f.write("   - [Insert observations about the impact of lazy execution]\n\n")

        f.write("3. Impact of Streaming:\n")
        f.write("   - [Insert observations about the effect of streaming]\n\n")

        f.write("4. Performance Distribution:\n")
        f.write(
            "   - [Insert observations about the performance distribution across functions]\n\n"
        )

        f.write("5. Correlation Analysis:\n")
        f.write("   - [Insert insights from the correlation matrix]\n\n")

        f.write("6. Performance by Limit:\n")
        f.write(
            "   - [Insert observations about how performance changes with different limits]\n\n"
        )

        f.write("7. Overall Performance Considerations:\n")
        f.write(
            "   - [Insert general conclusions and recommendations based on the analysis]\n"
        )

    print("Analysis complete. Results and plots saved in the 'results' directory.")

except Exception as e:
    print(f"An error occurred during analysis: {e}")
