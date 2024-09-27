import time
from pathlib import Path

import pandas as pd
import polars as pl
import pyspark.sql.functions as f
from pyspark import SparkContext
from pyspark.sql import SparkSession

data_dir = Path("data")
parquet_files = list(data_dir.glob("*.parquet"))


# Pandas benchmark
def pandas_benchmark() -> float:
    start_time = time.time()
    print("Starting pandas")
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in parquet_files)
    # Perform operations
    result = (
        df.groupby("customer_id")
        .agg({"order_id": "count", "quantity": "sum", "price": "mean"})
        .reset_index()
    )
    result = result.sort_values("order_id", ascending=False).head(10)

    end_time = time.time()
    print("Finished Pandas")
    return end_time - start_time


# Polars benchmark
def polars_benchmark() -> float:
    start_time = time.time()
    print("Starting polars")
    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(2_000_000)
        df = pl.scan_parquet(parquet_files)
        # Perform operations
        result = (
            df.group_by("customer_id")
            .agg([pl.count("order_id"), pl.sum("quantity"), pl.mean("price")])
            .sort("order_id", descending=True)
            .limit(10)
            .collect(streaming=True)
        )
        end_time = time.time()
        print("Finished Polars")
        return end_time - start_time


# PySpark benchmark
def pyspark_benchmark() -> float:
    print("Starting pyspark")
    start_time = time.time()

    SparkContext.setSystemProperty("spark.executor.memory", "20g")
    SparkContext.setSystemProperty("spark.executor.cores", "12")
    SparkContext.setSystemProperty("spark.driver.cores", "10gb")
    spark = SparkSession.builder.appName("PySpark Benchmark").getOrCreate()
    # df = spark.createDataFrame(pd.DataFrame(data))
    df = spark.read.parquet(*[x.as_posix() for x in parquet_files])

    # Perform operations
    result = (
        df.groupBy("customer_id")
        .agg(
            f.sum("order_id").alias("order_count"),
            f.sum("quantity"),
            f.sum("price") / f.sum("quantity"),
        )
        .orderBy(f.col("order_count").desc())
        .limit(10)
    )

    result.collect()  # Force computation

    spark.stop()
    end_time = time.time()
    print("Finished PySpark")
    return end_time - start_time


def time_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


# Run benchmarks
# pandas_time = pandas_benchmark()
# polars_time = polars_benchmark()
pyspark_time = pyspark_benchmark()

# print(f"Pandas execution time: {pandas_time:.4f} seconds")
# print(f"Polars execution time: {polars_time:.4f} seconds")
# print(f"PySpark execution time: {pyspark_time:.4f} seconds")
