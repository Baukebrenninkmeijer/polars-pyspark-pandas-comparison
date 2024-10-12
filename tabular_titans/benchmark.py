import logging
import os
import threading
from itertools import product
from pathlib import Path
from time import sleep

import fire
import pandas as pd
import polars as pl
import psutil
import pyspark.sql.functions as f

# from multiprocessing.pool import Pool

# from multiprocess import Process, Queue
from multiprocess.pool import Pool
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year
from rich.logging import RichHandler
from tqdm.auto import tqdm

from tabular_titans.utils import time_func

logger = logging.getLogger("tabular_titans")
logger.setLevel(logging.DEBUG)
if not len(logger.handlers):
    handler = RichHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)


data_dir = Path(__file__).parent.parent / "data"
if Path(__file__).parents[2].name == 'content':
    results_dir = Path(__file__).parents[2] / "drive/MyDrive"
else:
    results_dir = Path(__file__).parents[1] / "data"

parquet_files = [x.as_posix() for x in data_dir.glob("*.parquet")]
logger.debug(f"{len(parquet_files)} files loaded.")
JOIN_SIZE = 1_000_000
BASE_SIZE = 1_000_000


def read_pandas(parquet_files: list[str] = parquet_files) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(parquet_file) for parquet_file in parquet_files])
    return df

def pl_cast_strings(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("customer_id").cast(pl.String), pl.col("product_id").cast(pl.String)
    )


def read_polars_lazy(parquet_files: list[str] = parquet_files, preload: bool = False, limit: int | None = None) -> pl.LazyFrame:
    df = pl.scan_parquet(parquet_files, n_rows=limit).pipe(pl_cast_strings)
    if preload:
        df = df.collect().lazy()
    return df


def read_polars(parquet_files: list[str] = parquet_files, limit: int=0) -> pl.DataFrame:
    if limit:
        return pl.read_parquet(parquet_files, n_rows=limit).pipe(pl_cast_strings).limit(limit)
    return pl.read_parquet(parquet_files).pipe(pl_cast_strings)


def read_pyspark(
    spark: SparkSession, parquet_files: list[str] = parquet_files
) -> SparkDataFrame:
    df = spark.read.parquet(*[x for x in parquet_files])
    return df


################################
# Pandas operations
################################
def pandas_groupby(df: pd.DataFrame):
    # Perform operations
    result = (
        df.groupby("customer_id")
        .agg({"order_id": "count", "quantity": "sum", "price": "mean"})
        .reset_index()
    )
    return result


def pandas_join(df: pd.DataFrame) -> pd.DataFrame:
    df = df.head(JOIN_SIZE)
    return pd.merge(df, df, on="customer_id", how="left")


def pandas_filter(df: pd.DataFrame):
    return df[
        (df["quantity"] > 5) & (df["price"] < 500) & (df["order_date"].dt.year == 2023)
    ]


def pandas_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["order_date", "price"], ascending=[True, False])


################################
# Polars benchmark
################################
def polars_groupby(
    df: pl.LazyFrame, streaming: bool = False, gpu: bool = False
) -> None:
    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(2_000_000)
        result = (
            df.group_by("customer_id")
            .agg([pl.count("order_id"), pl.sum("quantity"), pl.mean("price")])
            .sort("order_id", descending=True)
            .limit(10)
        )
        if isinstance(result, pl.LazyFrame):
            result = result.collect(streaming=streaming, engine="gpu" if gpu else "cpu")


def polars_join(
    df: pl.LazyFrame | pl.DataFrame,
    streaming: bool = False,
    gpu: bool = False,
    n: int = JOIN_SIZE,
) -> pl.DataFrame:
    df = df.limit(n)
    result = df.join(df, on="product_id", how="left")
    if isinstance(result, pl.LazyFrame):
        result = result.collect(streaming=streaming, engine="gpu" if gpu else "cpu")
    return result


def polars_filter(
    df: pl.LazyFrame | pl.DataFrame, streaming: bool = False, gpu: bool = False
) -> pl.DataFrame:
    result = df.filter(
        (pl.col("quantity") > 5)
        & (pl.col("price") < 500)
        & (pl.col("order_date").dt.year() == 2023)
    )
    if isinstance(result, pl.LazyFrame):
        result = result.collect(streaming=streaming, engine="gpu" if gpu else "cpu")
    return result


def polars_sort(
    df: pl.LazyFrame | pl.DataFrame, streaming: bool = False, gpu: bool = False
) -> pl.DataFrame:
    result = df.sort(["order_date", "price"], descending=[False, True])
    if isinstance(result, pl.LazyFrame):
        result = result.collect(streaming=streaming, engine="gpu" if gpu else "cpu")
    return result


################################
# PySpark benchmark
################################
def pyspark_groupby(df: SparkDataFrame) -> None:
    result = df.groupBy("customer_id").agg(
        f.sum("order_id").alias("order_count"),
        f.sum("quantity"),
        f.sum("price") / f.sum("quantity"),
    )
    result.collect()


def pyspark_join(df: SparkDataFrame):
    df = df.limit(JOIN_SIZE)
    return df.join(df, on="product_id", how="left").collect()


def pyspark_filter(df: SparkDataFrame):
    return df.filter(
        (col("quantity") > 5) & (col("price") < 500) & (year(col("order_date")) == 2023)
    ).collect()


def pyspark_sort(df: SparkDataFrame):
    df.orderBy(["order_date", "price"], ascending=[True, False]).collect()


def benchmark_polars(do_gpu: bool = True) -> pd.DataFrame:
    logger.info("Starting polars benchmark.")
    polars_functions = [polars_filter, polars_sort, polars_groupby, polars_join]
    results = []
    combs = list(product(polars_functions, [True, False], [True, False], [True, False], [True, False], range(14, 100, 20)))
    for func, preload, gpu, streaming, lazy, data_inc in tqdm(combs):
        try:
            limit = data_inc * BASE_SIZE
            data = read_polars_lazy(limit=limit, preload=preload) if lazy else read_polars(limit=limit)
            data = data
            if not do_gpu and gpu:  # Currently unsupported
                continue
            if gpu and streaming:
                continue

            config = {
                    "func": func.__name__,
                    "gpu": gpu,
                    "streaming": streaming,
                    "lazy": lazy,
                    "limit": limit,
                    "preload": preload,
                }
            # if not lazy and streaming:
            #     continue
            logger.debug(f"Running: {config}")
            # duration = pool.apply(
            #     func=time_func(func),
            #     kwds=dict(df=data, gpu=gpu, streaming=streaming),
            # )
            duration = time_func(func)(df=data, gpu=gpu, streaming=streaming)

            results.append(
                config.update({"duration": duration})
            )
            results_df = pl.DataFrame(results)
            results_df.write_parquet(results_dir / "results_polars.parquet")
        except Exception as e:
            logger.error(e)
    return results_df


def benchmark_pyspark():
    spark: SparkSession = (
        SparkSession.builder.appName("PySpark Benchmark")
        .master("local[*]")
        # .config("spark.executor.cores", "10")
        .config("spark.executor.memory", "25g")
        .config("spark.driver.memory", "20g")
        .config("spark.driver.maxResultSize", "0")
        .getOrCreate()
    )

    logger.info("Starting pyspark benchmark.")
    pyspark_functions = [pyspark_filter, pyspark_sort, pyspark_groupby, pyspark_join]
    results = []
    for func in tqdm(pyspark_functions):
        data = read_pyspark(spark=spark)
        duration = time_func(func)(**dict(df=data))
        results.append({"func": func.__name__, "duration": duration})
        results_df = pd.DataFrame(results)
        results_df.to_parquet("results_pyspark.parquet")
    return results_df


def benchmark_pandas() -> pd.DataFrame:
    logger.info("Starting pandas benchmark.")
    pandas_functions = [pandas_filter, pandas_sort, pyspark_filter, pandas_join]
    results = []
    with Pool(processes=1) as pool:
        for func in tqdm(pandas_functions):
            data = read_pandas()
            duration = pool.apply(func=time_func(func), kwds=dict(df=data))
            results.append({"func": func.__name__, "duration": duration})
            results_df = pd.DataFrame(results)
            results_df.to_parquet("results_pandas.parquet")
    return results_df


def checkMemory(amount):
    while True:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1e9  # in bytes
        # logger.debug(process.memory_info())
        # logger.debug(process.memory_percent())
        total_used_memory = psutil.virtual_memory().used / 1e9
        sleep(2)
        # logger.debug(f'Process Usage: {memory_usage}, Total usage: {total_used_memory}')
        if memory_usage > amount or (total_used_memory > amount + 5):
            logger.error("Memory budget exceeded")
            break
    os._exit(0)


if __name__ == "__main__":
    fire.Fire()

