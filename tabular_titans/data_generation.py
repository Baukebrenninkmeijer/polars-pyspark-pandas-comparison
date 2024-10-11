from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from tqdm import trange

data_location = Path(__file__).parent.parent / "data"
if not data_location.exists():
    data_location.makedirs(exist_ok=True)


# Function to generate fake e-commerce data
def generate_fake_data(num_rows) -> dict[str, Any]:
    np.random.seed(42)
    data = {
        "order_id": np.arange(1, num_rows + 1),
        "customer_id": np.random.randint(1, 100000, num_rows),
        "product_id": np.random.randint(1, 10000, num_rows),
        "quantity": np.random.randint(1, 10, num_rows),
        "price": np.random.uniform(10, 1000, num_rows).round(2),
        "order_date": random_dates(
            pd.to_datetime("2022-1-1"), pd.to_datetime("2025-1-1"), num_rows
        ),
    }
    return data


def save_fake_data_n_times(n, num_rows):
    for i in trange(n):
        data = generate_fake_data(num_rows)
        df = pl.DataFrame(data)
        df.write_parquet(data_location / f"chunk_{i}.parquet")
        print(f"Saved chunk {i}")


def random_dates(start, end, n=10):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pl.from_epoch(
        pl.Series(np.random.randint(start_u, end_u, n)), time_unit="ms"
    )


# Generate data
num_rows = 10_000_000
save_fake_data_n_times(10, num_rows)
