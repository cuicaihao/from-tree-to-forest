"""Generate the four fixed, synthetic datasets used throughout the tree-model post.

Each business case has a small teaching table for hand calculation and a larger
experimental dataset for software comparison. Use ``--check`` to compare
freshly generated CSV bytes with the published files. No synthetic coefficient
is an estimate of a real bank or retailer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
FRAUD_TEACHING_CSV = HERE / "case-a-fraud-teaching.csv"
FRAUD_CSV = HERE / "case-a-fraud.csv"
RETAIL_TEACHING_CSV = HERE / "case-b-retail-teaching.csv"
RETAIL_CSV = HERE / "case-b-retail.csv"

FRAUD_ANCHORS = [
    ("S1", 30.0, 2.0, "retail", "M00", 0),
    ("S2", 80.0, 5.0, "retail", "M01", 0),
    ("S3", 150.0, 10.0, "retail", "M02", 0),
    ("S4", 250.0, 20.0, "retail", "M03", 0),
    ("S5", 450.0, 8.0, "digital", "M04", 1),
    ("S6", 1200.0, 50.0, "digital", "M05", 1),
    ("S7", 3000.0, 120.0, "digital", "M06", 1),
    ("S8", 80.0, 300.0, "retail", "M07", 1),
]

RETAIL_ANCHORS = [
    ("D1", 0, 10, "none", 10),
    ("D2", 0, 15, "none", 14),
    ("D3", 10, 12, "member", 18),
    ("D4", 10, 20, "member", 26),
    ("D5", 20, 15, "campaign", 40),
    ("D6", 30, 35, "campaign", 60),
    ("D7", 40, 40, "campaign", 92),
    ("D8", 50, 60, "campaign", 140),
]


def generate_fraud_teaching() -> pd.DataFrame:
    """Return the eight transactions used only for hand calculation."""
    return pd.DataFrame(
        FRAUD_ANCHORS,
        columns=[
            "record_id",
            "amount",
            "distance",
            "merchant_type",
            "merchant_id",
            "y",
        ],
    )


def generate_fraud() -> pd.DataFrame:
    """Return 6,000 seeded, noisy transactions for model comparison."""
    rng = np.random.default_rng(42)
    n = 6000
    amount = rng.lognormal(4.7, 0.9, n)
    distance = rng.lognormal(3.2, 1.1, n)
    merchant_type = rng.choice(
        ["retail", "digital", "travel", "services"],
        n,
        p=[0.45, 0.22, 0.13, 0.20],
    )
    merchant_no = rng.integers(0, 80, n)
    merchant_effect = rng.normal(0, 0.7, 80)
    raw_score = (
        -5.1
        + 1.0 * np.log1p(amount / 100)
        + 1.4 * (distance > 150)
        + 0.9 * (merchant_type == "digital")
        + merchant_effect[merchant_no]
    )
    fraud = rng.binomial(1, expit(raw_score))
    generated = pd.DataFrame(
        {
            "record_id": [f"F{i:06d}" for i in range(1, n + 1)],
            "amount": amount,
            "distance": distance,
            "merchant_type": merchant_type,
            "merchant_id": [f"M{i:02d}" for i in merchant_no],
            "y": fraud,
        }
    )
    train_idx, hold_idx = train_test_split(
        generated.index, test_size=2400, stratify=generated.y, random_state=42
    )
    valid_idx, test_idx = train_test_split(
        hold_idx,
        test_size=1200,
        stratify=generated.loc[hold_idx, "y"],
        random_state=42,
    )
    generated["split"] = "train"
    generated.loc[valid_idx, "split"] = "valid"
    generated.loc[test_idx, "split"] = "test"
    assert generated.split.value_counts().to_dict() == {
        "train": 3600,
        "valid": 1200,
        "test": 1200,
    }
    assert generated.record_id.is_unique
    return generated


def generate_retail_teaching() -> pd.DataFrame:
    """Return the eight product-day records used only for hand calculation."""
    rows = []
    for sku, (record_id, discount, lag7, promo_type, sales) in enumerate(
        RETAIL_ANCHORS
    ):
        rows.append(
            (
                record_id,
                f"SKU{sku:02d}",
                "ST0",
                "2025-01-13",
                0,
                promo_type,
                discount,
                lag7,
                sales,
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "record_id",
            "sku_id",
            "store_id",
            "business_date",
            "weekday",
            "promo_type",
            "discount",
            "lag7",
            "y",
        ],
    )


def generate_retail() -> pd.DataFrame:
    """Return a 20-SKU, two-store, 120-day panel for model comparison."""
    rng = np.random.default_rng(42)
    sku_effect = rng.normal(0, 5, 20)
    store_effect = [-2.0, 3.0]
    rows = []
    first_date = pd.Timestamp("2025-01-06")  # Monday
    for day in range(120):
        for sku in range(20):
            for store in range(2):
                promo_type = rng.choice(
                    ["none", "member", "campaign"], p=[0.72, 0.20, 0.08]
                )
                discount = {"none": 0, "member": 10, "campaign": 25}[promo_type]
                mean = (
                    23
                    + sku_effect[sku]
                    + store_effect[store]
                    + 4 * (day % 7 in [5, 6])
                    + 0.65 * discount
                    + 2 * np.sin(2 * np.pi * day / 30)
                )
                sales = max(0, round(mean + rng.normal(0, 4)))
                rows.append(
                    (
                        f"R{day:03d}-{sku:02d}-{store}",
                        day,
                        (first_date + pd.Timedelta(days=day)).date().isoformat(),
                        f"SKU{sku:02d}",
                        f"ST{store}",
                        promo_type,
                        discount,
                        day % 7,
                        sales,
                    )
                )
    panel = pd.DataFrame(
        rows,
        columns=[
            "record_id",
            "day",
            "business_date",
            "sku_id",
            "store_id",
            "promo_type",
            "discount",
            "weekday",
            "y",
        ],
    )

    panel = panel.sort_values(["sku_id", "store_id", "day"])
    panel["lag7"] = panel.groupby(["sku_id", "store_id"]).y.transform(
        lambda sales: sales.shift(1).rolling(7, min_periods=7).mean()
    )
    panel = panel.sort_values(["day", "sku_id", "store_id"]).reset_index(drop=True)
    panel["split"] = np.select(
        [panel.day < 7, panel.day < 80, panel.day < 100],
        ["history", "train", "valid"],
        default="test",
    )
    assert panel.record_id.is_unique
    assert panel.split.value_counts().to_dict() == {
        "history": 280,
        "train": 2920,
        "valid": 800,
        "test": 800,
    }
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify all fixed CSVs")
    args = parser.parse_args()
    for path, dataset in (
        (FRAUD_TEACHING_CSV, generate_fraud_teaching()),
        (FRAUD_CSV, generate_fraud()),
        (RETAIL_TEACHING_CSV, generate_retail_teaching()),
        (RETAIL_CSV, generate_retail()),
    ):
        data = dataset.to_csv(index=False, float_format="%.12g")
        if args.check:
            assert path.read_text(encoding="utf-8") == data, f"Mismatch: {path}"
            print(f"Verified {path.name}: {len(dataset)} rows")
        else:
            path.write_text(data, encoding="utf-8")
            print(f"Wrote {path.name}: {len(dataset)} rows")


if __name__ == "__main__":
    main()
