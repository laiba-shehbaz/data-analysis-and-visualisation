#!/usr/bin/env python3
"""
Creates data/sales_raw.csv - a SELF-CREATED dataset of 1,200 online-store orders
for the fictional shop "ShopEase" (year 2025, Pakistan).

The data is generated with a fixed random seed, so it is reproducible.
It is deliberately messy (missing values, duplicates, inconsistent spellings,
mixed date formats, prices stored as text, impossible values) so that the
cleaning step in analysis.py has real work to do.

Run:  python generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 1200
rng = np.random.default_rng(SEED)

CATALOG = {
    "Electronics": {"Wireless Earbuds": 4500, "Smart Watch": 9500, "Power Bank": 3200,
                    "Bluetooth Speaker": 5200, "Phone Case": 800},
    "Fashion": {"T-Shirt": 1500, "Lawn Suit": 4800, "Sneakers": 6500,
                "Handbag": 3900, "Hoodie": 3200},
    "Home & Kitchen": {"Air Fryer": 14500, "Bedsheet Set": 4200, "Dinner Set": 7800,
                       "Blender": 6200, "Storage Box": 1200},
    "Beauty": {"Face Wash": 850, "Perfume": 5500, "Hair Dryer": 4300,
               "Skincare Set": 6800, "Lipstick": 1400},
    "Books": {"Novel": 1100, "Urdu Poetry": 900, "Programming Book": 2800, "Kids Story Set": 1800},
    "Sports": {"Cricket Bat": 7500, "Yoga Mat": 2200, "Football": 2600, "Dumbbell Set": 6000},
}
CAT_W = {"Fashion": .26, "Electronics": .18, "Home & Kitchen": .18,
         "Beauty": .16, "Books": .12, "Sports": .10}
CITY_W = {"Karachi": .30, "Lahore": .27, "Islamabad": .15, "Faisalabad": .10, "Multan": .09, "Peshawar": .09}
PAY_W = {"Cash on Delivery": .42, "Credit/Debit Card": .20, "JazzCash": .15, "Easypaisa": .15, "Bank Transfer": .08}
# Relative number of orders per month (Ramadan sales in March, Independence Day in August, 11.11 / Black Friday in November)
MONTH_W = [.8, .8, 1.1, 1.0, .9, .9, .85, 1.0, .9, 1.0, 1.7, 1.3]
RATING_BASE = {"Electronics": 3.9, "Fashion": 4.1, "Home & Kitchen": 4.1,
               "Beauty": 4.3, "Books": 4.4, "Sports": 4.2}
CITY_RATING_ADJ = {"Peshawar": -.3, "Multan": -.25, "Faisalabad": -.05}


def pick(weights: dict, size: int):
    keys = list(weights)
    p = np.array(list(weights.values()), dtype=float)
    return rng.choice(keys, size=size, p=p / p.sum())


def build_clean_orders() -> pd.DataFrame:
    months = rng.choice(np.arange(1, 13), size=N_ORDERS, p=np.array(MONTH_W) / sum(MONTH_W))
    days = np.array([rng.integers(1, 29) for _ in months])
    dates = pd.to_datetime({"year": 2025, "month": months, "day": days})
    categories = pick(CAT_W, N_ORDERS)
    products, prices = [], []
    for cat in categories:
        name = rng.choice(list(CATALOG[cat]))
        products.append(name)
        prices.append(int(round(CATALOG[cat][name] * rng.uniform(.9, 1.1), -1)))
    cities = pick(CITY_W, N_ORDERS)

    # Discounts are deeper in November / December
    discount = []
    for m in months:
        if m == 11:
            discount.append(rng.choice([0, 10, 20, 30], p=[.15, .25, .30, .30]))
        elif m == 12:
            discount.append(rng.choice([0, 10, 20, 30], p=[.30, .30, .25, .15]))
        else:
            discount.append(rng.choice([0, 5, 10, 15, 20], p=[.45, .15, .2, .1, .1]))
    discount = np.array(discount)

    # Quantity: cheap items are bought in larger numbers, and discounts push people to add one more item
    quantity = []
    for cat, price, disc in zip(categories, prices, discount):
        base = 1 if price > 6000 else int(rng.choice([1, 2, 3], p=[.62, .28, .10]))
        if rng.random() < disc / 100 * 1.5:
            base += 1
        quantity.append(base)

    rating = []
    for cat, city in zip(categories, cities):
        mean = RATING_BASE[cat] + CITY_RATING_ADJ.get(city, 0)
        rating.append(int(np.clip(round(rng.normal(mean, .8)), 1, 5)))

    return pd.DataFrame({
        "order_id": [f"ORD-2025-{i:04d}" for i in range(1, N_ORDERS + 1)],
        "order_date": dates,
        "customer_id": [f"C{rng.integers(1, 701):04d}" for _ in range(N_ORDERS)],
        "city": cities, "category": categories, "product": products,
        "quantity": quantity, "unit_price": prices, "discount_pct": discount,
        "payment_method": pick(PAY_W, N_ORDERS), "rating": rating,
    })


def make_messy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    def idx(frac=None, count=None):
        return rng.choice(n, size=count or int(n * frac), replace=False)

    # 1) mixed date formats: 30 % of dates written as DD/MM/YYYY
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    for i in idx(.30):
        y, m, d = df.loc[i, "order_date"].split("-")
        df.loc[i, "order_date"] = f"{d}/{m}/{y}"

    # 2) prices stored as text such as "Rs 4,500"
    df["unit_price"] = df["unit_price"].astype(object)
    for i in idx(.08):
        df.loc[i, "unit_price"] = f"Rs {int(df.loc[i, 'unit_price']):,}"

    # 3) inconsistent spellings, capital letters and spaces
    city_noise = {"Lahore": ["lahore", "LAHORE", "Lahor", " Lahore "], "Karachi": ["karachi", "KARACHI", "Karachi "],
                  "Islamabad": ["islamabad", "Islamabd", " Islamabad"], "Faisalabad": ["Faislabad", "faisalabad"],
                  "Multan": ["multan", "MULTAN"], "Peshawar": ["peshawar", "Peshawer"]}
    for i in idx(.12):
        df.loc[i, "city"] = rng.choice(city_noise[df.loc[i, "city"]])
    cat_noise = {"Electronics": ["electronics", "ELECTRONICS"], "Fashion": ["fashion", "Fashion "],
                 "Home & Kitchen": ["Home and Kitchen", "home & kitchen"], "Beauty": ["beauty", "BEAUTY"],
                 "Books": ["books", "Books "], "Sports": ["sports", "SPORTS"]}
    for i in idx(.10):
        df.loc[i, "category"] = rng.choice(cat_noise[df.loc[i, "category"]])

    # 4) impossible values
    for i in idx(count=6):
        df.loc[i, "quantity"] = int(rng.choice([0, -1]))
    for i in idx(count=5):
        df.loc[i, "rating"] = int(rng.choice([0, 9, 10]))

    # 5) missing values
    for col, frac in [("rating", .08), ("city", .02), ("payment_method", .015),
                      ("discount_pct", .03), ("unit_price", .01)]:
        df[col] = df[col].astype(object)
        df.loc[idx(frac), col] = np.nan

    # 6) duplicate rows (same order exported twice)
    duplicates = df.sample(n=35, random_state=SEED)
    df = pd.concat([df, duplicates], ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    raw = make_messy(build_clean_orders())
    raw.to_csv(out / "sales_raw.csv", index=False)
    print(f"Saved {len(raw)} rows to {out / 'sales_raw.csv'}")
