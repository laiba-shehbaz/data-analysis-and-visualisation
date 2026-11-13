#!/usr/bin/env python3
"""
ShopEase 2025 - Data Analysis and Visualisation
===============================================
Steps
    1. Load the raw CSV with pandas and inspect columns / data types
    2. Clean missing, duplicate and inconsistent values (every decision is logged)
    3. Calculate summary statistics and grouped results
    4. Draw six charts that answer clear questions
    5. Write a short findings report

Run:  python analysis.py
Needs: pandas, numpy, matplotlib   (pip install -r requirements.txt)
"""
from __future__ import annotations

import calendar
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW_FILE = BASE / "data" / "sales_raw.csv"
CLEAN_FILE = BASE / "data" / "sales_clean.csv"
OUT_DIR = BASE / "outputs"
TABLE_DIR = OUT_DIR / "tables"
CHART_DIR = BASE / "charts"

INDIGO, TEAL, AMBER, ROSE, SLATE, LIGHT = "#4f46e5", "#0d9488", "#f59e0b", "#e11d48", "#475569", "#c7d2fe"
PALETTE = [INDIGO, TEAL, AMBER, ROSE, "#7c3aed", "#0891b2"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfbff", "axes.edgecolor": "#cbd5e1",
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
})


def money(value: float) -> str:
    """Short currency text: 1_250_000 -> 'Rs 1.25M'."""
    if abs(value) >= 1_000_000:
        return f"Rs {value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"Rs {value / 1_000:.0f}K"
    return f"Rs {value:,.0f}"


def rs(value: float) -> str:
    """Full-precision currency text: 5144.69 -> 'Rs 5,145'."""
    return f"Rs {value:,.0f}"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ───────────────────────────── 1. LOAD + INSPECT ─────────────────────────────
def inspect(raw: pd.DataFrame) -> str:
    buffer = io.StringIO()
    buffer.write(f"Shape: {raw.shape[0]} rows x {raw.shape[1]} columns\n\n")
    raw.info(buf=buffer)
    buffer.write("\nMissing values per column:\n" + raw.isna().sum().to_string())
    buffer.write(f"\n\nExact duplicate rows: {raw.duplicated().sum()}")
    buffer.write("\n\nFirst 5 rows:\n" + raw.head().to_string())
    buffer.write("\n\nUnique spellings in text columns:\n")
    for column in ["city", "category", "payment_method"]:
        buffer.write(f"  {column} ({raw[column].nunique()}): {sorted(raw[column].dropna().unique())}\n")
    buffer.write("\nNumeric summary (before cleaning):\n" + raw.describe().round(2).to_string())
    return buffer.getvalue()


# ───────────────────────────── 2. CLEANING ───────────────────────────────────
CITY_FIX = {"lahor": "Lahore", "islamabd": "Islamabad", "faislabad": "Faisalabad", "peshawer": "Peshawar"}


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    log: list[dict] = []

    def note(issue: str, rows: int, decision: str) -> None:
        log.append({"issue": issue, "rows_affected": int(rows), "decision": decision})

    # 1. duplicates --------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    assert df["order_id"].is_unique, "order_id should be unique after removing exact duplicates"
    note("Duplicate rows (same order exported twice)", before - len(df),
         "Removed. An order can only be counted once, so keeping copies would inflate revenue.")

    # 2. inconsistent text -------------------------------------------------
    original = df["city"].str.strip()
    df["city"] = df["city"].str.strip().str.lower().replace(CITY_FIX).str.title()
    note("City: different capital letters, extra spaces and typos (Lahor, Islamabd...)",
         (original.fillna("") != df["city"].fillna("")).sum(),
         "Trimmed, lower-cased, fixed known typos, then title-cased: 22 spellings became 6 cities.")

    original = df["category"].str.strip()
    df["category"] = (df["category"].str.strip().str.lower()
                      .str.replace(" and ", " & ", regex=False).str.title())
    note("Category: capital letters, spaces and 'Home and Kitchen' vs 'Home & Kitchen'",
         (original.fillna("") != df["category"].fillna("")).sum(),
         "Same normalisation: 18 spellings became 6 categories.")

    # 3. dates in two formats ---------------------------------------------
    text = df["order_date"].str.strip()
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    df["order_date"] = iso.fillna(dmy)
    assert df["order_date"].notna().all(), "unparsed dates remain"
    note("Dates written in two formats (YYYY-MM-DD and DD/MM/YYYY)", dmy.notna().sum(),
         "Parsed each format separately into one real datetime column (slash dates are day-first).")

    # 4. prices stored as text ---------------------------------------------
    as_text = df["unit_price"].str.contains("Rs", na=False)
    df["unit_price"] = pd.to_numeric(df["unit_price"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    note("Unit price stored as text such as 'Rs 4,500'", as_text.sum(),
         "Removed 'Rs' and commas and converted the column to numbers.")

    missing_price = df["unit_price"].isna()
    df["unit_price"] = df["unit_price"].fillna(df.groupby("product")["unit_price"].transform("median"))
    note("Unit price missing", missing_price.sum(),
         "Filled with the median price of the same product (a product's price barely changes).")

    # 5. impossible values -------------------------------------------------
    bad_quantity = df["quantity"] <= 0
    df = df.loc[~bad_quantity].reset_index(drop=True)
    note("Quantity of 0 or negative (impossible for a completed order)", bad_quantity.sum(),
         "Rows dropped. The true quantity is unknown and guessing would distort revenue.")

    rating_missing = int(df["rating"].isna().sum())
    bad_rating = df["rating"].notna() & ~df["rating"].between(1, 5)
    df.loc[bad_rating, "rating"] = np.nan
    note("Rating outside the 1-5 scale (0, 9, 10)", bad_rating.sum(),
         "Set to missing instead of guessing a value.")

    # 6. missing values ----------------------------------------------------
    df["month"] = df["order_date"].dt.month
    missing_discount = df["discount_pct"].isna()
    df["discount_pct"] = df["discount_pct"].fillna(df.groupby("month")["discount_pct"].transform("median"))
    note("Discount missing", missing_discount.sum(),
         "Filled with the median discount of the same month (discounts follow the season).")

    note("Rating missing (customer did not review)", rating_missing,
         "Left empty. A rating is an opinion and cannot be invented; rating charts use only real reviews.")

    for column in ["city", "payment_method"]:
        missing = df[column].isna()
        df[column] = df[column].fillna("Unknown")
        note(f"{column.replace('_', ' ').capitalize()} missing", missing.sum(),
             "Labelled 'Unknown' so the sale still counts in revenue totals but is left out of comparisons.")

    # 7. new columns ---------------------------------------------------------
    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)).round(2)
    df["month_name"] = df["month"].apply(lambda m: calendar.month_abbr[m])
    df["quantity"] = df["quantity"].astype(int)
    df = df.sort_values("order_date").reset_index(drop=True)
    return df, pd.DataFrame(log)


# ───────────────────────────── 3. STATISTICS ─────────────────────────────────
def compute_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    s: dict = {}
    s["overview"] = pd.Series({
        "Orders": len(df), "Units sold": int(df["quantity"].sum()),
        "Total revenue (Rs)": df["revenue"].sum(), "Average order value (Rs)": df["revenue"].mean(),
        "Median order value (Rs)": df["revenue"].median(), "Average rating": df["rating"].mean(),
        "Unique customers": df["customer_id"].nunique(),
    }).round(2)

    s["monthly"] = (df.groupby("month").agg(orders=("order_id", "count"), revenue=("revenue", "sum"))
                    .reindex(range(1, 13)).round(2))
    s["category"] = (df.groupby("category").agg(orders=("order_id", "count"), units=("quantity", "sum"),
                                                revenue=("revenue", "sum"), avg_rating=("rating", "mean"))
                     .sort_values("revenue", ascending=False).round(2))
    s["category"]["revenue_share_pct"] = (s["category"]["revenue"] / s["category"]["revenue"].sum() * 100).round(1)

    known_city = df[df["city"] != "Unknown"]
    s["city"] = (known_city.groupby("city").agg(orders=("order_id", "count"), revenue=("revenue", "sum"),
                                                avg_order_value=("revenue", "mean"), avg_rating=("rating", "mean"))
                 .sort_values("revenue", ascending=False).round(2))

    known_pay = df[df["payment_method"] != "Unknown"]
    pay = known_pay["payment_method"].value_counts()
    s["payment"] = pd.DataFrame({"orders": pay, "share_pct": (pay / pay.sum() * 100).round(1)})

    s["discount"] = (df.groupby("discount_pct").agg(orders=("order_id", "count"), avg_units=("quantity", "mean"),
                                                    avg_order_value=("revenue", "mean")).round(2))
    s["top_products"] = (df.groupby("product").agg(orders=("order_id", "count"), revenue=("revenue", "sum"))
                         .sort_values("revenue", ascending=False).head(10).round(2))

    orders_per_customer = df.groupby("customer_id").size()
    repeat_ids = orders_per_customer[orders_per_customer > 1].index
    s["customers"] = pd.Series({
        "Customers with 2+ orders (%)": round(len(repeat_ids) / len(orders_per_customer) * 100, 1),
        "Revenue from repeat customers (%)": round(df.loc[df["customer_id"].isin(repeat_ids), "revenue"].sum()
                                                   / df["revenue"].sum() * 100, 1),
        "Most orders by one customer": int(orders_per_customer.max()),
    })
    return s


# ───────────────────────────── 4. CHARTS ─────────────────────────────────────
def save(fig, name: str) -> None:
    fig.savefig(CHART_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_monthly(stats) -> None:
    monthly = stats["monthly"]["revenue"]
    average = monthly.mean()
    top_two = set(monthly.nlargest(2).index)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar([calendar.month_abbr[m] for m in monthly.index], monthly.values,
                  color=[AMBER if m in top_two else LIGHT for m in monthly.index], edgecolor="none")
    ax.axhline(average, color=ROSE, linestyle="--", linewidth=1.4, zorder=1,
               label=f"Monthly average: {money(average)}")
    labels = ax.bar_label(bars, labels=[money(v) for v in monthly.values], padding=3, fontsize=9, color=SLATE)
    for label in labels:
        label.set_bbox({"facecolor": "#fbfbff", "edgecolor": "none", "pad": 1})
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Q1. Which months bring the most revenue?", loc="left")
    ax.set_ylabel("Revenue (Rs)")
    ax.yaxis.set_major_formatter(lambda v, _: money(v))
    ax.set_ylim(0, monthly.max() * 1.15)
    ax.grid(axis="x", visible=False)
    save(fig, "01_monthly_revenue.png")


def chart_category(stats) -> None:
    cat = stats["category"].sort_values("revenue")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(cat.index, cat["revenue"], color=INDIGO)
    bars[-1].set_color(AMBER)
    ax.bar_label(bars, labels=[f"{money(r)}  ({p}%)" for r, p in zip(cat["revenue"], cat["revenue_share_pct"])],
                 padding=5, fontsize=9.5, color=SLATE)
    ax.set_title("Q2. Which product categories earn the most?", loc="left")
    ax.set_xlabel("Revenue (Rs)")
    ax.xaxis.set_major_formatter(lambda v, _: money(v))
    ax.set_xlim(0, cat["revenue"].max() * 1.25)
    ax.grid(axis="y", visible=False)
    save(fig, "02_category_revenue.png")


def chart_discount(stats) -> None:
    d = stats["discount"]
    labels = [f"{int(v)}%" for v in d.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bars = axes[0].bar(labels, d["avg_units"], color=TEAL)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9.5, color=SLATE)
    axes[0].set_title("Units per order", loc="left", fontsize=12)
    axes[0].set_ylim(0, d["avg_units"].max() * 1.2)
    bars = axes[1].bar(labels, d["avg_order_value"], color=INDIGO)
    axes[1].bar_label(bars, labels=[rs(v) for v in d["avg_order_value"]], padding=3, fontsize=9, color=SLATE)
    axes[1].set_title("Average order value (after discount)", loc="left", fontsize=12)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    axes[1].set_ylim(0, d["avg_order_value"].max() * 1.2)
    for ax in axes:
        ax.set_xlabel("Discount given")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Q3. Do bigger discounts make customers buy more?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_discount_effect.png")


def chart_city(stats) -> None:
    city = stats["city"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ordered = city.sort_values("revenue")
    bars = axes[0].barh(ordered.index, ordered["revenue"], color=INDIGO)
    axes[0].bar_label(bars, labels=[money(v) for v in ordered["revenue"]], padding=4, fontsize=9.5, color=SLATE)
    axes[0].set_title("Revenue", loc="left", fontsize=12)
    axes[0].set_xlim(0, ordered["revenue"].max() * 1.22)
    axes[0].xaxis.set_major_formatter(lambda v, _: money(v))
    rated = city.sort_values("avg_rating")
    axes[1].hlines(rated.index, 3.6, rated["avg_rating"], color=LIGHT, linewidth=3)
    axes[1].scatter(rated["avg_rating"], rated.index, s=120, color=ROSE, zorder=3)
    for name, value in rated["avg_rating"].items():
        axes[1].text(value + 0.02, name, f"{value:.2f}", va="center", fontsize=9.5, color=SLATE)
    axes[1].set_xlim(3.6, 4.5)
    axes[1].set_title("Average rating (axis starts at 3.6)", loc="left", fontsize=12)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Q4. Which cities buy the most, and where are customers happiest?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_city_revenue_rating.png")


def chart_payment(stats, unknown: int) -> None:
    pay = stats["payment"]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    wedges, _, autotexts = ax.pie(
        pay["orders"], colors=PALETTE[: len(pay)], startangle=90, counterclock=False,
        autopct="%1.0f%%", pctdistance=0.8, wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2})
    for text in autotexts:
        text.set(color="white", fontweight="bold", fontsize=10)
    ax.legend(wedges, pay.index, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title("Q5. How do customers prefer to pay?", loc="left")
    ax.text(0, 0, f"{pay['orders'].sum():,}\norders", ha="center", va="center", fontsize=13,
            fontweight="bold", color=SLATE)
    fig.text(0.01, 0.01, f"{unknown} orders with unknown payment method are excluded.", fontsize=8.5, color=SLATE)
    save(fig, "05_payment_methods.png")


def chart_cleaning(log: pd.DataFrame) -> None:
    data = log[log["rows_affected"] > 0].sort_values("rows_affected")
    short = (data["issue"].str.split(r" \(|:", regex=True).str[0]
             .replace({"City": "City spelling and spaces", "Category": "Category spelling and spaces"}))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(short, data["rows_affected"], color=ROSE)
    ax.bar_label(bars, padding=4, fontsize=9.5, color=SLATE)
    ax.set_title("Q6. How much of the raw data needed fixing?", loc="left")
    ax.set_xlabel("Rows affected")
    ax.set_xlim(0, data["rows_affected"].max() * 1.12)
    ax.grid(axis="y", visible=False)
    save(fig, "06_data_quality_issues.png")


# ───────────────────────────── 5. FINDINGS ───────────────────────────────────
def write_findings(df: pd.DataFrame, stats, raw_rows: int) -> str:
    monthly = stats["monthly"]["revenue"]
    best = int(monthly.idxmax())
    others = monthly.drop(best).mean()
    top_two_share = monthly.nlargest(2).sum() / monthly.sum() * 100
    top_two_names = " and ".join(calendar.month_name[m] for m in monthly.nlargest(2).index.sort_values())

    cat = stats["category"]
    top_cat = cat.index[0]
    top_two_cat_share = cat["revenue_share_pct"].iloc[:2].sum()
    orders_leader = cat["orders"].idxmax()
    aov_by_cat = cat["revenue"] / cat["orders"]

    disc = stats["discount"]
    low, high = disc.index.min(), disc.index.max()

    city = stats["city"]
    best_city, worst_city = city["avg_rating"].idxmax(), city["avg_rating"].idxmin()

    overview = stats["overview"]
    text = f"""# Findings Summary - ShopEase 2025 Orders

**Dataset:** self-created, {raw_rows:,} raw rows (generated by `generate_dataset.py`), {len(df):,} rows after cleaning.
**Total revenue analysed:** {money(overview['Total revenue (Rs)'])} from {overview['Orders']:,.0f} orders.

## Finding 1 - Sales are strongly seasonal
{calendar.month_name[best]} is the best month with **{money(monthly[best])}**, which is
**{monthly[best] / others:.1f} times** the average of the other eleven months ({money(others)}).
{top_two_names} together bring **{top_two_share:.0f}%** of the whole year's revenue.
*Action:* stock up and plan advertising before the November sale period.

## Finding 2 - Revenue depends on a few categories
**{top_cat}** earns the most (**{cat['revenue_share_pct'].iloc[0]}%** of revenue) and the top two categories together earn
**{top_two_cat_share:.0f}%**. {f"Note that {orders_leader} receives the most orders, but each order is worth only {rs(aov_by_cat[orders_leader])} on average, against {rs(aov_by_cat[top_cat])} for {top_cat}." if orders_leader != top_cat else top_cat + " also receives the most orders."}
*Action:* keep high-value categories well stocked and promote them first.

## Finding 3 - Discounts raise basket size
Orders with **{int(high)}% discount** contain **{disc.loc[high, 'avg_units']:.2f} units** on average, compared with
**{disc.loc[low, 'avg_units']:.2f} units** at {int(low)}% discount. The average order value is {rs(disc.loc[high, 'avg_order_value'])}
at {int(high)}% and {rs(disc.loc[low, 'avg_order_value'])} at {int(low)}%, so extra units do not always make up for the price cut.
*Action:* use deep discounts selectively (for example, to clear stock) instead of on every product.

## Finding 4 - Customer satisfaction differs between cities
**{best_city}** has the highest average rating (**{city.loc[best_city, 'avg_rating']:.2f}**) and **{worst_city}** the lowest
(**{city.loc[worst_city, 'avg_rating']:.2f}**), a gap of {city.loc[best_city, 'avg_rating'] - city.loc[worst_city, 'avg_rating']:.2f} stars.
*Action:* investigate delivery time and service quality in the lowest-rated city.

## Limitations
* The data is synthetic, so the findings describe this dataset only and are not real market facts.
* Ratings exist for only part of the orders ({df['rating'].notna().mean() * 100:.0f}% of cleaned rows).
* The data shows association, not proof of cause (for example, discounts and the November season overlap).
"""
    return text


# ───────────────────────────── MAIN ──────────────────────────────────────────
def main() -> None:
    if not RAW_FILE.exists():
        raise SystemExit(f"Missing {RAW_FILE.name}. Run 'python generate_dataset.py' first.")
    for folder in (TABLE_DIR, CHART_DIR, CLEAN_FILE.parent):
        folder.mkdir(parents=True, exist_ok=True)

    section("1. LOAD AND INSPECT")
    raw = pd.read_csv(RAW_FILE)
    report = inspect(raw)
    print(report)
    (OUT_DIR / "data_inspection.txt").write_text(report, encoding="utf-8")

    section("2. CLEANING LOG")
    df, log = clean(raw)
    log.to_csv(OUT_DIR / "cleaning_log.csv", index=False)
    df.to_csv(CLEAN_FILE, index=False)
    print(log.to_string(index=False, max_colwidth=70))
    print(f"\nRows: {len(raw)} raw -> {len(df)} clean | missing values left: {int(df.isna().sum().sum())}"
          f" (all in 'rating')")

    section("3. SUMMARY STATISTICS")
    stats = compute_stats(df)
    for name, table in stats.items():
        print(f"\n--- {name} ---\n{table.to_string()}")
        table.to_csv(TABLE_DIR / f"{name}.csv")

    section("4. CHARTS")
    chart_monthly(stats)
    chart_category(stats)
    chart_discount(stats)
    chart_city(stats)
    chart_payment(stats, int((df["payment_method"] == "Unknown").sum()))
    chart_cleaning(log)
    print("Saved:", ", ".join(sorted(p.name for p in CHART_DIR.glob("*.png"))))

    section("5. FINDINGS")
    findings = write_findings(df, stats, len(raw))
    (OUT_DIR / "findings.md").write_text(findings, encoding="utf-8")
    print(findings)


if __name__ == "__main__":
    main()
