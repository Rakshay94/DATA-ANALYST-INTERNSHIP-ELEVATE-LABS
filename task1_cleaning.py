import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_sample_dataset(path="raw_dataset.csv", n=200, seed=42):
    """Generate a synthetic raw dataset with common quality issues and save to CSV."""
    random.seed(seed)
    np.random.seed(seed)

    def random_date(start, end):
        dt = start + timedelta(days=random.randint(0, (end - start).days))
        formats = ["%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y"]
        fmt = random.choice(formats)
        return dt.strftime(fmt)

    customer_ids = [f"C{1000+i}" for i in range(n)]
    names = [f"User_{i}" for i in range(n)]
    genders_raw = ["Male", "male", "M", "Female", "female", "F", "Other", "other", None]
    countries_raw = ["USA", "United States", "US", "India", "IN", "india", "UK", "United Kingdom", "U.K.", "Australia", "AU", None]
    signup_dates = [random_date(datetime(2022,1,1), datetime(2023,12,31)) for _ in range(n)]
    last_purchase_dates = [random_date(datetime(2023,1,1), datetime(2025,7,31)) for _ in range(n)]

    ages = np.random.normal(loc=35, scale=12, size=n).astype(float)
    for i in random.sample(range(n), 5):
        ages[i] = random.choice([150, -5, 200, 0, 120])
    for i in random.sample(range(n), 10):
        ages[i] = np.nan

    purchase = np.random.normal(loc=200, scale=80, size=n)
    for i in random.sample(range(n), 5):
        purchase[i] = random.choice([5000, -100, 10000])
    purchase = np.round(purchase,2)

    emails = [f"user{i}@example.com" for i in range(n)]
    for i in random.sample(range(n), 15):
        replacement = random.choice(range(n))
        emails[i] = emails[replacement]

    df = pd.DataFrame({
        "Customer ID": customer_ids,
        "Name": names,
        "Gender": [random.choice(genders_raw) for _ in range(n)],
        "Country": [random.choice(countries_raw) for _ in range(n)],
        "Signup Date": signup_dates,
        "Last Purchase": last_purchase_dates,
        "Age": ages,
        "Purchase Amount": purchase,
        "Email": emails
    })

    for _ in range(5):
        dup = df.sample(1)
        df = pd.concat([df, dup], ignore_index=True)

    df.to_csv(path, index=False)
    print(f"Generated synthetic raw dataset with {len(df)} rows to {path}")
    return df

def clean_dataset(input_path="raw_dataset.csv", output_path="cleaned_dataset.csv", summary_path="cleaning_summary.txt"):
    if not os.path.exists(input_path):
        print(f"{input_path} not found. Generating synthetic sample dataset.")
        df_raw = generate_sample_dataset(input_path)
    else:
        df_raw = pd.read_csv(input_path)

    df = df_raw.copy()

    # 1. Remove exact duplicate rows
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    removed_dup = before_dup - after_dup

    # 2. Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 3. Handle age: fill missing and treat outliers
    # Compute median excluding extreme/unrealistic ages (<1, >100)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    valid_age_median = df.loc[df["age"].between(1, 100), "age"].median()
    df["age"] = df["age"].fillna(valid_age_median)
    outlier_age_mask = ~df["age"].between(18, 100)
    num_age_fixed = int(outlier_age_mask.sum())
    df.loc[outlier_age_mask, "age"] = valid_age_median
    df["age"] = df["age"].round().astype(int)

    # 4. Standardize gender
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].replace({"m": "male", "f": "female", "none": "unknown", "nan": "unknown"})
    df["gender"] = df["gender"].fillna("unknown")
    df["gender"] = df["gender"].str.capitalize()

    # 5. Standardize country
    country_map = {
        "usa": "United States", "united states": "United States", "us": "United States",
        "india": "India", "in": "India",
        "uk": "United Kingdom", "united kingdom": "United Kingdom", "u.k.": "United Kingdom",
        "australia": "Australia", "au": "Australia"
    }
    df["country"] = df["country"].astype(str).str.strip().str.lower().replace(country_map)
    df.loc[df["country"].isin(["nan", "none", ""]), "country"] = "Unknown"
    df["country"] = df["country"].fillna("Unknown")
    df.loc[df["country"] == "unknown", "country"] = "Unknown"

    # 6. Parse and unify dates
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce", infer_datetime_format=True)
    df["last_purchase"] = pd.to_datetime(df["last_purchase"], errors="coerce", infer_datetime_format=True)
    # Forward fill any unparsed dates
    df["signup_date"] = df["signup_date"].fillna(method="ffill")
    df["last_purchase"] = df["last_purchase"].fillna(method="ffill")
    df["signup_date"] = df["signup_date"].dt.strftime("%d-%m-%Y")
    df["last_purchase"] = df["last_purchase"].dt.strftime("%d-%m-%Y")

    # 7. Purchase amount: fix invalid and cap outliers
    df["purchase_amount"] = pd.to_numeric(df["purchase_amount"], errors="coerce")
    median_purchase = df.loc[df["purchase_amount"] > 0, "purchase_amount"].median()
    negatives = int((df["purchase_amount"] <= 0).sum())
    df.loc[df["purchase_amount"] <= 0, "purchase_amount"] = median_purchase
    Q1 = df["purchase_amount"].quantile(0.25)
    Q3 = df["purchase_amount"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df["purchase_amount"] = df["purchase_amount"].clip(lower=lower_bound, upper=upper_bound)

    # 8. Flag duplicate emails
    df["email_dup_flag"] = df.duplicated(subset=["email"], keep=False)

    # Export cleaned data
    df.to_csv(output_path, index=False)

    # Create summary
    summary_lines = [
        f"Initial rows (including exact duplicates): {len(df_raw)}",
        f"Rows after removing exact duplicates: {after_dup} (duplicates removed: {removed_dup})",
        f"Missing/invalid ages filled with median of realistic ages: {int(df_raw['Age'].isna().sum()) if 'Age' in df_raw.columns else 'N/A'}",
        f"Outlier ages (outside 18-100) replaced: {num_age_fixed}",
        f"Standardized gender values: {', '.join(sorted(df['gender'].unique()))}",
        f"Standardized countries: {', '.join(sorted(set(df['country'].unique())))}",
        f"Purchase amount negative/zero fixed count: {negatives}",
        f"IQR capping applied with bounds: lower={lower_bound:.2f}, upper={upper_bound:.2f}",
        f"Duplicate emails flagged: {int(df['email_dup_flag'].sum())}"
    ]
    with open(summary_path, "w") as f:
        f.write("Summary of Cleaning Task:\n")
        for line in summary_lines:
            f.write(f"- {line}\n")

    print("Cleaning complete.")
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    clean_dataset()
