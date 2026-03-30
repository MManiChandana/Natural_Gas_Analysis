import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("data/loan_data.csv")

print("=" * 70)
print("DATA OVERVIEW")
print("=" * 70)
print(df.head())

fico_df = df[["fico_score", "default"]].copy()

print("\nDataset Shape:", fico_df.shape)
print("\nFICO Range:", fico_df["fico_score"].min(), "to", fico_df["fico_score"].max())

# =========================================================
# 2. CHOOSE NUMBER OF BUCKETS
# =========================================================
NUM_BUCKETS = 5

# =========================================================
# 3. CREATE FICO BUCKETS USING QUANTIZATION
# =========================================================
# qcut creates equal-frequency buckets
fico_df["bucket"] = pd.qcut(
    fico_df["fico_score"],
    q=NUM_BUCKETS,
    duplicates="drop"
)

# Extract bucket boundaries
bucket_ranges = fico_df["bucket"].cat.categories
fico_boundaries = [int(interval.right) for interval in bucket_ranges[:-1]]

print("\n" + "=" * 70)
print("FICO BUCKET BOUNDARIES")
print("=" * 70)
print(fico_boundaries)

# =========================================================
# 4. ASSIGN RATINGS
# Lower rating = better score
# =========================================================
# Since higher FICO is better, reverse bucket order
bucket_order = list(bucket_ranges)
bucket_to_rating = {
    bucket: rating
    for rating, bucket in enumerate(reversed(bucket_order), start=1)
}

fico_df["rating"] = fico_df["bucket"].map(bucket_to_rating)

# =========================================================
# 5. BUILD RATING MAP
# =========================================================
rating_map = fico_df.groupby("rating").agg(
    min_fico=("fico_score", "min"),
    max_fico=("fico_score", "max"),
    borrowers=("default", "count"),
    defaults=("default", "sum")
).reset_index()

rating_map["pd"] = rating_map["defaults"] / rating_map["borrowers"]
rating_map = rating_map.sort_values("rating").reset_index(drop=True)

print("\n" + "=" * 70)
print("RATING MAP")
print("=" * 70)
print(rating_map)

# =========================================================
# 6. VISUALIZATIONS
# =========================================================
plt.figure(figsize=(10, 6))
plt.bar(rating_map["rating"].astype(str), rating_map["pd"])
plt.title("Probability of Default by Rating")
plt.xlabel("Rating (1 = Best, Higher = Riskier)")
plt.ylabel("Probability of Default")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("visualizations3/fico_rating_pd.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(fico_df["fico_score"], bins=30, alpha=0.7)
for boundary in fico_boundaries:
    plt.axvline(boundary, linestyle="--")
plt.title("FICO Score Distribution with Bucket Boundaries")
plt.xlabel("FICO Score")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations3/fico_score_distribution.png")
plt.show()

# =========================================================
# 7. FUNCTIONS FOR FUTURE DATA
# =========================================================
def fico_to_rating(fico_score):
    """
    Assign rating to a new FICO score based on learned bucket ranges.
    Lower rating = better credit
    """
    for interval, rating in bucket_to_rating.items():
        if fico_score in interval:
            return rating

    # Edge cases
    if fico_score < bucket_ranges[0].left:
        return max(bucket_to_rating.values())
    elif fico_score > bucket_ranges[-1].right:
        return min(bucket_to_rating.values())

def fico_to_pd(fico_score):
    rating = fico_to_rating(fico_score)
    pd_value = rating_map.loc[rating_map["rating"] == rating, "pd"].values[0]
    return {
        "fico_score": fico_score,
        "rating": int(rating),
        "probability_of_default": round(float(pd_value), 4)
    }

# =========================================================
# 8. TEST CASES
# =========================================================
print("\n" + "=" * 70)
print("TEST CASES")
print("=" * 70)

test_scores = [580, 620, 660, 700, 760]

for score in test_scores:
    print(fico_to_pd(score))

# =========================================================
# 9. CUSTOM INPUT
# =========================================================
print("\n" + "=" * 70)
print("CUSTOM FICO RATING LOOKUP")
print("=" * 70)

try:
    user_fico = int(input("Enter FICO score (300-850): "))

    if user_fico < 300 or user_fico > 850:
        raise ValueError("FICO score should be between 300 and 850")

    result = fico_to_pd(user_fico)
    print("\nResult:")
    print(result)

except ValueError as e:
    print("\nInput Error:", e)