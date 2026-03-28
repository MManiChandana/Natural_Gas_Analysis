import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("data/Nat_Gas.csv")

# Convert date column
df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
df = df.sort_values("Dates").reset_index(drop=True)

# Feature engineering
df["t"] = np.arange(len(df))              # time index
df["t2"] = df["t"] ** 2                  # quadratic trend
df["month"] = df["Dates"].dt.month
df["year"] = df["Dates"].dt.year
df["month_name"] = df["Dates"].dt.strftime("%b")

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(df.head())
print("\nDate Range:", df["Dates"].min().date(), "to", df["Dates"].max().date())
print("Total Records:", len(df))

# =========================================================
# 2. VISUALIZATION 1 - HISTORICAL PRICE TREND
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(df["Dates"], df["Prices"], marker="o", linewidth=2)
plt.title("Natural Gas Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/natural_gas_price_trend.png")  # Save the figure
plt.show()

# =========================================================
# 3. VISUALIZATION 2 - MONTHLY SEASONAL PATTERN
# =========================================================
monthly_avg = df.groupby("month_name", sort=False)["Prices"].mean()

# Reorder months correctly
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_avg = monthly_avg.reindex(month_order)

plt.figure(figsize=(12, 6))
plt.plot(monthly_avg.index, monthly_avg.values, marker="o", linewidth=2)
plt.title("Average Natural Gas Price by Month")
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/monthly_seasonal_pattern.png")     
plt.show()

# =========================================================
# 4. VISUALIZATION 3 - YEARLY PRICE COMPARISON
# =========================================================
plt.figure(figsize=(12, 6))

for year in sorted(df["year"].unique()):
    temp = df[df["year"] == year]
    plt.plot(temp["month"], temp["Prices"], marker="o", linewidth=2, label=str(year))

plt.title("Natural Gas Prices by Year")
plt.xlabel("Month")
plt.ylabel("Price")
plt.xticks(range(1, 13), month_order)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/yearly_price_comparison.png")
plt.show()

# =========================================================
# 5. BUILD FORECASTING MODEL
#    MODEL = QUADRATIC TREND + SEASONALITY
# =========================================================
X = np.column_stack([
    np.ones(len(df)),
    df["t"],
    df["t2"],
    np.sin(2 * np.pi * df["month"] / 12),
    np.cos(2 * np.pi * df["month"] / 12)
])

y = df["Prices"].values

# Fit regression coefficients
coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

# Historical predictions
df["Predicted"] = X @ coeffs

# =========================================================
# 6. MODEL EVALUATION
# =========================================================
rmse = np.sqrt(mean_squared_error(df["Prices"], df["Predicted"]))
mae = mean_absolute_error(df["Prices"], df["Predicted"])

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")

# =========================================================
# 7. VISUALIZATION 4 - ACTUAL VS FITTED
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(df["Dates"], df["Prices"], marker="o", linewidth=2, label="Actual Prices")
plt.plot(df["Dates"], df["Predicted"], linestyle="--", linewidth=2, label="Fitted Model")
plt.title("Actual vs Fitted Natural Gas Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/actual_vs_fitted.png")
plt.show()

# =========================================================
# 8. PRICE ESTIMATION FUNCTION
# =========================================================
start_date = df["Dates"].min()

def estimate_gas_price(input_date):
    """
    Estimate natural gas price for any given date
    (historical or up to 1 year into the future).
    """
    input_date = pd.to_datetime(input_date)

    # Months difference from start date
    months_diff = (input_date.year - start_date.year) * 12 + (input_date.month - start_date.month)

    month = input_date.month

    x_input = np.array([
        1,
        months_diff,
        months_diff ** 2,
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12)
    ])

    predicted_price = x_input @ coeffs
    return round(float(predicted_price), 2)

# =========================================================
# 9. FORECAST NEXT 12 MONTHS
# =========================================================
future_dates = pd.date_range(
    start=df["Dates"].max() + pd.offsets.MonthEnd(1),
    periods=12,
    freq="ME"
)

future_prices = [estimate_gas_price(d) for d in future_dates]

forecast_df = pd.DataFrame({
    "Dates": future_dates,
    "Forecasted_Price": future_prices
})

# Add confidence-style range (simple visual band)
forecast_df["Upper"] = forecast_df["Forecasted_Price"] + rmse
forecast_df["Lower"] = forecast_df["Forecasted_Price"] - rmse

# =========================================================
# 10. VISUALIZATION 5 - HISTORICAL + FORECAST
# =========================================================
plt.figure(figsize=(14, 7))

# Historical actual
plt.plot(df["Dates"], df["Prices"], marker="o", linewidth=2, label="Historical Prices")

# Historical fitted
plt.plot(df["Dates"], df["Predicted"], linestyle="--", linewidth=2, label="Fitted Model")

# Forecasted future
plt.plot(
    forecast_df["Dates"],
    forecast_df["Forecasted_Price"],
    marker="o",
    linewidth=2,
    label="Next 12 Months Forecast"
)

# Forecast band
plt.fill_between(
    forecast_df["Dates"],
    forecast_df["Lower"],
    forecast_df["Upper"],
    alpha=0.2,
    label="Forecast Range"
)

# Connect last actual to first forecast
plt.plot(
    [df["Dates"].iloc[-1], forecast_df["Dates"].iloc[0]],
    [df["Prices"].iloc[-1], forecast_df["Forecasted_Price"].iloc[0]],
    linestyle=":",
    linewidth=2
)

# Forecast divider
plt.axvline(df["Dates"].iloc[-1], color="gray", linestyle=":", label="Forecast Start")

# Forecast shaded region
plt.axvspan(forecast_df["Dates"].min(), forecast_df["Dates"].max(), alpha=0.08)

plt.title("Natural Gas Price Forecast (Next 12 Months)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/forecast_next_12_months.png")
plt.show()

# =========================================================
# 11. SHOW FORECAST TABLE
# =========================================================
print("\n" + "=" * 60)
print("NEXT 12 MONTHS FORECAST")
print("=" * 60)
print(forecast_df[["Dates", "Forecasted_Price"]])

# =========================================================
# 12. SAMPLE PREDICTIONS
# =========================================================
print("\n" + "=" * 60)
print("SAMPLE DATE ESTIMATIONS")
print("=" * 60)

sample_dates = [
    "2021-06-15",
    "2023-12-01",
    "2025-03-15",
    "2025-09-30"
]

for d in sample_dates:
    print(f"Estimated gas price on {d}: {estimate_gas_price(d)}")

# =========================================================
# 13. USER INPUT
# =========================================================
print("\n" + "=" * 60)
print("CUSTOM DATE PRICE ESTIMATION")
print("=" * 60)

user_date = input("Enter a date (YYYY-MM-DD): ")
predicted = estimate_gas_price(user_date)
print(f"Estimated natural gas price on {user_date}: {predicted}")