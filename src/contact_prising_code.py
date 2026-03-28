import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("data/Nat_Gas.csv")

df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
df = df.sort_values("Dates").reset_index(drop=True)

df["t"] = np.arange(len(df))
df["t2"] = df["t"] ** 2
df["month"] = df["Dates"].dt.month
df["year"] = df["Dates"].dt.year
df["month_name"] = df["Dates"].dt.strftime("%b")

# =========================================================
# 2. BUILD PRICE ESTIMATION MODEL
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
coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
df["Predicted"] = X @ coeffs

# =========================================================
# 3. MODEL EVALUATION
# =========================================================
rmse = np.sqrt(mean_squared_error(df["Prices"], df["Predicted"]))
mae = mean_absolute_error(df["Prices"], df["Predicted"])

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")

# =========================================================
# 4. PRICE ESTIMATION FUNCTION
# =========================================================
start_date = df["Dates"].min()

def estimate_gas_price(input_date):
    """
    Estimate natural gas price for any given date
    """
    input_date = pd.to_datetime(input_date)

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
# 5. STORAGE CONTRACT PRICING FUNCTION
# =========================================================
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_storage_volume,
    storage_cost_per_unit_per_month
):
    """
    Prices a gas storage contract.

    Parameters:
    -----------
    injection_dates : list of str
        Dates when gas is injected into storage
    withdrawal_dates : list of str
        Dates when gas is withdrawn from storage
    injection_rate : float
        Maximum units of gas that can be injected per month
    withdrawal_rate : float
        Maximum units of gas that can be withdrawn per month
    max_storage_volume : float
        Maximum storage capacity
    storage_cost_per_unit_per_month : float
        Cost to store one unit of gas per month

    Returns:
    --------
    total_contract_value : float
    breakdown_df : pandas.DataFrame
    """

    if len(injection_dates) != len(withdrawal_dates):
        raise ValueError("Injection dates and withdrawal dates must have the same length.")

    storage_level = 0
    total_contract_value = 0
    contract_details = []

    for inj_date, wdr_date in zip(injection_dates, withdrawal_dates):
        inj_date = pd.to_datetime(inj_date)
        wdr_date = pd.to_datetime(wdr_date)

        if wdr_date <= inj_date:
            raise ValueError(f"Withdrawal date {wdr_date.date()} must be after injection date {inj_date.date()}.")

        # Estimate prices using previous model
        buy_price = estimate_gas_price(inj_date)
        sell_price = estimate_gas_price(wdr_date)

        # Holding period in months
        months_held = max(1, (wdr_date.year - inj_date.year) * 12 + (wdr_date.month - inj_date.month))

        # Maximum possible volume for this cycle
        injectable_volume = min(injection_rate, max_storage_volume - storage_level)
        withdrawable_volume = min(withdrawal_rate, injectable_volume)

        volume = min(injectable_volume, withdrawable_volume)

        # If no space or no withdraw capacity, skip
        if volume <= 0:
            cycle_profit = 0
            storage_cost = 0
        else:
            # Injection cost
            purchase_cost = buy_price * volume

            # Sale revenue
            sale_revenue = sell_price * volume

            # Storage cost
            storage_cost = storage_cost_per_unit_per_month * volume * months_held

            # Net profit
            cycle_profit = sale_revenue - purchase_cost - storage_cost

            # Simulate storage usage
            storage_level += volume
            storage_level -= volume  # withdrawn fully at withdrawal date

        total_contract_value += cycle_profit

        contract_details.append({
            "Injection Date": inj_date.date(),
            "Withdrawal Date": wdr_date.date(),
            "Buy Price": buy_price,
            "Sell Price": sell_price,
            "Months Held": months_held,
            "Volume": volume,
            "Storage Cost": round(storage_cost, 2),
            "Cycle Profit": round(cycle_profit, 2)
        })

    breakdown_df = pd.DataFrame(contract_details)

    return round(total_contract_value, 2), breakdown_df

# =========================================================
# 6. TEST CASES
# =========================================================
print("\n" + "=" * 60)
print("STORAGE CONTRACT TESTING")
print("=" * 60)

# Example contract
injection_dates = ["2025-01-31", "2025-03-31", "2025-05-31"]
withdrawal_dates = ["2025-02-28", "2025-06-30", "2025-09-30"]
# Better test case
injection_dates = ["2025-04-30", "2025-05-31", "2025-06-30"]
withdrawal_dates = ["2025-10-31", "2025-11-30", "2025-12-31"]

injection_rate = 1000
withdrawal_rate = 1000
max_storage_volume = 3000
storage_cost_per_unit_per_month = 0.05   # reduced storage cost

contract_value, contract_table = price_storage_contract(
    injection_dates=injection_dates,
    withdrawal_dates=withdrawal_dates,
    injection_rate=injection_rate,
    withdrawal_rate=withdrawal_rate,
    max_storage_volume=max_storage_volume,
    storage_cost_per_unit_per_month=storage_cost_per_unit_per_month
)

print(f"\nEstimated Contract Value: {contract_value}")
print("\nContract Breakdown:")
print(contract_table)

# =========================================================
# 7. OPTIONAL VISUALIZATION OF BUY/SELL PRICES
# =========================================================
plt.figure(figsize=(12, 6))

for i in range(len(contract_table)):
    plt.plot(
        [pd.to_datetime(contract_table.loc[i, "Injection Date"]),
         pd.to_datetime(contract_table.loc[i, "Withdrawal Date"])],
        [contract_table.loc[i, "Buy Price"],
         contract_table.loc[i, "Sell Price"]],
        marker="o",
        linewidth=2,
        label=f"Cycle {i+1}"
    )

plt.title("Injection vs Withdrawal Price Cycles")
plt.xlabel("Date")
plt.ylabel("Estimated Gas Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/contract_price_cycles.png")
plt.show()

'''
# =========================================================
# 8. FUTURE EXTENSIONS
# ========================================================
# Storage Contract Pricing Interpretation
The contract value depends on the spread between the injection price and the withdrawal price after accounting for storage costs.
In the tested strategy:
gas was injected during relatively lower-priced months (April - June 2025) and withdrawn during relatively higher-priced months (October - December 2025) 
This created a positive arbitrage opportunity.
Although one cycle remained slightly unprofitable due to insufficient price spread,
the remaining cycles generated enough profit to produce an overall positive contract value of 1590.0.

============================================================
MODEL PERFORMANCE
============================================================
RMSE: 0.1997
MAE : 0.1498

============================================================
STORAGE CONTRACT TESTING
============================================================

Estimated Contract Value: 1590.0

Contract Breakdown:
  Injection Date Withdrawal Date  Buy Price  Sell Price  Months Held  Volume  Storage Cost  Cycle Profit
0     2025-04-30      2025-10-31      12.61       12.80            6    1000         300.0        -110.0
1     2025-05-31      2025-11-30      12.31       13.19            6    1000         300.0         580.0
2     2025-06-30      2025-12-31      12.08       13.50            6    1000         300.0        1120.0

'''