import pandas as pd
import numpy as np

# Load your generated weights
my_weights = pd.read_pickle("my_rp.pkl")

# Load the official answer weights
official_weights = pd.read_pickle("./Answer/rp.pkl")

# Choose a debug date
debug_date = "2019-03-15"

# Ensure both DataFrames are aligned
my_row = my_weights.loc[debug_date]
official_row = official_weights.loc[debug_date]

# Compute difference
diff = my_row - official_row
abs_diff = diff.abs()

# Judge-style threshold
tolerance = 1e-6

print("Official index head:", official_weights.index[:5])
print("Official index tail:", official_weights.index[-5:])
print("Official columns:", official_weights.columns)
print("Official row on debug date:\n", official_weights.loc[debug_date])

print("All non-zero rows in official weights:")
print(official_weights[(official_weights != 0).any(axis=1)])

print("Official dtypes:")
print(official_weights.dtypes)

# Print header
print(f"📅 Difference on {debug_date}\n")
print(f"{'Column':<6} | {'Expected':>10} | {'Yours':>10} | {'Δ':>10} | Result")
print("-" * 55)

# Iterate and judge
for col in official_row.index:
    expected = official_row[col]
    yours = my_row[col]
    delta = abs_diff[col]
    verdict = "✅" if delta < tolerance else "❌"
    print(f"{col:<6} | {expected:>10.6f} | {yours:>10.6f} | {delta:>10.6f} | {verdict}")

# Summary
print(f"\nMax difference: {abs_diff.max():.10f}")