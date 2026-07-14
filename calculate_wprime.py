import pandas as pd
import numpy as np

# =========================================================
# Load existing summary table
# =========================================================

MODEL = '1'
df = pd.read_csv(f"results/mcmc_parameter_summary_model_{MODEL}.csv")

# =========================================================
# Compute wDE'
# =========================================================

def compute_wDE0p(q0, j0, Om0):
    return -(
        2*j0*(Om0 - 1)
        + 2*q0*(2*q0 - 3*Om0 + 1)
        + Om0
    ) / (3 * (Om0 - 1)**2)

# Median / central value
df["wDE0p_mean"] = compute_wDE0p(
    df["q0_mean"],
    df["j0_mean"],
    df["Om0_mean"]
)

# =========================================================
# Propagate upper/lower bounds approximately
# =========================================================

# Upper parameter values
q0_hi  = df["q0_mean"]  + df["q0_plus"]
j0_hi  = df["j0_mean"]  + df["j0_plus"]
Om0_hi = df["Om0_mean"] + df["Om0_plus"]

# Lower parameter values
q0_lo  = df["q0_mean"]  - df["q0_minus"]
j0_lo  = df["j0_mean"]  - df["j0_minus"]
Om0_lo = df["Om0_mean"] - df["Om0_minus"]

# Compute corresponding extrema
w_hi = compute_wDE0p(q0_hi, j0_hi, Om0_hi)
w_lo = compute_wDE0p(q0_lo, j0_lo, Om0_lo)

# Convert into + / - format
df["wDE0p_plus"]  = w_hi - df["wDE0p_mean"]
df["wDE0p_minus"] = df["wDE0p_mean"] - w_lo

# =========================================================
# Reorder columns
# =========================================================

cols = list(df.columns)

insert_before = cols.index("chi2_red")

new_cols = ["wDE0p_mean", "wDE0p_plus", "wDE0p_minus"]

# Remove if already present
for c in new_cols:
    cols.remove(c)

# Insert before chi2_red
cols[insert_before:insert_before] = new_cols

df = df[cols]

# =========================================================
# Save updated file
# =========================================================

df.to_csv(f"results/mcmc_parameter_summary_model_{MODEL}_wDEp.csv", index=False)

print(df)