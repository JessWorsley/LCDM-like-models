import numpy as np

# =========================================================
# Priors and Initial Estimates
# =========================================================

# Priors from Rodrigues 2025 (changed 70->72 though)
H0_min, H0_max = 50.0, 72.0
Om0_min, Om0_max = 0.0, 1.0

q0_min, q0_max = -2.0, 2.0
j0_min, j0_max = -5.0, 5.0

eps_min, eps_max = -1.0, 5.0
c1_min, c1_max = 2.0, 8.0

wDE0_min, wDE0_max = -2.0, 0.0

theta_initial = np.array([68, -0.5, 1.0, 0.3])