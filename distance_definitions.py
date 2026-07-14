import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from constants import SPEED_OF_LIGHT_KM_S

# =========================================================
# Distance Definitions
# =========================================================

def luminosity_distance_from_H(z_eval, H_eval):
    """
    Compute luminosity distance for arbitrary redshifts via interpolation.
    z_eval : array_like
        Redshifts to evaluate DL(z)
    H_eval : array_like
        Hubble parameter evaluated on dense grid covering z_eval
    """
    z_dense = np.linspace(0.0, np.max(z_eval), 1000)  # dense grid for integration
    H_dense = np.interp(z_dense, z_eval, H_eval)      # interpolate H to dense grid

    chi_dense = comoving_distance_from_H(z_dense, H_dense)
    if chi_dense is None:
        return None

    # Luminosity distance: DL = (1+z) * chi
    DL_dense = (1.0 + z_dense) * chi_dense

    # Interpolate DL to z_eval points
    interp_DL = interp1d(z_dense, DL_dense, kind='cubic', fill_value='extrapolate')
    DL_vals = interp_DL(z_eval)

    # must be positive and finite
    if np.any(DL_vals <= 0) or np.any(~np.isfinite(DL_vals)):
        return None

    return DL_vals

def comoving_distance_from_H(z_grid, H_grid, c_km_s=SPEED_OF_LIGHT_KM_S):
    # Define grid for numerical stability, then interpolate from grid to data points
    # Integrating between the uneven data points would give discontinuous results
    # require z_grid[0] == 0.0 (or include 0 explicitly)
    inv_H = 1.0 / H_grid
    chi_vals = c_km_s * cumulative_trapezoid(inv_H, z_grid, initial=0)
    return chi_vals

def D_M_from_chi(chi_vals):
    # FLAT cosmology
    return chi_vals

def D_A_from_DM(DM_vals, z_grid):
    # D_A = H(z)/(1+z)
    return DM_vals / (1.0 + z_grid)

def D_V_from_DA_H(DA_vals, z_grid, H_grid, c_km_s=SPEED_OF_LIGHT_KM_S):
    # D_V = [ (1+z)^2 D_A^2 (c z / H(z)) ]^(1/3)
    return ( (1.0 + z_grid)**2 * DA_vals**2 * (c_km_s * z_grid / H_grid) )**(1.0/3.0)

def D_H_from_H(H_grid, c_km_s=SPEED_OF_LIGHT_KM_S):
    # D_H = c / H(z)
    return c_km_s / H_grid
