import numpy as np
import pandas as pd

# TODO: Add file paths

# =========================================================
# Marginalise over nuisance parameter M
# =========================================================

def marginalise_over_M(n, cov, means, reshape=True):
    """
    Performs analytic marginalisation over the nuisance parameter M.

    Args:
        n : length of dataset
        means  : 1D array of distance modulus means
        cov : NxN covariance matrix
        reshape : reshape into covariance matrix

    Returns:
        cov_sn, inv_cov_sn
    """
    # Reshape into covariance matrix
    if reshape:
        cov = cov.reshape((n, n))

    # Invert for use in likelihoods
    inv_cov_sn = np.linalg.inv(cov)

    # Analytically marginalise over absolute magnitude
    deriv = np.ones_like(means)[:, None]
    derivp = inv_cov_sn.dot(deriv)
    fisher = deriv.T.dot(derivp)
    inv_cov_sn -= derivp.dot(np.linalg.solve(fisher, derivp.T))
    
    cov_sn = cov

    return cov_sn, inv_cov_sn

# =========================================================
# Union 3
# =========================================================

def load_union():
    union_data_file = pd.read_csv('data/union/lcparam_full.txt', sep=' ', header=0)
    z_union, means_union = union_data_file['zcmb'].values, union_data_file['mb'].values # Heliocentric redshifts and observed distance moduli

    # Covariance matrix
    with open('data/union/mag_covmat.txt') as file:
        union_n = int(file.readline())  # read the first line (matrix size)
        union_cov_file = np.loadtxt(file)   # read the rest of the numbers

    cov_union, inv_cov_union = marginalise_over_M(union_n, union_cov_file, means_union)
    return means_union, cov_union, inv_cov_union, z_union

# =========================================================
# Pantheon+
# =========================================================

def load_pantheon():
    pantheon_data_file = pd.read_csv('data/pantheon/Pantheon+SH0ES.dat', delimiter=' ', header=0, usecols=['zHD', 'm_b_corr'])
    z_pantheon, means_pantheon = pantheon_data_file['zHD'], pantheon_data_file['m_b_corr']  # Heliocentric redshifts and observed distance moduli

    # Covariance matrix
    with open('data/pantheon/Pantheon+SH0ES_STAT+SYS.cov') as file:
        pantheon_header = int(file.readline())  # read the first line and pass
        pantheon_cov_file = np.loadtxt(file)   # read the rest of the numbers
        pantheon_n = int(np.sqrt(pantheon_cov_file.size)) # determine matrix size

    cov_pantheon, inv_cov_pantheon = marginalise_over_M(pantheon_n, pantheon_cov_file, means_pantheon)
    return means_pantheon, cov_pantheon, inv_cov_pantheon, z_pantheon

# =========================================================
# DESY
# =========================================================

def load_desy():
    """
    From DES Git: Please note that the Covariance Matrices provided here are INVERSE covariance matrices. YOU FOOL.
    """
    desy_data_file = pd.read_csv('data/desy/DES-Dovekie_HD.csv', sep=',', header=0)

    z_desy, means_desy, mag_err_desy = desy_data_file['zHD'].values, desy_data_file['MU'].values, desy_data_file['MUERR'].values

    # Remove outliers
    mask = np.isfinite(mag_err_desy) & (mag_err_desy < 1.0)
    # print('No. outliers removed from DESY5:', len(z_desy) - len(z_desy[mask]))

    z_desy = z_desy[mask]
    means_desy = means_desy[mask]
    mag_err_desy = mag_err_desy[mask]

    desy_inv_cov_file = np.load('data/desy/STAT+SYS.npz')

    flat_inv_cov = desy_inv_cov_file['cov']
    desy_n = int(desy_inv_cov_file['nsn'][0])

    # Only given upper half of symmetric matrix so we have to mirror it
    inv_cov_matrix_desy = np.zeros((desy_n, desy_n))
    triangular_inds = np.triu_indices(desy_n)
    inv_cov_matrix_desy[triangular_inds] = flat_inv_cov
    inv_cov_matrix_desy += np.triu(inv_cov_matrix_desy, 1).T  # mirror upper triangle to lower

    inv_cov_matrix_desy = inv_cov_matrix_desy[np.ix_(mask, mask)]

    inv_cov_desy = inv_cov_matrix_desy
    cov_desy = np.linalg.inv(inv_cov_desy)

    cov_desy, inv_cov_desy = marginalise_over_M(len(z_desy), cov_desy, means_desy, reshape=False)

    # Diagnostics
    # print('Length of data:', len(mag_err_desy))
    # print('First few mag_err:', mag_err_desy[:10])
    # print('Min mag_err:', np.min(mag_err_desy))
    # print('Mean squared mag_err:', np.mean(mag_err_desy**2))
    # print('Max mag_err:', np.max(mag_err_desy))
    # print('Min of diagonals:', np.min(np.diag(cov_desy)))
    # print('Mean of diagonals:', np.mean(np.diag(cov_desy)))
    # print('Max of diagonals:', np.max(np.diag(cov_desy)))

    return means_desy, cov_desy, inv_cov_desy, z_desy

load_desy()