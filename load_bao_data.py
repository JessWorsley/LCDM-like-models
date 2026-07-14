import numpy as np
import pandas as pd

# =========================================================
# DESI BAO
# =========================================================

def load_desi():
    desi_cov_file = 'data/desi/desi_gaussian_bao_ALL_GCcomb_cov.txt'      # Covariance matrix from DESI BAO
    desi_data_file = 'data/desi/desi_gaussian_bao_ALL_GCcomb_mean.txt'    # Means of z, DV/rs, DM/rs, DH/rs from DESI BAO

    cov_desi = np.loadtxt(desi_cov_file)
    data_desi = pd.read_csv(desi_data_file, sep=' ')
    z_desi, means_desi, kinds_desi = data_desi['z'].values, data_desi['value'].values, data_desi['quantity']
    inv_cov_desi = np.linalg.inv(cov_desi)
    return means_desi, cov_desi, inv_cov_desi, z_desi, kinds_desi