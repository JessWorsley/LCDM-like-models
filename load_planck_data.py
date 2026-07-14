import matplotlib.pyplot as plt
import numpy as np

from getdist import loadMCSamples
from classy import Class
from constants import SPEED_OF_LIGHT_KM_S

def load_planck():
    """
    Planck 2018 TTTEEE compressed distance priors.

    Returns:
        mean (3,)
        cov (3,3)
        inv_cov (3,3)
    """
    # From ArXiv 1808.05724 Table I: wCDM
    lA = 301.462
    R = 1.7493
    sigma_lA = 0.090
    sigma_R = 0.0047
    corr = 0.47   # correlation coefficient

    mean = np.array([
        R,     # shift parameter
        lA     # acoustic scale
    ])

    cov = np.array([
    [sigma_R**2, corr*sigma_R*sigma_lA],
    [corr*sigma_R*sigma_lA, sigma_lA**2]
    ])

    inv_cov = np.linalg.inv(cov)

    return mean, cov, inv_cov