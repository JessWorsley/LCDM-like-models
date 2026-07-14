import numpy as np

# =========================================================
# CC
# =========================================================

def load_cc():
    cc_data_file = "data/CC.txt"   # z, H_obs, H_err
    data_cc = np.loadtxt(cc_data_file)
    z_cc = np.array(data_cc[:,0].astype(float))
    means_cc = data_cc[:,1].astype(float)
    mu_cc = data_cc[:,2].astype(float)

    # ensure strictly increasing z (add a lil jitter to duplicates)
    # solve_ivp gets mad if z is not increasing
    for i in range(1, len(z_cc)):
        if z_cc[i] <= z_cc[i-1]:
            z_cc[i] = z_cc[i-1] + 1e-8

    zmin_cc, zmax_cc = z_cc[0], z_cc[-1]
    return means_cc, mu_cc, z_cc