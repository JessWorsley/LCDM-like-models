import numpy as np
from constants import EXP_THRESHOLD

def background_analytical(theta, z_eval, model='model_1', exp_threshold=EXP_THRESHOLD):
    """
    Analytic H(z) that includes q0 and j0 (eq 34a from Chakraborty, Louw, et al 2025).
    Returns H(z) array or None on failure (invalid params / overflow / negative inside sqrt).
    Set safe threshold: np.exp(700) is ≈ 1e304 (close to float max)
    """
    H0, q0, j0, Om0 = theta

    # Guard against infinite values
    if not np.isfinite(H0) or not np.isfinite(q0) or not np.isfinite(j0) or not np.isfinite(Om0):
        return None

    # Avoid q0=-1 which makes denominator blow up
    if abs(1.0 + q0) < 1e-12:
        return None
    
    # Compute (1+z)**alpha via exp(exponent * log1p(z)), but guard exponent size
    z_eval = np.asarray(z_eval)
    log_base = np.log1p(z_eval)  # log(1+z) is stable for small z

    if model == 'model_1':
        exponent = (2.0 + j0 + 3.0 * q0) / (1.0 + q0)
        denominator = (2.0 + j0 + 3.0 * q0) / (1.0 + q0)**2

        # avoid tiny denominator
        if abs(denominator) < 1e-12:
            return None

        # Compute max exponent magnitude
        max_exp_arg = np.max(exponent * log_base)
        min_exp_arg = np.min(exponent * log_base)
        # Overflow guard
        if max_exp_arg > exp_threshold:
            return None
        # Avoid extremely small values:
        if min_exp_arg < -1000:
            pass

        h = H0 * np.sqrt( ( 2 * (1 + z_eval)**exponent + (j0 - q0 * (1 + 2 * q0))/(1 + q0)**2 ) / denominator )
        Om = ( Om0 * (z_eval+1)**3 * (j0+3*q0+2) ) / ( (q0+1)**2 * ( 2*(z_eval+1)**((j0+3*q0+2)/(q0+1)) + (j0-q0*(2*q0+1))/(q0+1)**2))

        # try:
        #     return h, Om
        # except Exception:
        #     return None
        return h, Om
        
    elif model == 'model_2' :
        exponent = np.sqrt(8 * j0 + 1)
        denominator = 2 * np.sqrt(8 * j0 + 1)

        # avoid tiny denominator
        if abs(denominator) < 1e-12:
            return None

        # Compute max exponent magnitude
        max_exp_arg = np.max(exponent * log_base)
        min_exp_arg = np.min(exponent * log_base)
        
        # Overflow guard
        if max_exp_arg > exp_threshold:
            return None
        # Avoid extremely small values:
        if min_exp_arg < -1000:
            pass
        
        h = H0 * np.sqrt( (z_eval + 1)**(3/2 - exponent/2) * ( exponent + 4 * q0 + 1 ) * ( (z_eval + 1)**exponent - 1 ) / denominator + 1 )
        Om = ( Om0 * (z_eval + 1)**(1/2 * (exponent + 3))) / ( (exponent + 4 * q0 + 1) * ((z_eval + 1)**exponent - 1) / (2 * exponent) + 1 )

        # try:
        #     return h, Om
        # except Exception:
        #     return None
        return h, Om
    
    elif model == 'model_3':
        denominator = 2*j0 - 6*q0 + 1
        exponent = 2*(j0-1)/(2*q0-1)

        # avoid tiny denominator
        if abs(denominator) < 1e-3:
            return None
        if abs(2*q0 - 1) < 1e-3:
            return None
        
        # Ensure numerator stays positive at high z
        if j0 < q0*(2*q0 + 1):
            return None

        # # Enforce matter domination at high z
        # if j0 < 0.5:
        #     return None
        
        # Compute max exponent magnitude
        max_exp_arg = np.max(exponent * log_base)
        min_exp_arg = np.min(exponent * log_base)
        # Overflow guard
        if max_exp_arg > exp_threshold:
            return None
        # Avoid extremely small values:
        if min_exp_arg < -1000:
            pass
        
        numerator = (1-2*q0)**2 * (z_eval+1)**(2*(j0-1)/(2*q0-1)) + 2*(z_eval+1)**3 * (j0 - q0*(2*q0+1))
        # h  = H0 * np.sqrt ( c1 * (1+z_eval)**3 + (1 + c1) * (1 + z_eval)**(3*eps) )

        ratio = numerator / denominator

        if not np.all(np.isfinite(ratio)):
            return None

        ratio = np.maximum(ratio, 1e-300)
        
        h = H0 * np.sqrt(ratio)
        Om = Om0 * denominator / ( -(1-2*q0)**2 * (z_eval+1)**(exponent-3) - 2*j0 + 2*q0*(2*q0+1) )

        if not np.all(np.isfinite(Om)):
            return None

        Om = np.maximum(Om, 1e-300)

        # try:
        #     return h, Om
        # except Exception:
        #     return None
        return h, Om

    else:
        raise ValueError(f"No model named {model}. Please specify 'model_1', 'model_2', or 'model_3'.")
    
def background_cpl(theta, z_eval):
    H0, Om0, w0, wa = theta
    
    if not np.all(np.isfinite(theta)):
        return None
    
    if Om0 <= 0 or Om0 >= 1:
        return None

    z_eval = np.asarray(z_eval)

    rho_de_normalised = ((1 + z_eval)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z_eval / (1 + z_eval))    )

    E_squared = Om0 * (1 + z_eval)**3 + (1 - Om0) * rho_de_normalised

    if np.any(E_squared <= 0) or np.any(~np.isfinite(E_squared)):
        return None

    H = H0 * np.sqrt(E_squared)

    Om = Om0 * (1 + z_eval)**3 / E_squared

    return H, Om