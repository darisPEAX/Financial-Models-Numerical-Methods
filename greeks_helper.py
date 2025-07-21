import numpy as np
import scipy.stats as ss

def delta_BS(IV, S0, K, T, r):
    """
    Compute delta numerically via central difference.
    """
    d1 = (np.log(S0/K) + (r + 0.5*IV**2)*T) / (IV*np.sqrt(T))
    delta = ss.norm.cdf(d1) - 1
    return delta

def gamma_BS(IV, S0, K, T, r):
    """
    Compute gamma numerically via central difference.
    """
    d1 = (np.log(S0/K) + (r + 0.5*IV**2)*T) / (IV*np.sqrt(T))
    gamma = ss.norm.pdf(d1) / (S0*IV*np.sqrt(T))
    return gamma

def theta_BS(IV, S0, K, T, r):
    """
    Compute theta numerically via central difference.
    """
    d1 = (np.log(S0/K) + (r + 0.5*IV**2)*T) / (IV*np.sqrt(T))
    theta = -S0*ss.norm.pdf(d1)*IV/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*ss.norm.cdf(d1)
    return theta