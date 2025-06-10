#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:57:19 2019

@author: cantaro86
"""

import numpy as np


def cf_normal(u, mu=1, sig=2):
    """
    Characteristic function of a Normal random variable
    """
    return np.exp(1j * u * mu - 0.5 * u**2 * sig**2)


def cf_gamma(u, a=1, b=2):
    """
    Characteristic function of a Gamma random variable
    - shape: a
    - scale: b
    """
    return (1 - b * u * 1j) ** (-a)


def cf_poisson(u, lam=1):
    """
    Characteristic function of a Poisson random variable
    - rate: lam
    """
    return np.exp(lam * (np.exp(1j * u) - 1))


def cf_mert(u, t=1, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5):
    """
    Characteristic function of a Merton random variable at time t
    mu: drift
    sig: diffusion coefficient
    lam: jump activity
    muJ: jump mean size
    sigJ: jump size standard deviation
    """
    return np.exp(
        t * (1j * u * mu - 0.5 * u**2 * sig**2 + lam * (np.exp(1j * u * muJ - 0.5 * u**2 * sigJ**2) - 1))
    )


def cf_VG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Variance Gamma random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Gamma process variance
    """
    return np.exp(t * (1j * mu * u - np.log(1 - 1j * theta * kappa * u + 0.5 * kappa * sigma**2 * u**2) / kappa))


def cf_NIG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Normal Inverse Gaussian random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Inverse Gaussian process variance
    """
    return np.exp(
        t * (1j * mu * u + 1 / kappa - np.sqrt(1 - 2j * theta * kappa * u + kappa * sigma**2 * u**2) / kappa)
    )


def cf_Heston(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed in the original paper of Heston (1993)
    """
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi + d) * t - 2 * np.log((1 - g1 * np.exp(d * t)) / (1 - g1)))
        + (v0 / sigma**2) * (xi + d) * (1 - np.exp(d * t)) / (1 - g1 * np.exp(d * t))
    )
    return cf


def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * t)) / (1 - g2 * np.exp(-d * t))
    )
    return cf

def cf_Bates(u, t, v0, mu, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j):
    heston_cf = cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho)
    
    # Jump component characteristic function (Merton jumps)
    jump_cf = np.exp(t * lambda_j * (np.exp(1j * u * mu_j - 0.5 * (sigma_j**2) * (u**2)) - 1))
    return heston_cf * jump_cf


def cf_BSM(u, tau, sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta):
    """
    u : The input variable of the characteristic function.
    tau : Time to maturity parameter.
    sigma : Volatility parameter.
    beta_tilde : Skewness adjustment parameter.
    rho : Correlation parameter.
    alpha_Q : Risk-neutral alpha parameter.
    delta_tilde : Adjusted delta parameter.
    eta : Kurtosis adjustment parameter.

    Returns: The value of the characteristic function evaluated at u.
    """
    mu_tilde = 0.04 - (sigma**2)/2

    # Compute the exponent term
    exponent_term = (1j * u * mu_tilde * np.sqrt(tau)) / sigma - (u**2) / 2
    
    # Third moment adjustment term
    term2 = -1j * (u**3 * beta_tilde * rho) / (2 * sigma) * np.sqrt(tau)
    
    # Second moment and other adjustments
    part3 = ((alpha_Q + delta_tilde) / (2 * sigma)) + (beta_tilde**2 / (4 * sigma**2))
    term3 = - (u**2) * part3 * tau
    
    # Fourth moment and higher adjustments
    inner_part = 4 * u**2 - rho**2 * u**2 * (3 * u**2 - 8)
    term4 = (1/24) * (beta_tilde**2 / sigma**2) * (u**2 * inner_part) * tau
    
    # Additional kurtosis term
    term5 = (eta / (6 * sigma)) * u**4 * tau
    
    # Combine all terms in the polynomial
    polynomial = 1 + term2 + term3 + term4 + term5
    
    # Compute the characteristic function value
    cf_value = np.exp(exponent_term) * polynomial
    
    return cf_value

def cf_edgeworth(u, tau,
                 sigma,        # B-S vol
                 beta_tilde,   # skew adj.
                 rho,          # correlation/mixing
                 alpha_Q,      # risk-neutral drift
                 delta_tilde,  # variance adj.
                 eta):         # kurtosis adj.
    """
    Edgeworth expanded CF of ln S_T.

    Params
    ------
    u           : scalar or array of freq
    tau         : time to expiry
    sigma       : B-S vol
    beta_tilde  : skewness adjustment
    rho         : mixing weight for skew
    alpha_Q     : risk-neutral drift (should equal r)
    delta_tilde : extra variance adjustment
    eta         : kurtosis adjustment

    Returns
    -------
    φ(u) = exp(iu c1 - .5 u^2 c2 + i u^3 c3/6 - u^4 c4/24)
    """
    r = 0.04

    # 1) first cumulant = drift×τ
    mu_tilde = r - 0.5*sigma**2
    c1 = mu_tilde * tau

    # 2) second cumulant = total var×τ
    c2 = (sigma**2 + delta_tilde) * tau

    # 3) third cumulant ≈ skew_coef × vol^3 × τ
    c3 = beta_tilde * rho * sigma**3 * tau

    # 4) fourth cumulant ≈ kurtosis_coef × vol^4 × τ
    c4 = eta * sigma**4 * tau

    return np.exp(
        1j*u*c1
        - 0.5*c2*(u**2)
        + 1j*(u**3)*c3/6
        -     (u**4)*c4/24
    )


def cf_standardized(u, tau,
                    sigma,           # σₜ (BS “instantaneous” vol)
                    beta_tilde,          # β̃ₜ (skew adjustment)
                    rho,           # ρₜ (mixing / correlation)
                    alpha_Q,          # αₜ^Q (risk‐neutral drift)
                    delta_tilde,          # δ̃ₜ (extra variance adj.)
                    eta):          # ηₜ (kurtosis adj.)
    # 1) standardized first cumulant:
    mu_tilde = alpha_Q - 0.5*sigma**2
    A  = mu_tilde * tau / (sigma * np.sqrt(tau))

    # 2) third‐moment adjustment
    term2 = -1j*(u**3 * beta_tilde * rho)/(2*sigma)*np.sqrt(tau)

    # 3) combined second/fourth/sixth‐moment adjustment
    part3 = (alpha_Q + delta_tilde)/(2*sigma) + (beta_tilde**2)/(4*sigma**2)
    term3 = -u**2 * part3 * tau

    # 4) extra 4th‐and‐6th‐moment “polynomial” term
    term4 = (1/24)*(beta_tilde**2/sigma**2)*u**2*(4*u**2 - rho**2*u**2*(3*u**2 - 8))*tau

    # 5) pure 6th‐moment adjustment
    term5 = (eta/(6*sigma))*u**4 * tau

    return np.exp(1j*u*A - 0.5*u**2) * (1 + term2 + term3 + term4 + term5)

def cf_logS(u, tau, sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta):
    # 1) compute the drift
    mu_tilde = alpha_Q - 0.5*sigma**2

    # 2) call the standardized CF at v = u*σ*√τ
    v = u * sigma * np.sqrt(tau)
    cf_Z = cf_standardized(v, tau, sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta)

    # 3) multiply by the drift‐term to get φ_{ln S}(u)
    return np.exp(1j*u*mu_tilde*tau) * cf_Z



def cf_with_gaussian_price_jumps(
    u, tau,
    sigma,           # σₜ
    beta_tilde,          # β̃ₜ
    rho,           # ρₜ
    alpha_Q,          # αₜ^Q
    delta_tilde,          # δ̃ₜ
    eta,           # ηₜ
    lambda_J,         # jump intensity λ^X_t
    mu_J, sigma_J     # Gaussian jump mean & stdev
):
    # —– 1) the paper’s baseline “standardized” CF up to 6th moments —–
    mu_tilde = alpha_Q - 0.5*sigma**2
    A  = mu_tilde*tau/(sigma*np.sqrt(tau))

    term2 = -1j*(u**3 * beta_tilde * rho)/(2*sigma)*np.sqrt(tau)
    part3 = (alpha_Q+delta_tilde)/(2*sigma) + beta_tilde**2/(4*sigma**2)
    term3 = -u**2 * part3 * tau
    term4 = (beta_tilde**2/sigma**2)/24 * u**2*(4*u**2 - rho**2*u**2*(3*u**2 - 8))*tau
    term5 = (eta/(6*sigma)) * u**4 * tau

    base_factor = np.exp(1j*u*A - 0.5*u**2)
    poly       = (1 + term2 + term3 + term4 + term5)

    # —– 2) closed-form Gaussian price-jump integral —–
    #    ∫(e^{i u x/(σ√τ)} – 1) N(μ_J,σ_J^2)(dx)
    jump_cf = np.exp(
        1j*u*mu_J/(sigma*np.sqrt(tau))
        - 0.5*u**2*(sigma_J**2/(sigma**2*tau))
    ) - 1

    # —– 3) assemble standardized CF with jumps —–
    # Eq.(4)+(6) ⇒ baseline×[ poly + τ*λ_J*jump_cf ]
    return base_factor * (poly + tau*lambda_J*jump_cf)

def cf_logS_with_jumps(u, tau,
                       sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta,
                       lambda_J, mu_J, sigma_J):
    """
    CF of ln S_T with Edgeworth moments + Gaussian price jumps.
    """
    # 1) physical drift term for ln S_T
    mu_tilde = alpha_Q - 0.5*sigma**2

    # 2) argument for the standardized CF
    v = u * sigma * np.sqrt(tau)

    # 3) standardized CF with jumps
    cf_Z = cf_with_gaussian_price_jumps(v, tau,
                                      sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta,
                                      lambda_J, mu_J, sigma_J)

    # 4) lift back to ln S_T
    return np.exp(1j*u*mu_tilde*tau) * cf_Z


# import numpy as np
# from scipy.integrate import quad, nquad

# # Example parameters (adjust based on data/estimates)
# sigma = 0.2
# beta_tilde = 0.1
# rho = -0.7
# alpha_Q = 0.05
# delta_tilde = 0.02
# eta = 0.01
# lambda_X = 0.1  # Price jump intensity
# lambda_X_sigma = 0.05  # Joint jump intensity
# lambda_sigma = 0.03  # Volatility jump intensity

# # Example jump distributions (mean and variance)
# mu_X = 0.0  # Mean of price jumps
# std_X = 0.1  # Std of price jumps
# mu_sigma = 0.0  # Mean of volatility jumps
# std_sigma = 0.05  # Std of volatility jumps

# r=0.05


# def cf_Edgeworth(u, tau, sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta, lambda_X, lambda_X_sigma, lambda_sigma):
#     # Exponential term
#     def integrand_1(x):
#         return (np.exp(x) - 1) * lambda_X * np.exp(-(x - mu_X)**2 / (2 * std_X**2)) / (std_X * np.sqrt(2 * np.pi))
#     integral_1, _ = nquad(integrand_1, [[-np.inf, np.inf]])
#     def integrand_2(x):
#         return (np.exp(x) - 1) * lambda_X_sigma * np.exp(-(x - mu_X)**2 / (2 * std_X**2)) / (std_X * np.sqrt(2 * np.pi)) # probably wrong
#     integral_2, _ = nquad(integrand_2, [[-np.inf, np.inf]])

#     mu_full = r - (sigma**2)/2 - integral_1 - integral_2
#     print('mu_full', mu_full)

#     exp_term = np.exp(1j * u * (mu_full / (sigma * np.sqrt(tau))) - (u**2)/2)
#     print('exp_term', exp_term)
    
#     # Polynomial terms
#     term1 = -1j * u**3 * (beta_tilde * rho) / (2 * sigma) * np.sqrt(tau)
#     term2 = -u**2 * ((alpha_Q + delta_tilde)/(2 * sigma) + beta_tilde**2/(4 * sigma**2)) * tau
#     term3 = (1/24) * (beta_tilde**2 / sigma**2) * u**2 * (4 * u**2 - rho**2 * u**2 * (3 * u**2 - 8)) * tau
#     term4 = (eta / (6 * sigma)) * u**4 * tau
    
#     # Idiosyncratic price jumps integral
#     def integrand_price(x):
#         return (np.exp(1j * u * x / (sigma * np.sqrt(tau))) - 1) * lambda_X * np.exp(-(x - mu_X)**2 / (2 * std_X**2)) / (std_X * np.sqrt(2 * np.pi))
#     integral_price, _ = nquad(integrand_price, [[-np.inf, np.inf]])
    
#     # Joint price/volatility jumps integral (simplified)
#     def integrand_joint(x, s, v):
#         exponent = -0.5 * u**2 * (s**2 / sigma**2 + 2 * x / sigma) * v
#         return (np.exp(1j * u * x / (sigma * np.sqrt(tau))) * np.exp(exponent) - 1) * lambda_X_sigma * np.exp(-(x**2 + s**2)/(2*(std_X**2 + std_sigma**2))) / (2 * np.pi * std_X * std_sigma)
#     integral_joint, _ = nquad(integrand_joint, [[-np.inf, np.inf], [-np.inf, np.inf], [0, 1]])
    
#     # Idiosyncratic volatility jumps integral
#     def integrand_vol(s, v):
#         exponent = -0.5 * u**2 * (s**2 / sigma**2) * v
#         return (np.exp(exponent) - 1) * lambda_sigma * np.exp(-(s - mu_sigma)**2 / (2 * std_sigma**2)) / (std_sigma * np.sqrt(2 * np.pi))
#     integral_vol, _ = nquad(integrand_vol, [[-np.inf, np.inf], [0, 1]])
    
#     # Combine all terms
#     total = exp_term * (1 + term1 + term2 + term3 + term4 + tau * (integral_price + integral_joint + integral_vol))
#     return total
