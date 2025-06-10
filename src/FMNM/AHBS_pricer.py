#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:18:39 2019

@author: cantaro86
"""

import numpy as np
import scipy as scp
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
import scipy.stats as ss
from FMNM.Solvers import Thomas
from FMNM.cython.solvers import SOR
from FMNM.CF import cf_normal
from FMNM.probabilities import Q1, Q2
from functools import partial
from FMNM.FFT import fft_Lewis, IV_from_Lewis


class AHBS_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference Black-Scholes PDE:
     df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """

    def __init__(self, Option_info, Process_info):
        """
        Option_info: of type Option_param. It contains (S0,K,T)
                i.e. current price, strike, maturity in years
        Process_info: of type Diffusion_process. It contains (r, mu, sig) i.e.
                interest rate, drift coefficient, diffusion coefficient
        """
        self.r = Process_info.r  # interest rate
        self.sig = Process_info.sig  # diffusion coefficient
        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years
        self.exp_RV = Process_info.exp_RV  # function to generate solution of GBM

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

        # optional polynomial vol surface: highest‑order first,
        # e.g. [a₂,a₁,a₀] → σ(K)=a₂K²+a₁K+a₀
        self.poly_coeffs = getattr(Process_info, 'poly_coeffs', None)
 
    def sigma_K(self, K):
        """Return polynomial σ(K), or constant self.sig if none supplied."""
        if self.poly_coeffs is None:
            return self.sig
        return np.polyval(self.poly_coeffs, K)

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    @staticmethod
    def BlackScholes(payoff="call", S0=100.0, K=100.0, T=1.0, r=0.1, sigma=0.2):
        """Black Scholes closed formula:
        payoff: call or put.
        S0: float.    initial stock/index level.
        K: float strike price.
        T: float maturity (in year fractions).
        r: float constant risk-free short rate.
        sigma: volatility factor in diffusion term."""

        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

        if payoff == "call":
            return S0 * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
        elif payoff == "put":
            return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    @staticmethod
    def vega(sigma, S0, K, T, r):
        """BS vega: derivative of the price with respect to the volatility"""
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        return S0 * np.sqrt(T) * ss.norm.pdf(d1)

    def closed_formula(self):
        """
        Black Scholes closed formula:
        """
        sig = self.sigma_K(self.K)
        d1 = (np.log(self.S0 / self.K) + (self.r + sig**2 / 2) * self.T) / (sig * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.r - sig**2 / 2) * self.T) / (sig * np.sqrt(self.T))

        if self.payoff == "call":
            return self.S0 * ss.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
        elif self.payoff == "put":
            return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)
        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_GBM, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_GBM, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_GBM, np.inf)) - self.S0 * (
                1 - Q1(k, cf_GBM, np.inf)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding
        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding
        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_GBM)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
