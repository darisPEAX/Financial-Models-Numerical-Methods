#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:13:10 2020

@author: cantaro86
"""

from time import time
import numpy as np
import scipy as scp
import scipy.stats as ss

from FMNM.CF import cf_Bates
from FMNM.cython.heston import Heston_paths
from FMNM.probabilities import Q1, Q2
from functools import partial
from FMNM.FFT import fft_Lewis, IV_from_Lewis


class Bates_pricer:
    """
    Class to price the options with the Heston model by:
    - Fourier-inversion.
    - Monte Carlo.
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type VG_process. It contains the interest rate r
        and the VG parameters (sigma, theta, kappa)

        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price,
        strike, maturity in years
        """
        self.r = Process_info.mu  # interest rate
        self.sigma = Process_info.sigma  # Heston parameter
        self.theta = Process_info.theta  # Heston parameter
        self.kappa = Process_info.kappa  # Heston parameter
        self.rho = Process_info.rho  # Heston parameter
        self.lambda_j = Process_info.lambda_j  # Bates parameter
        self.mu_j = Process_info.mu_j  # Bates parameter
        self.sigma_j = Process_info.sigma_j  # Bates parameter

        self.S0 = Option_info.S0  # current price
        self.v0 = Option_info.v0  # spot variance
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def MC(self, N, paths, Err=False, Time=False):
        """
        Heston Monte Carlo
        N = time steps
        paths = number of simulated paths
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T, _ = Heston_paths(
            N=N,
            paths=paths,
            T=self.T,
            S0=self.S0,
            v0=self.v0,
            mu=self.r,
            rho=self.rho,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            lambda_j=self.lambda_j,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
        )
        S_T = S_T.reshape((paths, 1))
        DiscountedPayoff = np.exp(-self.r * self.T) * self.payoff_f(S_T)
        V = scp.mean(DiscountedPayoff, axis=0)
        std_err = ss.sem(DiscountedPayoff)

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, std_err, elapsed
            else:
                return V, std_err
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        cf_H_b_good = partial(
            cf_Bates,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
            lambda_j=self.lambda_j,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
        )

        limit_max = 2000  # right limit in the integration

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_H_b_good, limit_max) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_H_b_good, limit_max
            )
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_H_b_good, limit_max)) - self.S0 * (
                1 - Q1(k, cf_H_b_good, limit_max)
            )
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_H_b_good = partial(
            cf_Bates,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
            lambda_j=self.lambda_j,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp="cubic")
                - self.S0
                + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_H_b_good = partial(
            cf_Bates,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
        )
        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_H_b_good)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def delta(self, x, h=None):
        """
        Compute delta numerically via central difference.
        method: one of "closed_formula", "Fourier", or "MC"
        h: step size for perturbation. If None, defaults to 1% of S0.
        """
        if h is None:
            h = 0.01 * self.S0

        # Store original S0
        S0_orig = self.S0

        # Perturb up
        self.S0 = S0_orig + h
        price_up = self.FFT(x)

        # Perturb down
        self.S0 = S0_orig - h
        price_down = self.FFT(x)
        
        # Restore original S0
        self.S0 = S0_orig

        # Central difference
        delta = (price_up - price_down) / (2 * h)
        return delta

    def gamma(self, x, h=None):
        """
        Compute gamma numerically via central difference.
        """
        if h is None:
            h = 0.01 * self.S0

        # Store original S0
        S0_orig = self.S0

        # Perturb up
        self.S0 = S0_orig + h
        delta_up = self.delta(x, h)

        # Perturb down
        self.S0 = S0_orig - h
        delta_down = self.delta(x, h)

        # Restore original S0
        self.S0 = S0_orig

        # Central difference
        gamma = (delta_up - delta_down) / (2 * h)
        return gamma

    def theta_greek(self, x, h=None):
        """
        Compute theta numerically via central difference.
        """
        if h is None:
            h = 0.01 * self.S0

        T_orig = self.T

        price_now = self.FFT(x)

        # Go forward in time by 1 day
        self.T = T_orig - 1.0/252
        price_future = self.FFT(x)
        
        # Restore original T
        self.T = T_orig

        # Central difference
        theta = (price_future - price_now)*252
        return theta