#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
from FMNM.CF import cf_logS_with_jumps
from FMNM.probabilities import Q1, Q2
from FMNM.FFT  import fft_Lewis, IV_from_Lewis


class Edgeworth_pricer:
    """
    Fourier‐based pricer using the BSM characteristic function.
    """

    def __init__(self, Option_info, Process_info):
        # risk‐free rate
        self.r           = Process_info.r
        # BSM CF parameters
        self.sigma       = Process_info.sigma
        self.beta_tilde  = Process_info.beta_tilde
        self.rho         = Process_info.rho
        self.alpha_Q     = Process_info.alpha_Q
        self.delta_tilde = Process_info.delta_tilde
        self.eta         = Process_info.eta
        self.lambda_J    = Process_info.lambda_J
        self.mu_J        = Process_info.mu_J
        self.sigma_J     = Process_info.sigma_J


        # option data
        self.S0     = Option_info.S0
        self.K      = Option_info.K
        self.T      = Option_info.T
        self.payoff = Option_info.payoff  # "call" or "put"
        self.exercise = Option_info.exercise  # unused here

    def payoff_f(self, S):
        if self.payoff == "call":
            return np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            return np.maximum(self.K - S, 0)
        else:
            raise ValueError("payoff must be 'call' or 'put'")

    def Fourier_inversion(self):
        k = np.log(self.K / self.S0)
        cf = partial(cf_logS_with_jumps,
                     tau=self.T,
                     sigma=self.sigma,
                     beta_tilde=self.beta_tilde,
                     rho=self.rho,
                     alpha_Q=self.alpha_Q,
                     delta_tilde=self.delta_tilde,
                     eta=self.eta,
                     lambda_J=self.lambda_J,
                     mu_J=self.mu_J,
                     sigma_J=self.sigma_J)

        if self.payoff == "call":
            return ( self.S0 * Q1(k, cf, np.inf)
                   - self.K  * np.exp(-self.r*self.T) * Q2(k, cf, np.inf) )
        else:  # put via put-call parity
            return ( self.K*np.exp(-self.r*self.T)*(1-Q2(k, cf, np.inf))
                   - self.S0*(1-Q1(k, cf, np.inf)) )

    def FFT(self, K):
        K = np.array(K)
        cf = partial(cf_logS_with_jumps,
                     tau=self.T,
                     sigma=self.sigma,
                     beta_tilde=self.beta_tilde,
                     rho=self.rho,
                     alpha_Q=self.alpha_Q,
                     delta_tilde=self.delta_tilde,
                     eta=self.eta,
                     lambda_J=self.lambda_J,
                     mu_J=self.mu_J,
                     sigma_J=self.sigma_J)
        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf, interp="cubic")
        else:
            # put by parity
            base = fft_Lewis(K, self.S0, self.r, self.T, cf, interp="cubic")
            return base - self.S0 + K*np.exp(-self.r*self.T)

    def IV_Lewis(self):
        cf = partial(cf_logS_with_jumps,
                     tau=self.T,
                     sigma=self.sigma,
                     beta_tilde=self.beta_tilde,
                     rho=self.rho,
                     alpha_Q=self.alpha_Q,
                     delta_tilde=self.delta_tilde,
                     eta=self.eta,
                     lambda_J=self.lambda_J,
                     mu_J=self.mu_J,
                     sigma_J=self.sigma_J)

        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf)
        else:
            raise NotImplementedError("Implied vol for puts not implemented")

    def MC(self, *args, **kwargs):
        raise NotImplementedError("Monte Carlo not implemented for Edgeworth_pricer")
    
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
