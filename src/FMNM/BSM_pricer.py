#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
from FMNM.CF import cf_BSM, cf_edgeworth, cf_standardized, cf_logS
from FMNM.probabilities import Q1, Q2
from FMNM.FFT  import fft_Lewis, IV_from_Lewis


class BSM_pricer:
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
        cf = partial(cf_logS,
                     tau=self.T,
                     sigma=self.sigma,
                     beta_tilde=self.beta_tilde,
                     rho=self.rho,
                     alpha_Q=self.alpha_Q,
                     delta_tilde=self.delta_tilde,
                     eta=self.eta)

        if self.payoff == "call":
            return ( self.S0 * Q1(k, cf, np.inf)
                   - self.K  * np.exp(-self.r*self.T) * Q2(k, cf, np.inf) )
        else:  # put via put-call parity
            return ( self.K*np.exp(-self.r*self.T)*(1-Q2(k, cf, np.inf))
                   - self.S0*(1-Q1(k, cf, np.inf)) )

    def FFT(self, K):
        K = np.array(K)
        cf = partial(cf_logS,
                     tau=self.T,
                     sigma=self.sigma,
                     beta_tilde=self.beta_tilde,
                     rho=self.rho,
                     alpha_Q=self.alpha_Q,
                     delta_tilde=self.delta_tilde,
                     eta=self.eta)
        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf, interp="cubic")
        else:
            # put by parity
            base = fft_Lewis(K, self.S0, self.r, self.T, cf, interp="cubic")
            return base - self.S0 + K*np.exp(-self.r*self.T)

    def IV_Lewis(self):
        cf = partial(cf_logS,
                     tau=self.T,
                     sigma=self.sigma,
                     beta_tilde=self.beta_tilde,
                     rho=self.rho,
                     alpha_Q=self.alpha_Q,
                     delta_tilde=self.delta_tilde,
                     eta=self.eta)

        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf)
        else:
            raise NotImplementedError("Implied vol for puts not implemented")

    def MC(self, *args, **kwargs):
        raise NotImplementedError("Monte Carlo not implemented for Edgeworth_pricer")
