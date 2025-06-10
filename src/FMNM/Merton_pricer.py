#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:47:49 2019

@author: cantaro86
"""

from scipy import sparse
from scipy.sparse.linalg import splu
from time import time
import numpy as np
import scipy as scp
import scipy.stats as ss
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
from FMNM.BS_pricer import BS_pricer
from math import factorial
from FMNM.CF import cf_mert
from FMNM.probabilities import Q1, Q2
from functools import partial
from FMNM.FFT import fft_Lewis, IV_from_Lewis


class Merton_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme

        0 = dV/dt + (r -(1/2)sig^2 -m) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type Merton_process. It contains (r, sig, lam, muJ, sigJ) i.e.
        interest rate, diffusion coefficient, jump activity and jump distribution params

        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price,
        strike, maturity in years
        """
        self.r = Process_info.r  # interest rate
        self.sig = Process_info.sig  # diffusion coefficient
        self.lam = Process_info.lam  # jump activity
        self.muJ = Process_info.muJ  # jump mean
        self.sigJ = Process_info.sigJ  # jump std
        self.exp_RV = Process_info.exp_RV  # function to generate exponential Merton Random Variables

        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def closed_formula(self):
        """
        Merton closed formula.
        """

        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        lam2 = self.lam * np.exp(self.muJ + (self.sigJ**2) / 2)

        tot = 0
        for i in range(18):
            tot += (np.exp(-lam2 * self.T) * (lam2 * self.T) ** i / factorial(i)) * BS_pricer.BlackScholes(
                self.payoff,
                self.S0,
                self.K,
                self.T,
                self.r - m + i * (self.muJ + 0.5 * self.sigJ**2) / self.T,
                np.sqrt(self.sig**2 + (i * self.sigJ**2) / self.T),
            )
        return tot

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_Mert, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_Mert, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_Mert, np.inf)) - self.S0 * (
                1 - Q1(k, cf_Mert, np.inf)
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
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_Mert, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_Mert, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_Mert)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        """
        Merton Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        V = scp.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T), axis=0)

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)), elapsed
            else:
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T))
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V

    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time()

        Nspace = steps[0]
        Ntime = steps[1]

        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)

        dev_X = np.sqrt(self.lam * self.sigJ**2 + self.lam * self.muJ**2)

        dx = (x_max - x_min) / (Nspace - 1)
        extraP = int(np.floor(5 * dev_X / dx))  # extra points beyond the B.C.
        x = np.linspace(x_min - extraP * dx, x_max + extraP * dx, Nspace + 2 * extraP)  # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)  # time discretization

        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace - 2)
        V = np.zeros((Nspace + 2 * extraP, Ntime))  # grid initialization

        if self.payoff == "call":
            V[:, -1] = Payoff  # terminal conditions
            V[-extraP - 1 :, :] = np.exp(x[-extraP - 1 :]).reshape(extraP + 1, 1) * np.ones(
                (extraP + 1, Ntime)
            ) - self.K * np.exp(-self.r * t[::-1]) * np.ones(
                (extraP + 1, Ntime)
            )  # boundary condition
            V[: extraP + 1, :] = 0
        else:
            V[:, -1] = Payoff
            V[-extraP - 1 :, :] = 0
            V[: extraP + 1, :] = self.K * np.exp(-self.r * t[::-1]) * np.ones((extraP + 1, Ntime))

        cdf = ss.norm.cdf(
            [np.linspace(-(extraP + 1 + 0.5) * dx, (extraP + 1 + 0.5) * dx, 2 * (extraP + 2))],
            loc=self.muJ,
            scale=self.sigJ,
        )[0]
        nu = self.lam * (cdf[1:] - cdf[:-1])

        lam_appr = sum(nu)
        m_appr = np.array([np.exp(i * dx) - 1 for i in range(-(extraP + 1), extraP + 2)]) @ nu

        sig2 = self.sig**2
        dxx = dx**2
        a = (dt / 2) * ((self.r - m_appr - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r + lam_appr)
        c = -(dt / 2) * ((self.r - m_appr - 0.5 * sig2) / dx + sig2 / dxx)

        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)
        if self.exercise == "European":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="fft"
                )
                V[extraP + 1 : -extraP - 1, i] = DD.solve(V_jump - offset)
        elif self.exercise == "American":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="fft"
                )
                V[extraP + 1 : -extraP - 1, i] = np.maximum(DD.solve(V_jump - offset), Payoff[extraP + 1 : -extraP - 1])

        X0 = np.log(self.S0)  # current log-price
        self.S_vec = np.exp(x[extraP + 1 : -extraP - 1])  # vector of S
        self.price = np.interp(X0, x, V[:, 0])
        self.price_vec = V[extraP + 1 : -extraP - 1, 0]
        self.mesh = V[extraP + 1 : -extraP - 1, :]

        if Time is True:
            elapsed = time() - t_init
            return self.price, elapsed
        else:
            return self.price

    def plot(self, axis=None):
        if type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray:
            self.PIDE_price((5000, 4000))

        plt.plot(self.S_vec, self.payoff_f(self.S_vec), color="blue", label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color="red", label="Merton curve")
        if type(axis) == list:
            plt.axis(axis)
        plt.xlabel("S")
        plt.ylabel("price")
        plt.title("Merton price")
        plt.legend(loc="upper left")
        plt.show()

    def mesh_plt(self):
        if type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray:
            self.PIDE_price((7000, 5000))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(np.linspace(0, self.T, self.mesh.shape[1]), self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title("Merton price surface")
        ax.set_xlabel("S")
        ax.set_ylabel("t")
        ax.set_zlabel("V")
        ax.view_init(30, -100)  # this function rotates the 3d plot
        plt.show()

    def delta_one(self, method="closed_formula", h_percent=0.01):
        """
        Calculate the option delta using finite difference method with central difference.
        
        Parameters:
        -----------
        method : str
            Method to calculate option prices. Options: "closed_formula", "Fourier_inversion", "PIDE_price"
        h_percent : float
            Percentage of stock price to use as step size
            
        Returns:
        --------
        float
            The delta of the option
        """
        h = self.S0 * h_percent
        
        # Store original S0
        original_S0 = self.S0
        
        # Calculate price at S0 + h
        self.S0 = original_S0 + h
        if method == "closed_formula":
            price_up = self.closed_formula()
        elif method == "Fourier_inversion":
            price_up = self.Fourier_inversion()
        elif method == "PIDE_price":
            price_up = self.PIDE_price((5000, 4000))
        else:
            raise ValueError("Invalid method. Choose 'closed_formula', 'Fourier_inversion' or 'PIDE_price'")
            
        # Calculate price at S0 - h
        self.S0 = original_S0 - h
        if method == "closed_formula":
            price_down = self.closed_formula()
        elif method == "Fourier_inversion":
            price_down = self.Fourier_inversion()
        elif method == "PIDE_price":
            price_down = self.PIDE_price((5000, 4000))
        
        # Restore original S0
        self.S0 = original_S0
        
        # Calculate delta using central difference
        delta = (price_up - price_down) / (2 * h)
        
        return delta
    
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

    def theta(self, x, h=None):
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
        