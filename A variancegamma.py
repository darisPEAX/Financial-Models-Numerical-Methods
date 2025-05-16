from FMNM.Parameters import Option_param
from FMNM.Processes import Merton_process, VG_process
from FMNM.Merton_pricer import Merton_pricer
from FMNM.VG_pricer import VG_pricer
from FMNM.utils import implied_volatility
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

import numpy as np
import pandas as pd
import time
import scipy as scp
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.optimize as scpo
import os
import warnings

warnings.filterwarnings("ignore")


r = 0.05
T = 1 / 252


data_1dte = pd.read_csv("data/data_1dte.csv")
CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)




# ——— Black-Scholes pricer for fallback ———
def bs_price(S, K, T, r, vol):
    d1 = (np.log(S/K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return np.exp(-r*T)*(S*norm.cdf(d1) - K*norm.cdf(d2))

# ——— safe IV (Newton → bisection) ———
def safe_iv(price, S0, K, T, r, iv_fun):
    try:
        return iv_fun(price, S0, K, T, r)
    except Exception:
        f = lambda v: bs_price(S0, K, T, r, v) - price
        try:
            return brentq(f, 1e-6, 5.0)
        except Exception:
            return np.nan
        

def train_VG(strikes, prices, spreads, r, S0, T, market_ivs):
    def f_VG(x, sig, lam, muJ, sigJ):
        VG_param = VG_process(r=r, sig=sig, lam=lam, muJ=muJ, sigJ=sigJ)
        opt_param = Option_param(S0=S0, K=x, T=T, v0=0.04, exercise="European", payoff="call")
        VG = VG_pricer(opt_param, VG_param)
        return VG

    def obj_fun(params):
        # model = f_Mert(strikes, params[0], params[1], params[2], params[3])
        model_ivs = [f_VG(strike, params[0], params[1], params[2], params[3]).IV_Lewis() for strike in strikes]
        # print(model_ivs)
        # model_prices = model.closed_formula()
        # model_ivs = np.array([implied_volatility(p, S0, k, T, r) for p, k in zip(model_prices, strikes)])
        return np.mean((market_ivs - model_ivs)**2)

    init_vals = [0.2, 1, -0.5, 0.2]
    bounds = ([0, 0, -np.inf, 0], [2, np.inf, 5, 5])
    # params_Mert = scpo.curve_fit(f_Mert, strikes, prices, p0=init_vals, bounds=bounds, sigma=spreads)
    # return params_Mert[0]

    # bounds = [(0, 2), (0, np.inf), (-np.inf, 5), (0, 5)]
    params_Mert = scpo.least_squares(obj_fun, x0=init_vals, bounds=bounds, method='trf')
    return params_Mert.x

def get_Merton_pricer(params, strikes, r, S0, T):
    Merton_param = Merton_process(r=r, sig=params[0], lam=params[1], muJ=params[2], sigJ=params[3])
    opt_param = Option_param(S0=S0, K=strikes, T=T, v0=0.04, exercise="European", payoff="call")
    Mert = Merton_pricer(opt_param, Merton_param)
    return Mert

def calibrate_model(dataframe, T, model="Merton", disp=False):
    MSE_list = []
    params_list = []
    for exdate in dataframe.exdate.unique():
        print("Doing date: ", exdate)
        option_type_exdate = dataframe[dataframe.exdate == exdate]
        sort_idx = np.argsort(option_type_exdate.Strike.values)
        option_type_exdate = option_type_exdate.iloc[sort_idx]
        option_type_exdate = option_type_exdate[option_type_exdate.IV.notna()]
        strikes = option_type_exdate.Strike.values
        prices = option_type_exdate.Midpoint.values
        spreads = option_type_exdate.Spread.values
        S0 = option_type_exdate.Close.values[0]
        IV_actual = option_type_exdate.IV.values

        print("Training Model")
        start_time = time.time()
        params_Mert = train_Merton(strikes, prices, spreads, r, S0, T, IV_actual)
        Mert = get_Merton_pricer(params_Mert, strikes, r, S0, T)
        prices_model = Mert.closed_formula()
        IV_model = [implied_volatility(prices_model[i], S0, strikes[i], T, r, disp=False, method="brent") for i in range(len(strikes))]
        MSE = np.mean((IV_model - IV_actual)**2)
        MSE_list.append(MSE)
        if disp:
            print("exdate: ", exdate)
            print("date: ", set(option_type_exdate.date.values))
            print("MSE: ", MSE)
            print("Training time: ", time.time() - start_time, "seconds")
            plt.figure(figsize=(10,6))
            plt.scatter(strikes, IV_actual, label='Actual IV', s=20, alpha=0.3)
            plt.plot(strikes, IV_model, label='Merton IV')
            plt.xlabel('Strike Price')
            plt.ylabel('Implied Volatility')
            plt.title('Implied Volatility Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()
    print(f"Results for {model}: {np.mean(MSE_list):.5f} ({np.std(MSE_list):.5f})")
    return MSE_list, params_list

MSE_Merton_1 = calibrate_model(CALL_1dte, T=1/252, model="Merton", disp=True)
# MSE_Merton_7 = calibrate_model(CALL_7dte, T=5/252, model="Merton")
# MSE_Merton_30 = calibrate_model(CALL_30dte, T=22/252, model="Merton")