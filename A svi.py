from FMNM.Parameters import Option_param
from FMNM.Processes import Merton_process
from FMNM.Merton_pricer import Merton_pricer
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
data_1dte = data_1dte[data_1dte.Close.notna()]
CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_7dte = pd.read_csv("data/data_7dte.csv")
data_7dte = data_7dte[data_7dte.Close.notna()]
CALL_7dte = data_7dte[data_7dte.cp_flag == "C"].reset_index(drop=True)
PUT_7dte = data_7dte[data_7dte.cp_flag == "P"].reset_index(drop=True)

data_30dte = pd.read_csv("data/data_30dte.csv")
data_30dte = data_30dte[data_30dte.Close.notna()]
CALL_30dte = data_30dte[data_30dte.cp_flag == "C"].reset_index(drop=True)
PUT_30dte = data_30dte[data_30dte.cp_flag == "P"].reset_index(drop=True)




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
        
def raw_svi(x, a, b, sigma, rho, m):
    return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))

# SVI implied volatility function (vol = sqrt(total variance / T))
def svi_vol(x, a, b, sigma, rho, m):
    raw = raw_svi(x, a, b, sigma, rho, m)
    # clip to avoid negative total variance -> no NaNs
    raw = np.maximum(raw, 1e-6)
    return np.sqrt(raw / T)

def calibrate_model(dataframe, T, model="Merton", disp=False):
    MSE_list = []
    params_list = []
    for exdate in dataframe.exdate.unique():
        if disp == True:
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

        log_moneyness = np.log(strikes/S0)

        # --- vectorized residuals with weighting ---
        def residuals(params):
            vols   = svi_vol(log_moneyness, *params)
            weights = np.sqrt(np.maximum(IV_actual, 1e-4))
            return (vols - IV_actual) / weights

        # Multi-start least-squares calibration
        starting_points = [
            [0.04, 0.04, 0.2, -0.7, 0.0],
            [0.0, 0.1, 0.15, -0.5, 0.1],
            [0.05, 0.2, 0.1, -0.8, -0.1],
            [-0.05, 0.15, 0.3, -0.4, 0.05],
            [0.02, 0.3, 0.25, -0.6, -0.05]
        ]

        # Parameter bounds: a ∈ [-0.1,0.1], b ≥ 0, sigma > 0, rho ∈ (-1,1), m ∈ [-1,1]
        bounds_lower = [-0.1, 0.0, 1e-3, -0.999, -1.0]
        bounds_upper = [ 0.1, 1.0,   5.0,  0.999,  1.0]

        best_result = None
        for x0 in starting_points:
            res = least_squares(
                residuals,
                x0,
                bounds=(bounds_lower, bounds_upper),
                method='trf',
                loss='soft_l1',      # robust loss
                x_scale='jac',       # automatic scaling
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9
            )
            if best_result is None or res.cost < best_result.cost:
                best_result = res
        result = best_result
        IV_model = [svi_vol(x,*result.x) for x in log_moneyness]

        params_list.append(result.x)
        MSE = np.mean((IV_model - IV_actual)**2)
        MSE_list.append(MSE)
        if disp == True:
            print("exdate: ", exdate)
            print("date: ", set(option_type_exdate.date.values))
            print("MSE: ", MSE)
            plt.figure(figsize=(10,6))
            plt.scatter(strikes, IV_actual, label='Actual IV', s=20, alpha=0.3)
            plt.plot(strikes, IV_model, label='Merton IV')
            plt.xlabel('Strike Price')
            plt.ylabel('Implied Volatility')
            plt.title('Implied Volatility Comparison')
            plt.legend()
            plt.grid(True)
    print(f"Results for {model}: {np.mean(MSE_list):.5f} ({np.std(MSE_list):.5f})")
    return MSE_list, params_list


MSE_SVI_1, params_list_1 = calibrate_model(CALL_1dte, T=1/252, model="SVI")
MSE_SVI_7, params_list_7 = calibrate_model(CALL_7dte, T=5/252, model="SVI")
MSE_SVI_30, params_list_30 = calibrate_model(CALL_30dte, T=22/252, model="SVI")

def out_of_sample_test(params_list, dataframe, T, model="SVI", disp=False):
    MSE_list = []
    for index in range(1,len(dataframe.exdate.unique())):
        exdate = dataframe.exdate.unique()[index]
        option_type_exdate = dataframe[dataframe.exdate == exdate]
        sort_idx = np.argsort(option_type_exdate.Strike.values)
        option_type_exdate = option_type_exdate.iloc[sort_idx]
        option_type_exdate = option_type_exdate[option_type_exdate.IV.notna()]
        strikes = option_type_exdate.Strike.values
        prices = option_type_exdate.Midpoint.values
        spreads = option_type_exdate.Spread.values
        S0 = option_type_exdate.Close.values[0]
        IV_actual = option_type_exdate.IV.values

        moneyness = strikes / S0
        beta = params_list[index-1]
        a, b, sigma, rho, m = beta
        prices_model = []
        IV_model = []
        for i in range(len(strikes)):
            strike = strikes[i]
            vol = svi_vol(np.log(strike/S0), a, b, sigma, rho, m)
            prices_model.append(bs_price(S0, strike, T, r, vol))
            IV_model.append(vol)
        MSE = np.mean((IV_model - IV_actual)**2)
        MSE_list.append(MSE)
        if disp == True:
            print("exdate: ", exdate)
            print("date: ", set(option_type_exdate.date.values))
            print("MSE: ", MSE)
    return MSE_list


MSE_SVI_1_oos = out_of_sample_test(params_list_1, CALL_1dte, T=1/252, model="SVI")
MSE_SVI_7_oos = out_of_sample_test(params_list_7, CALL_7dte, T=5/252, model="SVI")
MSE_SVI_30_oos = out_of_sample_test(params_list_30, CALL_30dte, T=22/252, model="SVI")

pd.DataFrame({"MSE_SVI_1_oos": MSE_SVI_1_oos}).to_csv("data/MSE_SVI_1_oos.csv", index=False)
pd.DataFrame({"MSE_SVI_7_oos": MSE_SVI_7_oos}).to_csv("data/MSE_SVI_7_oos.csv", index=False)
pd.DataFrame({"MSE_SVI_30_oos": MSE_SVI_30_oos}).to_csv("data/MSE_SVI_30_oos.csv", index=False)


print(np.mean(MSE_SVI_1_oos), np.mean(MSE_SVI_7_oos), np.mean(MSE_SVI_30_oos))
print(np.std(MSE_SVI_1_oos), np.std(MSE_SVI_7_oos), np.std(MSE_SVI_30_oos))