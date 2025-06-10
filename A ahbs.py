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


# data_1dte = pd.read_csv("data/data_1dte.csv")
# data_1dte = data_1dte[data_1dte.Close.notna()]
# CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
# PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_1dte = pd.read_csv("data/data_1dte_2023.csv")
data_1dte = data_1dte[data_1dte.Close.notna()]
CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_2dte = pd.read_csv("data/data_2dte_2023.csv")
data_2dte = data_2dte[data_2dte.Close.notna()]
CALL_2dte = data_2dte[data_2dte.cp_flag == "C"].reset_index(drop=True)
PUT_2dte = data_2dte[data_2dte.cp_flag == "P"].reset_index(drop=True)

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

        moneyness = strikes / S0
        X = np.column_stack([np.ones_like(moneyness), moneyness, moneyness**2, moneyness**3])
        beta = np.linalg.lstsq(X, IV_actual, rcond=None)[0]
        beta_0, beta_1, beta_2, beta_3 = beta
        params_list.append(beta)
        prices_model = []
        IV_model = []
        for i in range(len(strikes)):
            strike = strikes[i]
            vol = beta_0 + beta_1*moneyness[i] + beta_2*moneyness[i]**2 + beta_3*moneyness[i]**3
            prices_model.append(bs_price(S0, strike, T, r, vol))
            IV_model.append(vol)

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
            plt.show()
    print(f"Results for {model}: {np.mean(MSE_list):.5f} ({np.std(MSE_list):.5f})")
    return MSE_list, params_list


# MSE_AHBS_1, params_list_1 = calibrate_model(CALL_1dte, T=1/252, model="AHBS")
# MSE_AHBS_7, params_list_7 = calibrate_model(CALL_7dte, T=5/252, model="AHBS")
# MSE_AHBS_30, params_list_30 = calibrate_model(CALL_30dte, T=22/252, model="AHBS")



from hedging import simulate_hedging


def greeks_function(strikes, prices, spreads, S0, IV_actual):
    moneyness = strikes / S0
    X = np.column_stack([np.ones_like(moneyness), moneyness, moneyness**2, moneyness**3])
    beta = np.linalg.lstsq(X, IV_actual, rcond=None)[0]
    beta_0, beta_1, beta_2, beta_3 = beta
    IV_model = beta_0 + beta_1*moneyness + beta_2*moneyness**2 + beta_3*moneyness**3

    deltas = []
    gammas = []
    for i, k in enumerate(strikes):
        vol = IV_model[i]
        d1 = (np.log(S0/k) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        delta = norm.cdf(d1)
        deltas.append(delta)
        gamma = norm.pdf(d1) / (S0 * vol * np.sqrt(T))
        gammas.append(gamma)

    return deltas, gammas

delta_pnl_list, delta_gamma_pnl_list = simulate_hedging(CALL_1dte, CALL_2dte, greeks_function)
print(delta_pnl_list)
print("Mean PnL: ", np.mean(delta_pnl_list))
print("Std PnL: ", np.std(delta_pnl_list))
print("Mean Delta Gamma PnL: ", np.mean(delta_gamma_pnl_list))
print("Std Delta Gamma PnL: ", np.std(delta_gamma_pnl_list))







# def out_of_sample_test(params_list, dataframe, T, model="AHBS", disp=False):
#     MSE_list = []
#     for index in range(1,len(dataframe.exdate.unique())):
#         exdate = dataframe.exdate.unique()[index]
#         option_type_exdate = dataframe[dataframe.exdate == exdate]
#         sort_idx = np.argsort(option_type_exdate.Strike.values)
#         option_type_exdate = option_type_exdate.iloc[sort_idx]
#         option_type_exdate = option_type_exdate[option_type_exdate.IV.notna()]
#         strikes = option_type_exdate.Strike.values
#         prices = option_type_exdate.Midpoint.values
#         spreads = option_type_exdate.Spread.values
#         S0 = option_type_exdate.Close.values[0]
#         IV_actual = option_type_exdate.IV.values

#         moneyness = strikes / S0
#         X = np.column_stack([np.ones_like(moneyness), moneyness, moneyness**2, moneyness**3])
#         beta = params_list[index-1]
#         beta_0, beta_1, beta_2, beta_3 = beta
#         prices_model = []
#         IV_model = []
#         for i in range(len(strikes)):
#             strike = strikes[i]
#             vol = beta_0 + beta_1*moneyness[i] + beta_2*moneyness[i]**2 + beta_3*moneyness[i]**3
#             prices_model.append(bs_price(S0, strike, T, r, vol))
#             IV_model.append(vol)
#         MSE = np.mean((IV_model - IV_actual)**2)
#         MSE_list.append(MSE)
#         if disp == True:
#             print("exdate: ", exdate)
#             print("date: ", set(option_type_exdate.date.values))
#             print("MSE: ", MSE)
#     return MSE_list


# MSE_AHBS_1_oos = out_of_sample_test(params_list_1, CALL_1dte, T=1/252, model="AHBS")
# MSE_AHBS_7_oos = out_of_sample_test(params_list_7, CALL_7dte, T=5/252, model="AHBS")
# MSE_AHBS_30_oos = out_of_sample_test(params_list_30, CALL_30dte, T=22/252, model="AHBS")

# pd.DataFrame({"MSE_AHBS_1_oos": MSE_AHBS_1_oos}).to_csv("data/MSE_AHBS_1_oos.csv", index=False)
# pd.DataFrame({"MSE_AHBS_7_oos": MSE_AHBS_7_oos}).to_csv("data/MSE_AHBS_7_oos.csv", index=False)
# pd.DataFrame({"MSE_AHBS_30_oos": MSE_AHBS_30_oos}).to_csv("data/MSE_AHBS_30_oos.csv", index=False)

# print(np.mean(MSE_AHBS_1_oos), np.mean(MSE_AHBS_7_oos), np.mean(MSE_AHBS_30_oos))
# print(np.std(MSE_AHBS_1_oos), np.std(MSE_AHBS_7_oos), np.std(MSE_AHBS_30_oos))