from FMNM.Parameters import Option_param
from FMNM.Processes import Bates_process
from FMNM.Bates_pricer import Bates_pricer
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

data_1dte_2023 = pd.read_csv("data/data_1dte_2023.csv")
CALL_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "P"].reset_index(drop=True)

data_2dte_2023 = pd.read_csv("data/data_2dte_2023.csv")
CALL_2dte_2023 = data_2dte_2023[data_2dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_2dte_2023 = data_2dte_2023[data_2dte_2023.cp_flag == "P"].reset_index(drop=True)


data_7dte = pd.read_csv("data/data_7dte.csv")
CALL_7dte = data_7dte[data_7dte.cp_flag == "C"].reset_index(drop=True)
PUT_7dte = data_7dte[data_7dte.cp_flag == "P"].reset_index(drop=True)

data_30dte = pd.read_csv("data/data_30dte.csv")
CALL_30dte = data_30dte[data_30dte.cp_flag == "C"].reset_index(drop=True)
PUT_30dte = data_30dte[data_30dte.cp_flag == "P"].reset_index(drop=True)


import py_vollib.black_scholes_merton.implied_volatility
import py_vollib_vectorized


def train_Bates(strikes, prices, spreads, r, S0, T, market_ivs):
    def f_Bates(x, sig, theta, kappa, rho, lam, muJ, sigJ):
        Bates_param = Bates_process(mu=r, sigma=sig, theta=theta, kappa=kappa, rho=rho, lambda_j=lam, mu_j=muJ, sigma_j=sigJ)
        opt_param = Option_param(S0=S0, K=x, T=T, v0=0.04, exercise="European", payoff="call")
        Bates = Bates_pricer(opt_param, Bates_param)
        return Bates

    def obj_fun(params):
        # model_ivs = [f_Mert(strike, params[0], params[1], params[2], params[3]).IV_Lewis() for strike in strikes]

        model = f_Bates(strikes, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
        model_prices = model.FFT(strikes)
        flag = ['c' for _ in range(len(strikes))]
        t = pd.Series([T for _ in range(len(strikes))])
        model_ivs = py_vollib_vectorized.vectorized_implied_volatility(model_prices, S0, strikes, t, r, flag, q=0, return_as='numpy')
        # model_ivs = np.array([implied_volatility(p, S0, k, T, r) for p, k in zip(model_prices, strikes)])
        return np.mean((market_ivs - model_ivs)**2)

    # sig, theta, kappa, rho, lam, muJ, sigJ
    init_vals = [0.5, 0.05, 1.5, -0.7, 0.1, -0.05, 0.2]
    # bounds = ([0, 0, -np.inf, 0], [2, np.inf, 5, 5])
    bounds = ([0.01, 0.001, 0.1, -0.99, 0.001, -0.5, 0.01], [5, 0.5, 10, 0.99, 1, 0.5, 2])
    # params_Mert = scpo.curve_fit(f_Mert, strikes, prices, p0=init_vals, bounds=bounds, sigma=spreads)
    # return params_Mert[0]

    # bounds = [(0, 2), (0, np.inf), (-np.inf, 5), (0, 5)]
    params_Mert = scpo.least_squares(obj_fun, x0=init_vals, bounds=bounds, method='trf')
    return params_Mert.x

def get_Bates_pricer(params, strikes, r, S0, T):
    Bates_param = Bates_process(mu=r, sigma=params[0], theta=params[1], kappa=params[2], rho=params[3], lambda_j=params[4], mu_j=params[5], sigma_j=params[6])
    opt_param = Option_param(S0=S0, K=strikes, T=T, v0=0.04, exercise="European", payoff="call")
    Bates = Bates_pricer(opt_param, Bates_param)
    return Bates

def calibrate_model(dataframe, T, model="Bates", disp=False):
    MSE_list = []
    MSE_deep_itm_list = []
    MSE_mid_itm_list = []
    MSE_near_itm_list = []
    MSE_atm_list = []
    MSE_near_otm_list = []
    MSE_mid_otm_list = []
    MSE_deep_otm_list = []
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
        params_Bates = train_Bates(strikes, prices, spreads, r, S0, T, IV_actual)
        Bates = get_Bates_pricer(params_Bates, strikes, r, S0, T)
        prices_model = Bates.FFT(strikes)
        flag = ['c' for _ in range(len(strikes))]
        t = pd.Series([T for _ in range(len(strikes))])
        IV_model = py_vollib_vectorized.vectorized_implied_volatility(prices_model, S0, strikes, t, r, flag, q=0, return_as='numpy')
        # Calculate MSE for all strikes
        MSE = np.mean((IV_model - IV_actual)**2)
        MSE_list.append(MSE)
        
        # Calculate MSE
        moneyness = strikes/S0
        ranges = [
            (0, 0.85, 'deep_itm'),
            (0.85, 0.95, 'mid_itm'),
            (0.95, 0.99, 'near_itm'),
            (0.99, 1.01, 'atm'),
            (1.01, 1.05, 'near_otm'),
            (1.05, 1.15, 'mid_otm'),
            (1.15, 2, 'deep_otm')
        ]
        for lower, upper, name in ranges:
            mask = (moneyness >= lower) & (moneyness < upper)
            mse = np.mean((np.array(IV_model)[mask] - np.array(IV_actual)[mask])**2)
            locals()[f'MSE_{name}_list'].append(mse)

        if disp:
            print("exdate: ", exdate)
            print("date: ", set(option_type_exdate.date.values))
            print("MSE: ", MSE)
            print("Training time: ", time.time() - start_time, "seconds")
            plt.figure(figsize=(10,6))
            plt.scatter(strikes, IV_actual, label='Actual IV', s=20, alpha=0.3)
            plt.plot(strikes, IV_model, label='Bates IV')
            plt.xlabel('Strike Price')
            plt.ylabel('Implied Volatility')
            plt.title('Implied Volatility Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()
    print(f"Results for {model}: {np.mean(MSE_list):.5f} ({np.std(MSE_list):.5f})")
    MSE_dict = {
        "MSE_list": MSE_list,
        "MSE_deep_itm_list": MSE_deep_itm_list,
        "MSE_mid_itm_list": MSE_mid_itm_list,
        "MSE_near_itm_list": MSE_near_itm_list,
        "MSE_atm_list": MSE_atm_list,
        "MSE_near_otm_list": MSE_near_otm_list,
        "MSE_mid_otm_list": MSE_mid_otm_list,
        "MSE_deep_otm_list": MSE_deep_otm_list
    }
    return MSE_dict, params_list

# MSE_Merton_1, params_Merton_1 = calibrate_model(CALL_1dte, T=1/252, model="Merton")
# print(MSE_Merton_1["MSE_list"])


# MSE_Merton_7 = calibrate_model(CALL_7dte, T=5/252, model="Merton")
# MSE_Merton_30 = calibrate_model(CALL_30dte, T=22/252, model="Merton")


# MSE_Bates_1, params_Bates_1 = calibrate_model(CALL_1dte_2023, T=1/252, model="Bates")
# pd.DataFrame(pd.DataFrame(MSE_Bates_1)).to_csv("data/MSE_Bates_1.csv", index=False)

# MSE_Bates_7, params_Bates_7 = calibrate_model(CALL_7dte, T=5/252, model="Bates")
# pd.DataFrame(pd.DataFrame(MSE_Bates_7)).to_csv("data/MSE_Bates_7.csv", index=False)

# MSE_Bates_30, params_Bates_30 = calibrate_model(CALL_30dte, T=22/252, model="Bates")
# pd.DataFrame(pd.DataFrame(MSE_Bates_30)).to_csv("data/MSE_Bates_30.csv", index=False)



from hedging import simulate_hedging


def greeks_function(strikes, prices, spreads, S0, IV_actual):
    params_Bates = train_Bates(strikes, prices, spreads, r, S0, T, IV_actual)
    Bates = get_Bates_pricer(params_Bates, strikes, r, S0, T)
    deltas = Bates.delta(strikes)
    gammas = Bates.gamma(strikes)
    thetas = Bates.theta_greek(strikes)
    return deltas, gammas, thetas

delta_pnl_list, delta_gamma_pnl_list, delta_gamma_theta_pnl_list = simulate_hedging(CALL_1dte_2023, CALL_2dte_2023, greeks_function)
print("delta_pnl_list: ", delta_pnl_list)
print("delta_gamma_pnl_list: ", delta_gamma_pnl_list)
print("delta_gamma_theta_pnl_list: ", delta_gamma_theta_pnl_list)
print("----------- Delta PnL -----------")
print("Mean PnL: ", np.mean(delta_pnl_list))
print("Std PnL: ", np.std(delta_pnl_list))
print("----------- Delta Gamma PnL -----------")
print("Mean PnL: ", np.mean(delta_gamma_pnl_list))
print("Std PnL: ", np.std(delta_gamma_pnl_list))
print("----------- Delta Gamma Theta PnL -----------")
print("Mean PnL: ", np.mean(delta_gamma_theta_pnl_list))
print("Std PnL: ", np.std(delta_gamma_theta_pnl_list))


pd.DataFrame(pd.DataFrame(
    {
        "delta_pnl_list": delta_pnl_list,
        "delta_gamma_pnl_list": delta_gamma_pnl_list,
        "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
    }
    )).to_csv("data/delta_pnl_list_2023_bates.csv", index=False)