from FMNM.Parameters import Option_param
from FMNM.Processes import Heston_process
from FMNM.Heston_pricer import Heston_pricer
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

import py_vollib.black_scholes_merton.implied_volatility
import py_vollib_vectorized

warnings.filterwarnings("ignore")


r = 0.05
T = 1 / 252

data_1dte = pd.read_csv("data/data_1dte_2023.csv")
data_1dte = data_1dte[data_1dte.Close.notna()]
CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_2dte = pd.read_csv("data/data_2dte_2023.csv")
data_2dte = data_2dte[data_2dte.Close.notna()]
CALL_2dte = data_2dte[data_2dte.cp_flag == "C"].reset_index(drop=True)
PUT_2dte = data_2dte[data_2dte.cp_flag == "P"].reset_index(drop=True)

# data_1dte = pd.read_csv("data/data_1dte.csv")
# data_1dte = data_1dte[data_1dte.Close.notna()]
# CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
# PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_7dte = pd.read_csv("data/data_7dte.csv")
data_7dte = data_7dte[data_7dte.Close.notna()]
CALL_7dte = data_7dte[data_7dte.cp_flag == "C"].reset_index(drop=True)
PUT_7dte = data_7dte[data_7dte.cp_flag == "P"].reset_index(drop=True)

data_30dte = pd.read_csv("data/data_30dte.csv")
data_30dte = data_30dte[data_30dte.Close.notna()]
CALL_30dte = data_30dte[data_30dte.cp_flag == "C"].reset_index(drop=True)
PUT_30dte = data_30dte[data_30dte.cp_flag == "P"].reset_index(drop=True)


def train_Heston(strikes, prices, spreads, r, S0, T, market_ivs):
    def f_Heston(x, rho, sigma, theta, kappa):
        Heston_param = Heston_process(mu=r, rho=rho, sigma=sigma, theta=theta, kappa=kappa)
        opt_param = Option_param(S0=S0, K=x, T=T, v0=0.04, exercise="European", payoff="call")
        Heston = Heston_pricer(opt_param, Heston_param)
        return Heston

    def obj_fun(params):
        # print('params: ', params)
        try:
            model = f_Heston(strikes, params[0], params[1], params[2], params[3])
            model_prices = model.FFT(strikes)
            # model_prices = np.maximum(model.FFT(strikes), 0.1) # TODO: Remove this
            
            # Check if any model prices are negative, NaN, or infinity
            if np.any(np.isnan(model_prices)) or np.any(np.isinf(model_prices)) or np.any(model_prices <= 0):
                return 1e10  # Return a large value instead of NaN
                
            flag = ['c' for _ in range(len(strikes))]
            t = pd.Series([T for _ in range(len(strikes))])
            model_ivs = py_vollib_vectorized.vectorized_implied_volatility(model_prices, S0, strikes, t, r, flag, q=0, return_as='numpy')
            # print(model_ivs)
            # Check if any implied volatilities are NaN or infinity
            if np.any(np.isnan(model_ivs)) or np.any(np.isinf(model_ivs)):
                return 1e10  # Return a large value instead of NaN
                
            return np.mean((market_ivs - model_ivs)**2)
        except Exception as e:
            print(f"Error in objective function with params {params}: {e}")
            return 1e10  # Return a large value on error

    # Use more conservative initial values and bounds
    init_vals = [-0.003, 3, 0.001, 2]  # [rho, sigma, theta, kappa]
    bounds = ([-0.999, 0.0,  0.0, 1e-6], 
              [0.999,  20.0, 10.0, 20.0])
    
    params_Heston = scpo.least_squares(obj_fun, x0=init_vals, bounds=bounds, method='trf')
    return params_Heston.x

def get_Heston_pricer(params, strikes, r, S0, T):
    Heston_param = Heston_process(mu=r, rho=params[0], sigma=params[1], theta=params[2], kappa=params[3])
    opt_param = Option_param(S0=S0, K=strikes, T=T, v0=0.04, exercise="European", payoff="call")
    Heston = Heston_pricer(opt_param, Heston_param)
    return Heston

def calibrate_model(dataframe, T, model="Heston", disp=False):
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
        params_Heston = train_Heston(strikes, prices, spreads, r, S0, T, IV_actual)
        Heston = get_Heston_pricer(params_Heston, strikes, r, S0, T)
        prices_model = Heston.FFT(strikes)
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
            plt.plot(strikes, IV_model, label='Heston IV')
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

MSE_Heston_1, params_Heston_1 = calibrate_model(CALL_1dte, T=1/252, model="Heston")
pd.DataFrame(pd.DataFrame(MSE_Heston_1)).to_csv("data/MSE_Heston_1.csv", index=False)

MSE_Heston_7, params_Heston_7 = calibrate_model(CALL_7dte, T=5/252, model="Heston")
pd.DataFrame(pd.DataFrame(MSE_Heston_7)).to_csv("data/MSE_Heston_7.csv", index=False)

MSE_Heston_30, params_Heston_30 = calibrate_model(CALL_30dte, T=22/252, model="Heston")
pd.DataFrame(pd.DataFrame(MSE_Heston_30)).to_csv("data/MSE_Heston_30.csv", index=False)



from hedging import simulate_hedging


def greeks_function(strikes, prices, spreads, S0, IV_actual):
    params_Heston = train_Heston(strikes, prices, spreads, r, S0, T, IV_actual)
    Heston = get_Heston_pricer(params_Heston, strikes, r, S0, T)
    deltas = Heston.delta(strikes)
    gammas = Heston.gamma(strikes)
    return deltas, gammas

delta_pnl_list, delta_gamma_pnl_list = simulate_hedging(CALL_1dte, CALL_2dte, greeks_function)
print(delta_pnl_list)
print("Mean PnL: ", np.mean(delta_pnl_list))
print("Std PnL: ", np.std(delta_pnl_list))
print("Mean Delta Gamma PnL: ", np.mean(delta_gamma_pnl_list))
print("Std Delta Gamma PnL: ", np.std(delta_gamma_pnl_list))