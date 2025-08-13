from FMNM.Parameters import Option_param
from FMNM.Processes import NIG_process
from FMNM.NIG_pricer import NIG_pricer
from FMNM.utils import implied_volatility
from scipy.optimize import brentq, least_squares
from scipy.stats import norm
import py_vollib.black_scholes_merton.implied_volatility
import py_vollib_vectorized
from hedging import simulate_hedging
from greeks_helper import delta_BS, gamma_BS, theta_BS

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


data_1dte = pd.read_csv("data/source/data_1dte_2023_new.csv")
data_1dte = data_1dte[data_1dte.Close.notna()]
data_1dte = data_1dte[data_1dte.IV.notna()]
CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_2dte = pd.read_csv("data/source/data_2dte_2023_new.csv")
data_2dte = data_2dte[data_2dte.Close.notna()]
data_2dte = data_2dte[data_2dte.IV.notna()]
CALL_2dte = data_2dte[data_2dte.cp_flag == "C"].reset_index(drop=True)
PUT_2dte = data_2dte[data_2dte.cp_flag == "P"].reset_index(drop=True)

data_7dte = pd.read_csv("data/data_7dte.csv")
data_7dte = data_7dte[data_7dte.Close.notna()]
data_7dte = data_7dte[data_7dte.IV.notna()]
CALL_7dte = data_7dte[data_7dte.cp_flag == "C"].reset_index(drop=True)
PUT_7dte = data_7dte[data_7dte.cp_flag == "P"].reset_index(drop=True)

data_30dte = pd.read_csv("data/data_30dte.csv")
data_30dte = data_30dte[data_30dte.Close.notna()]
data_30dte = data_30dte[data_30dte.IV.notna()]
CALL_30dte = data_30dte[data_30dte.cp_flag == "C"].reset_index(drop=True)
PUT_30dte = data_30dte[data_30dte.cp_flag == "P"].reset_index(drop=True)


def train_NIG(strikes, prices, spreads, r, S0, T, market_ivs):
    def f_NIG(x, sigma, theta, kappa):
        NIG_param = NIG_process(r=r, sigma=sigma, theta=theta, kappa=kappa)
        opt_param = Option_param(S0=S0, K=x, T=T, v0=0.04, exercise="European", payoff="call")
        NIG = NIG_pricer(opt_param, NIG_param)
        return NIG

    def obj_fun(params):
        try:
            model = f_NIG(strikes, params[0], params[1], params[2])
            model_prices = model.FFT(strikes)
            flag = ['c' for _ in range(len(strikes))]
            t = pd.Series([T for _ in range(len(strikes))])
            model_ivs = py_vollib_vectorized.vectorized_implied_volatility(model_prices, S0, strikes, t, r, flag, q=0, return_as='numpy')
            nan_mask = ~np.isfinite(model_ivs)
            if np.any(nan_mask):
                return 1e6
            return np.mean((market_ivs - model_ivs)**2)
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            return 1e6  # Return a large penalty value

    # init_vals = [0.2, -0.5, 0.2]
    # bounds = ([0, -np.inf, 0.01], [20, 50, 50])
    # params_NIG = scpo.least_squares(obj_fun, x0=init_vals, method='trf', bounds=bounds)
    init_vals = [0.4, -1, 1]
    try:
        bounds = ([0.01, -50.0, 0.01], [10.0, 50.0, 10.0])  # More restrictive bounds
        params_VG = scpo.least_squares(obj_fun, x0=init_vals, bounds=bounds, method='trf', 
                                       ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=500)
        return params_VG.x
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fall back to Nelder-Mead which doesn't require derivatives
        result = scpo.minimize(obj_fun, init_vals, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        return result.x
    
    bounds = [(0, 20), (-100, 100), (0.01, 50)]
    params_NIG = scpo.differential_evolution(obj_fun, bounds)
    return params_NIG.x

def get_NIG_pricer(params, strikes, r, S0, T):
    NIG_param = NIG_process(r=r, sigma=params[0], theta=params[1], kappa=params[2])
    opt_param = Option_param(S0=S0, K=strikes, T=T, v0=0.04, exercise="European", payoff="call")
    NIG = NIG_pricer(opt_param, NIG_param)
    return NIG

def calibrate_model(dataframe, T, model="NIG", disp=False):
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
        params_NIG = train_NIG(strikes, prices, spreads, r, S0, T, IV_actual)
        NIG = get_NIG_pricer(params_NIG, strikes, r, S0, T)
        prices_model = NIG.FFT(strikes)
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
            plt.plot(strikes, IV_model, label='NIG IV')
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

# ----------------------------- Calibration -----------------------------

# for ticker in ['aapl', 'amzn', 'msft']:
#     print(f"Running NIG for {ticker}")
#     data_1dte_2023_ticker = pd.read_csv(f"data/{ticker}_1dte_2023.csv")
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.Close.notna()]
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.IV.notna()]
#     CALL_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
#     MSE_NIG_1_ticker, params_NIG_1_ticker = calibrate_model(CALL_1dte_2023_ticker, T=1/252, model="NIG")
#     pd.DataFrame(pd.DataFrame(MSE_NIG_1_ticker)).to_csv(f"data/results/MSE_NIG_1_{ticker}.csv", index=False)
#     rmse_values = [np.sqrt(mse) for mse in MSE_NIG_1_ticker["MSE_list"]]
#     print(f"{np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")

MSE_NIG_1, params_NIG_1 = calibrate_model(CALL_1dte, T=1/252, model="NIG", disp=True)
pd.DataFrame(pd.DataFrame(MSE_NIG_1)).to_csv("data/results/MSE_NIG_1_2023.csv", index=False)

# MSE_NIG_7, params_NIG_7 = calibrate_model(CALL_7dte, T=5/252, model="NIG")
# pd.DataFrame(pd.DataFrame(MSE_NIG_7)).to_csv("data/results/MSE_NIG_7_2023.csv", index=False)

# MSE_NIG_30, params_NIG_30 = calibrate_model(CALL_30dte, T=22/252, model="NIG")
# pd.DataFrame(pd.DataFrame(MSE_NIG_30)).to_csv("data/results/MSE_NIG_30_2023.csv", index=False)


# ----------------------------- Hedging -----------------------------

T = 2/252
def greeks_function(strikes, prices, spreads, S0, IV_actual):
    deltas_BS = delta_BS(IV_actual, S0, strikes, T, r)
    gammas_BS = gamma_BS(IV_actual, S0, strikes, T, r)
    thetas_BS = theta_BS(IV_actual, S0, strikes, T, r)
    return deltas_BS, gammas_BS, thetas_BS


def get_model_predicted(strikes, prices, spreads, r, S0, T, IV_actual):
    params = train_NIG(strikes, prices, spreads, r, S0, T, IV_actual)
    NIG_param = NIG_process(r=r, sigma=params[0], theta=params[1], kappa=params[2])
    opt_param = Option_param(S0=S0, K=strikes, T=T, v0=0.04, exercise="European", payoff="call")
    NIG = NIG_pricer(opt_param, NIG_param)
    prices_model = NIG.FFT(strikes)
    flag = ['c' for _ in range(len(strikes))]
    t = pd.Series([T for _ in range(len(strikes))])
    IV_model = py_vollib_vectorized.vectorized_implied_volatility(prices_model, S0, strikes, t, r, flag, q=0, return_as='numpy')
    return prices_model, IV_model


# delta_pnl_list, delta_gamma_pnl_list, delta_gamma_theta_pnl_list, greeks_df = simulate_hedging(CALL_1dte, CALL_2dte, greeks_function, get_model_predicted)
# print("delta_pnl_list: ", delta_pnl_list)
# print("delta_gamma_pnl_list: ", delta_gamma_pnl_list)
# print("delta_gamma_theta_pnl_list: ", delta_gamma_theta_pnl_list)
# print("----------- Delta PnL -----------")
# print("Mean PnL: ", np.mean(delta_pnl_list))
# print("Std PnL: ", np.std(delta_pnl_list))
# print("----------- Delta Gamma PnL -----------")
# print("Mean PnL: ", np.mean(delta_gamma_pnl_list))
# print("Std PnL: ", np.std(delta_gamma_pnl_list))
# print("----------- Delta Gamma Theta PnL -----------")
# print("Mean PnL: ", np.mean(delta_gamma_theta_pnl_list))
# print("Std PnL: ", np.std(delta_gamma_theta_pnl_list))
# greeks_df.to_csv("data/hedging_results/greeks/greeks_df_nig_2023.csv", index=False)
# pd.DataFrame(pd.DataFrame(
#     {
#         "delta_pnl_list": delta_pnl_list,
#         "delta_gamma_pnl_list": delta_gamma_pnl_list,
#         "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
#     }
#     )).to_csv("data/hedging_results/delta/delta_pnl_list_2023_nig.csv", index=False)



for ticker in ['aapl', 'amzn', 'msft']:
    print("Running hedging for ", ticker)
    data_2dte_2023_ticker = pd.read_csv(f"data/source/{ticker}_2dte_2023.csv")
    data_2dte_2023_ticker = data_2dte_2023_ticker[data_2dte_2023_ticker.Close.notna()]
    data_2dte_2023_ticker = data_2dte_2023_ticker[data_2dte_2023_ticker.IV.notna()]
    CALL_2dte_2023_ticker = data_2dte_2023_ticker[data_2dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
    data_1dte_2023_ticker = pd.read_csv(f"data/source/{ticker}_1dte_2023.csv")
    data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.Close.notna()]
    data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.IV.notna()]
    CALL_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
    delta_pnl_list, delta_gamma_pnl_list, delta_gamma_theta_pnl_list, greeks_df = simulate_hedging(CALL_1dte_2023_ticker, CALL_2dte_2023_ticker, greeks_function, get_model_predicted)
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
    greeks_df.to_csv(f"data/hedging_results/greeks/greeks_df_nig_2023_{ticker}.csv", index=False)
    pd.DataFrame(pd.DataFrame(
        {
            "delta_pnl_list": delta_pnl_list,
            "delta_gamma_pnl_list": delta_gamma_pnl_list,
            "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
        }
        )).to_csv(f"data/hedging_results/delta/delta_pnl_list_2023_nig_{ticker}.csv", index=False)