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
data_1dte = data_1dte[data_1dte.Close.notna()]
CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_1dte_2023 = pd.read_csv("data/data_1dte_2023.csv")
data_1dte_2023 = data_1dte_2023[data_1dte_2023.Close.notna()]
CALL_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "P"].reset_index(drop=True)

data_2dte_2023 = pd.read_csv("data/data_2dte_2023.csv")
data_2dte_2023 = data_2dte_2023[data_2dte_2023.Close.notna()]
CALL_2dte_2023 = data_2dte_2023[data_2dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_2dte_2023 = data_2dte_2023[data_2dte_2023.cp_flag == "P"].reset_index(drop=True)


data_7dte = pd.read_csv("data/data_7dte.csv")
data_7dte = data_7dte[data_7dte.Close.notna()]
CALL_7dte = data_7dte[data_7dte.cp_flag == "C"].reset_index(drop=True)
PUT_7dte = data_7dte[data_7dte.cp_flag == "P"].reset_index(drop=True)

data_30dte = pd.read_csv("data/data_30dte.csv")
data_30dte = data_30dte[data_30dte.Close.notna()]
CALL_30dte = data_30dte[data_30dte.cp_flag == "C"].reset_index(drop=True)
PUT_30dte = data_30dte[data_30dte.cp_flag == "P"].reset_index(drop=True)

import py_vollib.black_scholes_merton.implied_volatility
import py_vollib_vectorized


def train_VG(strikes, prices, spreads, r, S0, T, market_ivs):
    def f_VG(x, theta, sigma, kappa):
        VG_param = VG_process(r=r, theta=theta, sigma=sigma, kappa=kappa)
        opt_param = Option_param(S0=S0, K=x, T=T, v0=0.04, exercise="European", payoff="call")
        VG = VG_pricer(opt_param, VG_param)
        return VG

    def obj_fun(params):
        model = f_VG(strikes, params[0], params[1], params[2])
        model_prices = model.FFT(strikes)
        flag = ['c' for _ in range(len(strikes))]
        t = pd.Series([T for _ in range(len(strikes))])
        model_ivs = py_vollib_vectorized.vectorized_implied_volatility(model_prices, S0, strikes, t, r, flag, q=0, return_as='numpy')
        return np.mean((market_ivs - model_ivs)**2)
        # try:
        #     model_prices = model.FFT(strikes)
        #     model_ivs = np.array([safe_iv(p, S0, K, T, r, implied_volatility)
        #                         for p, K in zip(model_prices, strikes)])
            
        #     # Check for NaN or infinite values and replace them
        #     nan_mask = ~np.isfinite(model_ivs)
        #     if np.any(nan_mask):
        #         # Use a large penalty for non-finite values
        #         return 1e6
            
        #     return np.mean((market_ivs - model_ivs)**2)
        # except Exception as e:
        #     print(f"Error with parameters {params}: {e}")
        #     return 1e6  # Return a large penalty value

    # Try different initial values
    init_vals = [0.01, 0.2, 0.1]  # More conservative initial values
    bounds = ([-1.0, 0.01, 0.01], [1.0, 2.0, 1.0])  # More restrictive bounds
    
    # Test initial values to ensure they work
    try:
        print("Testing initial parameters...")
        test_model = f_VG(strikes, init_vals[0], init_vals[1], init_vals[2])
        test_prices = test_model.FFT(strikes)
        print(f"Initial test successful, got {len(test_prices)} prices")
    except Exception as e:
        print(f"Initial parameter test failed: {e}")
        # If initial test fails, try even more conservative values
        init_vals = [0.0, 0.2, 0.1]
        print(f"Trying more conservative values: {init_vals}")
    
    # Try with different optimizer settings
    try:
        params_VG = scpo.least_squares(obj_fun, x0=init_vals, bounds=bounds, method='trf', 
                                       ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=500)
        return params_VG.x
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fall back to Nelder-Mead which doesn't require derivatives
        result = scpo.minimize(obj_fun, init_vals, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        return result.x

def get_VG_pricer(params, strikes, r, S0, T):
    VG_param = VG_process(r=r, theta=params[0], sigma=params[1], kappa=params[2])
    opt_param = Option_param(S0=S0, K=strikes, T=T, v0=0.04, exercise="European", payoff="call")
    VG = VG_pricer(opt_param, VG_param)
    return VG

def calibrate_model(dataframe, T, model="Merton", disp=False):
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

        start_time = time.time()
        params_VG = train_VG(strikes, prices, spreads, r, S0, T, IV_actual)
        VG = get_VG_pricer(params_VG, strikes, r, S0, T)
        prices_model = VG.FFT(strikes)
        flag = ['c' for _ in range(len(strikes))]
        t = pd.Series([T for _ in range(len(strikes))])
        IV_model = py_vollib_vectorized.vectorized_implied_volatility(prices_model, S0, strikes, t, r, flag, q=0, return_as='numpy')
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
            (1.15, float('inf'), 'deep_otm')
        ]
        for lower, upper, name in ranges:
            mask = (moneyness >= lower) & (moneyness < upper)
            mse = np.mean((IV_model[mask] - IV_actual[mask])**2)
            locals()[f'MSE_{name}_list'].append(mse)

        print("MSE: ", MSE)
        if disp:
            print("exdate: ", exdate)
            print("date: ", set(option_type_exdate.date.values))
            
            print("Training time: ", time.time() - start_time, "seconds")
            plt.figure(figsize=(10,6))
            plt.scatter(strikes, IV_actual, label='Actual IV', s=20, alpha=0.3)
            plt.plot(strikes, IV_model, label='VG IV')
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

# MSE_VG_1, params_VG_1 = calibrate_model(CALL_1dte, T=1/252, model="VG")
# pd.DataFrame(pd.DataFrame(MSE_VG_1)).to_csv("data/MSE_VG_1.csv", index=False)

# MSE_VG_7, params_VG_7 = calibrate_model(CALL_7dte, T=5/252, model="VG")
# pd.DataFrame(pd.DataFrame(MSE_VG_7)).to_csv("data/MSE_VG_7.csv", index=False)

# MSE_VG_30, params_VG_30 = calibrate_model(CALL_30dte, T=22/252, model="VG")
# pd.DataFrame(pd.DataFrame(MSE_VG_30)).to_csv("data/MSE_VG_30.csv", index=False)



from hedging import simulate_hedging


def greeks_function(strikes, prices, spreads, S0, IV_actual):
    params_VG = train_VG(strikes, prices, spreads, r, S0, T, IV_actual)
    VG = get_VG_pricer(params_VG, strikes, r, S0, T)
    deltas = VG.delta(strikes)
    gammas = VG.gamma(strikes)
    thetas = VG.theta_greek(strikes)
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
    )).to_csv("data/delta_pnl_list_2023.csv", index=False)




# def out_of_sample_test(params_list, dataframe, T, model="VG", disp=False):
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

#         params = params_list[index-1]
#         VG = get_VG_pricer(params, strikes, r, S0, T)
#         prices_model = VG.FFT(strikes)
#         IV_model = np.array([safe_iv(p, S0, K, T, r, implied_volatility)
#                             for p, K in zip(prices_model, strikes)])
        
#         MSE = np.mean((IV_model - IV_actual)**2)
#         MSE_list.append(MSE)
#         if disp == True:
#             print("exdate: ", exdate)
#             print("date: ", set(option_type_exdate.date.values))
#             print("MSE: ", MSE)
#     return MSE_list


# MSE_VG_1_oos = out_of_sample_test(params_VG_1, CALL_1dte, T=1/252, model="VG")
# MSE_VG_7_oos = out_of_sample_test(params_VG_7, CALL_7dte, T=5/252, model="VG")
# MSE_VG_30_oos = out_of_sample_test(params_VG_30, CALL_30dte, T=22/252, model="VG")

# pd.DataFrame({"MSE_VG_1_oos": MSE_VG_1_oos}).to_csv("data/MSE_VG_1_oos.csv", index=False)
# pd.DataFrame({"MSE_VG_7_oos": MSE_VG_7_oos}).to_csv("data/MSE_VG_7_oos.csv", index=False)
# pd.DataFrame({"MSE_VG_30_oos": MSE_VG_30_oos}).to_csv("data/MSE_VG_30_oos.csv", index=False)


# print(np.mean(MSE_VG_1_oos), np.mean(MSE_VG_7_oos), np.mean(MSE_VG_30_oos))
# print(np.std(MSE_VG_1_oos), np.std(MSE_VG_7_oos), np.std(MSE_VG_30_oos))