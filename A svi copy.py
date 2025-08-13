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

import py_vollib.black_scholes_merton.implied_volatility
import py_vollib_vectorized

warnings.filterwarnings("ignore")


r = 0.05
T = 1 / 252

# data_1dte = pd.read_csv("data/data_1dte.csv")
# data_1dte = data_1dte[data_1dte.Close.notna()]
# CALL_1dte = data_1dte[data_1dte.cp_flag == "C"].reset_index(drop=True)
# PUT_1dte = data_1dte[data_1dte.cp_flag == "P"].reset_index(drop=True)

data_1dte_2023 = pd.read_csv("data/source/data_1dte_2023_new.csv")
data_1dte_2023 = data_1dte_2023[data_1dte_2023.Close.notna()]
CALL_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "P"].reset_index(drop=True)

data_2dte_2023 = pd.read_csv("data/source/data_2dte_2023_new.csv")
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



        
def raw_svi(x, a, b, sigma, rho, m):
    return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))

# SVI implied volatility function (vol = sqrt(total variance / T))
def svi_vol(x, a, b, sigma, rho, m):
    raw = raw_svi(x, a, b, sigma, rho, m)
    # clip to avoid negative total variance -> no NaNs
    raw = np.maximum(raw, 1e-6)
    return np.sqrt(raw / T)

def calibrate_model(dataframe_call, dataframe_put, T, model="Merton", disp=False):
    MSE_list = []
    MSE_deep_itm_list = []
    MSE_mid_itm_list = []
    MSE_near_itm_list = []
    MSE_atm_list = []
    MSE_near_otm_list = []
    MSE_mid_otm_list = []
    MSE_deep_otm_list = []
    params_list = []
    for exdate in dataframe_call.exdate.unique():
        if disp == True:
            print("Doing date: ", exdate)
        call_exdate = dataframe_call[dataframe_call.exdate == exdate]
        put_exdate = dataframe_put[dataframe_put.exdate == exdate]
        sort_idx_call = np.argsort(call_exdate.Strike.values)
        call_exdate = call_exdate.iloc[sort_idx_call]
        sort_idx_put = np.argsort(put_exdate.Strike.values)
        put_exdate = put_exdate.iloc[sort_idx_put]
        call_exdate = call_exdate[call_exdate.IV.notna()]
        call_exdate = call_exdate[call_exdate.Strike / call_exdate.Close > 1]
        put_exdate = put_exdate[put_exdate.IV.notna()]
        put_exdate = put_exdate[put_exdate.Strike / put_exdate.Close < 1]
        strikes_call = call_exdate.Strike.values
        prices_call = call_exdate.Midpoint.values
        spreads_call = call_exdate.Spread.values
        strikes_put = put_exdate.Strike.values
        prices_put = put_exdate.Midpoint.values
        spreads_put = put_exdate.Spread.values
        IV_actual_call = call_exdate.IV.values
        IV_actual_put = put_exdate.IV.values
        S0 = call_exdate.Close.values[0]

        moneyness_call = strikes_call / S0
        moneyness_put = strikes_put / S0
        
        # Concatenate call and put data into single arrays
        strikes_all = np.concatenate([strikes_put, strikes_call])
        prices_all = np.concatenate([prices_put, prices_call])
        spreads_all = np.concatenate([spreads_put, spreads_call])
        IV_actual_all = np.concatenate([IV_actual_put, IV_actual_call])
        moneyness_all = np.concatenate([moneyness_put, moneyness_call])
        
        log_moneyness = np.log(strikes_all/S0)
        # --- vectorized residuals with weighting ---
        def residuals(params):
            vols   = svi_vol(log_moneyness, *params)
            weights = np.sqrt(np.maximum(IV_actual_all, 1e-4))
            return (vols - IV_actual_all) / weights

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
        MSE = np.mean((IV_model - IV_actual_all)**2)
        MSE_list.append(MSE)

        # Calculate MSE
        moneyness = strikes_all/S0
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
            mse = np.mean((np.array(IV_model)[mask] - np.array(IV_actual_all)[mask])**2)
            locals()[f'MSE_{name}_list'].append(mse)

        if disp == True:
            print("exdate: ", exdate)
            print("date: ", set(call_exdate.date.values))
            print("MSE: ", MSE)
            plt.figure(figsize=(10,6))
            plt.scatter(strikes_all, IV_actual_all, label='Actual IV', s=20, alpha=0.3)
            plt.plot(strikes_all, IV_model, label='SVI IV')
            plt.xlabel('Strike Price')
            plt.ylabel('Implied Volatility')
            plt.title('Implied Volatility Comparison')
            plt.legend()
            plt.grid(True)
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


# for ticker in ['aapl', 'amzn', 'msft']:
#     print(f"Running SVI for {ticker}")
#     data_1dte_2023_ticker = pd.read_csv(f"data/{ticker}_1dte_2023.csv")
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.Close.notna()]
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.IV.notna()]
#     CALL_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
#     MSE_SVI_1_ticker, params_SVI_1_ticker = calibrate_model(CALL_1dte_2023_ticker, T=1/252, model="SVI")
#     pd.DataFrame(pd.DataFrame(MSE_SVI_1_ticker)).to_csv(f"data/results/MSE_SVI_1_{ticker}.csv", index=False)
#     rmse_values = [np.sqrt(mse) for mse in MSE_SVI_1_ticker["MSE_list"]]
#     print(f"{np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")

# MSE_SVI_1, params_SVI_1 = calibrate_model(CALL_1dte_2023, PUT_1dte_2023, T=1/252, model="SVI", disp=True)
# for column in MSE_SVI_1.keys():
#     rmse_values = [np.sqrt(mse) for mse in MSE_SVI_1[column]]
#     print(f"{column}: {np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")
# pd.DataFrame(pd.DataFrame(MSE_SVI_1)).to_csv("data/results/MSE_SVI_1_2023_copy.csv", index=False)

# MSE_SVI_7, params_SVI_7 = calibrate_model(CALL_7dte, PUT_7dte, T=5/252, model="SVI", disp=True)
# for column in MSE_SVI_7.keys():
#     rmse_values = [np.sqrt(mse) for mse in MSE_SVI_7[column]]
#     print(f"{column}: {np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")
# pd.DataFrame(pd.DataFrame(MSE_SVI_7)).to_csv("data/results/MSE_SVI_7_2023_copy.csv", index=False)

# MSE_SVI_30, params_SVI_30 = calibrate_model(CALL_30dte, PUT_30dte, T=22/252, model="SVI", disp=True)
# for column in MSE_SVI_30.keys():
#     rmse_values = [np.sqrt(mse) for mse in MSE_SVI_30[column]]
#     print(f"{column}: {np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")
# pd.DataFrame(pd.DataFrame(MSE_SVI_30)).to_csv("data/results/MSE_SVI_30_2023_copy.csv", index=False)


from hedging import simulate_hedging


def greeks_function(strikes, prices, spreads, S0, IV_actual):
    log_moneyness = np.log(strikes/S0)
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

    # Calculate greeks via finite difference
    deltas = []
    gammas = []
    thetas = []

    def delta(x, h=None, S0=S0, T=T, r=r, IV_model=IV_model):
        """
        Compute delta numerically via central difference.
        method: one of "closed_formula", "Fourier", or "MC"
        h: step size for perturbation. If None, defaults to 1% of S0.
        """
        if h is None:
            h = 0.01 * S0

        # Store original S0
        S0_orig = S0

        # Perturb up
        S0 = S0_orig + h
        price_up = py_vollib_vectorized.vectorized_black_scholes_merton('c', S0, x, T, r, IV_model, q=0)['Price']

        # Perturb down
        S0 = S0_orig - h
        price_down = py_vollib_vectorized.vectorized_black_scholes_merton('c', S0, x, T, r, IV_model, q=0)['Price']
        
        # Restore original S0
        S0 = S0_orig

        # Central difference
        delta = (price_up - price_down) / (2 * h)
        return delta

    def gamma(x, h=None, S0=S0, T=T, r=r, IV_model=IV_model):
        """
        Compute gamma numerically via central difference.
        """
        if h is None:
            h = 0.01 * S0

        # Store original S0
        S0_orig = S0

        # Perturb up
        S0 = S0_orig + h
        delta_up = delta(x, h)

        # Perturb down
        S0 = S0_orig - h
        delta_down = delta(x, h)

        # Restore original S0
        S0 = S0_orig

        # Central difference
        gamma = (delta_up - delta_down) / (2 * h)
        return gamma

    def theta(x, h=None, S0=S0, T=T, r=r, IV_model=IV_model):
        """
        Compute theta numerically via central difference.
        """
        if h is None:
            h = 0.01 * S0

        T_orig = T

        price_now = py_vollib_vectorized.vectorized_black_scholes_merton('c', S0, x, T, r, IV_model, q=0)['Price']

        # Go forward in time by 1 day
        T = T_orig - 1.0/252
        price_future = py_vollib_vectorized.vectorized_black_scholes_merton('c', S0, x, T, r, IV_model, q=0)['Price']
        
        # Restore original T
        T = T_orig

        # Central difference
        theta = (price_future - price_now)*252
        return theta
    
    deltas = np.array([delta(k) for k in strikes])
    gammas = np.array([gamma(k) for k in strikes])
    thetas = np.array([theta(k) for k in strikes])
    
    # for i, k in enumerate(strikes):
    #     # Get the SVI volatility for this strike
    #     vol = IV_model[i]
        
    #     # Calculate d1 from Black-Scholes
    #     d1 = (np.log(S0/k) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        
    #     # Calculate delta
    #     delta = norm.cdf(d1)
    #     deltas.append(delta)
        
    #     # Calculate gamma
    #     gamma = norm.pdf(d1) / (S0 * vol * np.sqrt(T))
    #     gammas.append(gamma)
        
    #     # Calculate theta
    #     theta = -S0 * vol * norm.pdf(d1) / (2 * np.sqrt(T))
    #     thetas.append(theta)
    return deltas, gammas, thetas

from hedging import simulate_hedging
from greeks_helper import delta_BS, gamma_BS, theta_BS


def greeks_function(strikes, prices, spreads, S0, IV_actual):
    deltas_BS = delta_BS(IV_actual, S0, strikes, T, r)
    gammas_BS = gamma_BS(IV_actual, S0, strikes, T, r)
    thetas_BS = theta_BS(IV_actual, S0, strikes, T, r)
    return deltas_BS, gammas_BS, thetas_BS

def get_model_predicted(strikes, prices, spreads, r, S0, T, IV_actual):
    log_moneyness = np.log(strikes/S0)
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
    prices_model = py_vollib_vectorized.vectorized_black_scholes_merton('c', S0, strikes, T, r, IV_model, q=0)['Price']
    return np.array(prices_model), np.array(IV_model)

delta_pnl_list, delta_gamma_pnl_list, delta_gamma_theta_pnl_list, greeks_df = simulate_hedging(CALL_1dte_2023, CALL_2dte_2023, greeks_function, get_model_predicted)
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
greeks_df.to_csv("data/hedging_results/greeks/greeks_df_svi_2023_test.csv", index=False)
pd.DataFrame(pd.DataFrame(
    {
        "delta_pnl_list": delta_pnl_list,
        "delta_gamma_pnl_list": delta_gamma_pnl_list,
        "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
    }
    )).to_csv("data/hedging_results/delta/delta_pnl_list_2023_svi_test.csv", index=False)



# for ticker in ['aapl', 'amzn', 'msft']:
#     print("Running hedging for ", ticker)
#     data_2dte_2023_ticker = pd.read_csv(f"data/source/{ticker}_2dte_2023.csv")
#     data_2dte_2023_ticker = data_2dte_2023_ticker[data_2dte_2023_ticker.Close.notna()]
#     data_2dte_2023_ticker = data_2dte_2023_ticker[data_2dte_2023_ticker.IV.notna()]
#     CALL_2dte_2023_ticker = data_2dte_2023_ticker[data_2dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
#     data_1dte_2023_ticker = pd.read_csv(f"data/source/{ticker}_1dte_2023.csv")
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.Close.notna()]
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.IV.notna()]
#     CALL_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
#     delta_pnl_list, delta_gamma_pnl_list, delta_gamma_theta_pnl_list, greeks_df = simulate_hedging(CALL_1dte_2023_ticker, CALL_2dte_2023_ticker, greeks_function, get_model_predicted)
#     print("delta_pnl_list: ", delta_pnl_list)
#     print("delta_gamma_pnl_list: ", delta_gamma_pnl_list)
#     print("delta_gamma_theta_pnl_list: ", delta_gamma_theta_pnl_list)
#     print("----------- Delta PnL -----------")
#     print("Mean PnL: ", np.mean(delta_pnl_list))
#     print("Std PnL: ", np.std(delta_pnl_list))
#     print("----------- Delta Gamma PnL -----------")
#     print("Mean PnL: ", np.mean(delta_gamma_pnl_list))
#     print("Std PnL: ", np.std(delta_gamma_pnl_list))
#     print("----------- Delta Gamma Theta PnL -----------")
#     print("Mean PnL: ", np.mean(delta_gamma_theta_pnl_list))
#     print("Std PnL: ", np.std(delta_gamma_theta_pnl_list))
#     greeks_df.to_csv(f"data/hedging_results/greeks/greeks_df_svi_2023_{ticker}.csv", index=False)
#     pd.DataFrame(pd.DataFrame(
#         {
#             "delta_pnl_list": delta_pnl_list,
#             "delta_gamma_pnl_list": delta_gamma_pnl_list,
#             "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
#         }
#         )).to_csv(f"data/hedging_results/delta/delta_pnl_list_2023_svi_{ticker}.csv", index=False)