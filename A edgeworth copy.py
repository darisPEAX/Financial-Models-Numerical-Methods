from FMNM.Parameters import Option_param
from FMNM.Processes import Edgeworth_process
from FMNM.Edgeworth_pricer1 import Edgeworth_pricer
from FMNM.utils import implied_volatility
from scipy.optimize import brentq, least_squares
from scipy.optimize import differential_evolution
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

data_1dte_2023 = pd.read_csv("data/source/data_1dte_2023_new.csv")
data_1dte_2023 = data_1dte_2023[data_1dte_2023.Close.notna()]
CALL_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "P"].reset_index(drop=True)

data_2dte_2023 = pd.read_csv("data/source/data_2dte_2023_new.csv")
data_2dte_2023 = data_2dte_2023[data_2dte_2023.Close.notna()]
CALL_2dte_2023 = data_2dte_2023[data_2dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_2dte_2023 = data_2dte_2023[data_2dte_2023.cp_flag == "P"].reset_index(drop=True)

data_7dte = pd.read_csv("data/source/data_7dte_2023_new.csv")
CALL_7dte = data_7dte[data_7dte.cp_flag == "C"].reset_index(drop=True)
PUT_7dte = data_7dte[data_7dte.cp_flag == "P"].reset_index(drop=True)

data_30dte = pd.read_csv("data/source/data_30dte_2023_new.csv")
CALL_30dte = data_30dte[data_30dte.cp_flag == "C"].reset_index(drop=True)
PUT_30dte = data_30dte[data_30dte.cp_flag == "P"].reset_index(drop=True)


def train_Edgeworth(dataframe_call, dataframe_put, r, S0, T):
    def f_Edgeworth(x, sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta, lambda_J, mu_J, sigma_J):
        Edgeworth_param = Edgeworth_process(r=r,
                                        sigma=sigma, beta_tilde=beta_tilde, rho=rho,
                                        alpha_Q=alpha_Q, delta_tilde=delta_tilde, eta=eta,
                                        lambda_J=lambda_J, mu_J=mu_J, sigma_J=sigma_J)
        opt_param = Option_param(S0=S0, K=x, T=T, v0=0.04, exercise="European", payoff="call")
        Edgeworth = Edgeworth_pricer(opt_param, Edgeworth_param)
        return Edgeworth

    def obj_fun(params):
        try:
            # Call
            model = f_Edgeworth(dataframe_call, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8])
            model_prices = model.FFT(dataframe_call.Strike.values)
            flag = ['c' for _ in range(len(dataframe_call.Strike.values))]
            t = pd.Series([T for _ in range(len(dataframe_call.Strike.values))])
            model_ivs = py_vollib_vectorized.vectorized_implied_volatility(model_prices, S0, dataframe_call.Strike.values, t, r, flag, q=0, return_as='numpy')
            nan_mask = ~np.isfinite(model_ivs)
            if np.any(nan_mask):
                return 1e6
            mse_call = np.mean((dataframe_call.IV.values - model_ivs)**2)

            # Put
            model = f_Edgeworth(dataframe_put, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8])
            model_prices = model.FFT(dataframe_put.Strike.values)
            flag = ['p' for _ in range(len(dataframe_put.Strike.values))]
            t = pd.Series([T for _ in range(len(dataframe_put.Strike.values))])
            model_ivs = py_vollib_vectorized.vectorized_implied_volatility(model_prices, S0, dataframe_put.Strike.values, t, r, flag, q=0, return_as='numpy')
            mse_put = np.mean((dataframe_put.IV.values - model_ivs)**2)
            return mse_call + mse_put
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            return 1e6  # Return a large penalty value

    bounds = [(0.05, 10), (0.01, 40), (-0.999, 0.999), (0.001, 10), (0.001, 30), (0.001, 10), (0.0001, 20), (0.001, 20), (0.001, 20)]    
    params_Edgeworth = scpo.differential_evolution(obj_fun, bounds)
    return params_Edgeworth.x

def get_Edgeworth_pricer(params, strikes, r, S0, T, payoff="call"):
    Edgeworth_param = Edgeworth_process(r=r,
                                        sigma=params[0], beta_tilde=params[1], rho=params[2],
                                        alpha_Q=params[3], delta_tilde=params[4], eta=params[5],
                                        lambda_J=params[6], mu_J=params[7], sigma_J=params[8])
    opt_param = Option_param(S0=S0, K=strikes, T=T, v0=0.04, exercise="European", payoff=payoff)
    Edgeworth = Edgeworth_pricer(opt_param, Edgeworth_param)
    return Edgeworth

def calibrate_model(dataframe_call, dataframe_put, T, model="Edgeworth", disp=False):
    MSE_list = []
    MSE_deep_itm_list = []
    MSE_mid_itm_list = []
    MSE_near_itm_list = []
    MSE_atm_list = []
    MSE_near_otm_list = []
    MSE_mid_otm_list = []
    MSE_deep_otm_list = []
    params_list = []
    for exdate in dataframe_call.exdate.unique()[:40]:
        print("Doing date: ", exdate)
        call_exdate = dataframe_call[dataframe_call.exdate == exdate]
        put_exdate = dataframe_put[dataframe_put.exdate == exdate]
        sort_idx_call = np.argsort(call_exdate.Strike.values)
        call_exdate = call_exdate.iloc[sort_idx_call]
        sort_idx_put = np.argsort(put_exdate.Strike.values)
        put_exdate = put_exdate.iloc[sort_idx_put]
        call_exdate = call_exdate[call_exdate.IV.notna()]
        put_exdate = put_exdate[put_exdate.IV.notna()]
        S0 = call_exdate.Close.values[0]

        # print("Training Model")
        start_time = time.time()
        params_Edgeworth = train_Edgeworth(call_exdate, put_exdate, r, S0, T)
        Edgeworth = get_Edgeworth_pricer(params_Edgeworth, call_exdate.Strike.values, r, S0, T)
        prices_model = Edgeworth.FFT(call_exdate.Strike.values)
        flag_c = ['c' for _ in range(len(call_exdate.Strike.values))]
        t_c = pd.Series([T for _ in range(len(call_exdate.Strike.values))])
        IV_model_call = py_vollib_vectorized.vectorized_implied_volatility(prices_model, S0, call_exdate.Strike.values, t_c, r, flag_c, q=0, return_as='numpy')
        Edgeworth = get_Edgeworth_pricer(params_Edgeworth, put_exdate.Strike.values, r, S0, T)
        prices_model = Edgeworth.FFT(put_exdate.Strike.values)
        flag_p = ['p' for _ in range(len(put_exdate.Strike.values))]
        t_p = pd.Series([T for _ in range(len(put_exdate.Strike.values))])
        IV_model_put = py_vollib_vectorized.vectorized_implied_volatility(prices_model, S0, put_exdate.Strike.values, t_p, r, flag_p, q=0, return_as='numpy')
        # Calculate MSE for all strikes
        # MSE_call = np.mean((IV_model_call - call_exdate.IV.values)**2)
        # MSE_put = np.mean((IV_model_put - put_exdate.IV.values)**2)
        MSE = np.mean((IV_model_call - call_exdate.IV.values)**2) + np.mean((IV_model_put - put_exdate.IV.values)**2)
        MSE_list.append(MSE)
        print("MSE: ", MSE)
        
        # Calculate MSE
        moneyness_call = call_exdate.Strike.values/S0
        moneyness_put = put_exdate.Strike.values/S0
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
            mask_call = (moneyness_call >= lower) & (moneyness_call < upper)
            mask_put = (moneyness_put >= lower) & (moneyness_put < upper)
            diff_call = np.array(IV_model_call)[mask_call] - np.array(call_exdate.IV.values)[mask_call]
            curr_mse = np.array([])
            if diff_call.size != 0:
                curr_mse = np.concatenate([curr_mse, diff_call**2])
            diff_put = np.array(IV_model_put)[mask_put] - np.array(put_exdate.IV.values)[mask_put]
            if diff_put.size != 0:
                curr_mse = np.concatenate([curr_mse, diff_put**2])
            locals()[f'MSE_{name}_list'].append(np.mean(curr_mse))

        if disp:
            print("exdate: ", exdate)
            print("date: ", set(call_exdate.date.values))
            print("MSE: ", MSE_call + MSE_put)
            print("Training time: ", time.time() - start_time, "seconds")
            plt.figure(figsize=(10,6))
            plt.scatter(call_exdate.Strike.values, call_exdate.IV.values, label='Actual IV', s=20, alpha=0.3)
            plt.plot(call_exdate.Strike.values, IV_model_call, label='Edgeworth IV')
            plt.scatter(put_exdate.Strike.values, put_exdate.IV.values, label='Actual IV', s=20, alpha=0.3)
            plt.plot(put_exdate.Strike.values, IV_model_put, label='Edgeworth IV')
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

# for ticker in ['aapl', 'amzn', 'msft']:
#     print(f"Running Edgeworth for {ticker}")
#     data_1dte_2023_ticker = pd.read_csv(f"data/source/{ticker}_1dte_2023.csv")
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.Close.notna()]
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.IV.notna()]
#     CALL_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
#     MSE_Edgeworth_1_ticker, params_Edgeworth_1_ticker = calibrate_model(CALL_1dte_2023_ticker, T=1/252, model="Edgeworth")
#     pd.DataFrame(pd.DataFrame(MSE_Edgeworth_1_ticker)).to_csv(f"data/results/MSE_Edgeworth_1_{ticker}.csv", index=False)
#     rmse_values = [np.sqrt(mse) for mse in MSE_Edgeworth_1_ticker["MSE_list"]]
#     print(f"{np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")

# MSE_Edgeworth_1, params_Edgeworth_1 = calibrate_model(CALL_1dte_2023, PUT_1dte_2023, T=1/252, model="Edgeworth")
# pd.DataFrame(pd.DataFrame(MSE_Edgeworth_1)).to_csv("data/results/MSE_Edgeworth_1_2023_copy.csv", index=False)
# print(MSE_Edgeworth_1["MSE_list"])
# rmse_values = [np.sqrt(mse) for mse in MSE_Edgeworth_1["MSE_list"]]
# print(f"{np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")

MSE_Edgeworth_7, params_Edgeworth_7 = calibrate_model(CALL_7dte, PUT_7dte, T=5/252, model="Edgeworth", disp=True)
pd.DataFrame(pd.DataFrame(MSE_Edgeworth_7)).to_csv("data/results/MSE_Edgeworth_7_2023_copy.csv", index=False)
rmse_values = [np.sqrt(mse) for mse in MSE_Edgeworth_7["MSE_list"]]
print("7dte")
print(f"{np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")

MSE_Edgeworth_30, params_Edgeworth_30 = calibrate_model(CALL_30dte, PUT_30dte, T=22/252, model="Edgeworth")
pd.DataFrame(pd.DataFrame(MSE_Edgeworth_30)).to_csv("data/results/MSE_Edgeworth_30_2023_copy.csv", index=False)
rmse_values = [np.sqrt(mse) for mse in MSE_Edgeworth_30["MSE_list"]]
print("30dte")
print(f"{np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")


from hedging import simulate_hedging
from greeks_helper import delta_BS, gamma_BS, theta_BS


def greeks_function(strikes, prices, spreads, S0, IV_actual):
    deltas_BS = delta_BS(IV_actual, S0, strikes, T, r)
    gammas_BS = gamma_BS(IV_actual, S0, strikes, T, r)
    thetas_BS = theta_BS(IV_actual, S0, strikes, T, r)
    return deltas_BS, gammas_BS, thetas_BS

def get_model_predicted(strikes, prices, spreads, r, S0, T, IV_actual):
    params = train_Edgeworth(strikes, prices, spreads, r, S0, T, IV_actual)
    Edgeworth = get_Edgeworth_pricer(params, strikes, r, S0, T)
    prices_model = Edgeworth.FFT(strikes)
    flag = ['c' for _ in range(len(strikes))]
    t = pd.Series([T for _ in range(len(strikes))])
    IV_model = py_vollib_vectorized.vectorized_implied_volatility(prices_model, S0, strikes, t, r, flag, q=0, return_as='numpy')
    return prices_model, IV_model

# delta_pnl_list, delta_gamma_pnl_list, delta_gamma_theta_pnl_list, greeks_df = simulate_hedging(CALL_1dte_2023, CALL_2dte_2023, greeks_function, get_model_predicted)
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
# greeks_df.to_csv("data/hedging_results/greeks/greeks_df_edgeworth_2023.csv", index=False)
# pd.DataFrame(pd.DataFrame(
#     {
#         "delta_pnl_list": delta_pnl_list,
#         "delta_gamma_pnl_list": delta_gamma_pnl_list,
#         "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
#     }
#     )).to_csv("data/hedging_results/delta/delta_pnl_list_2023_edgeworth.csv", index=False)





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
    greeks_df.to_csv(f"data/hedging_results/greeks/greeks_df_edgeworth_2023_{ticker}.csv", index=False)
    pd.DataFrame(pd.DataFrame(
        {
            "delta_pnl_list": delta_pnl_list,
            "delta_gamma_pnl_list": delta_gamma_pnl_list,
            "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
        }
        )).to_csv(f"data/hedging_results/delta/delta_pnl_list_2023_edgeworth_{ticker}.csv", index=False)