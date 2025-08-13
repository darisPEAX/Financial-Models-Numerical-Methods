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

# data_1dte_2023 = pd.read_csv("data/data_1dte_2023.csv")
data_1dte_2023 = pd.read_csv("data/source/data_1dte_2023_new.csv")
data_1dte_2023 = data_1dte_2023[data_1dte_2023.Close.notna()]
CALL_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "C"].reset_index(drop=True)
PUT_1dte_2023 = data_1dte_2023[data_1dte_2023.cp_flag == "P"].reset_index(drop=True)

# data_2dte_2023 = pd.read_csv("data/data_2dte_2023.csv")
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




# ——— Black-Scholes pricer for fallback ———
def bs_price(S, K, T, r, vol):
    d1 = (np.log(S/K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return np.exp(-r*T)*(S*norm.cdf(d1) - K*norm.cdf(d2))
        


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
        S0 = call_exdate.Close.values[0]
        strikes_call = call_exdate.Strike.values
        prices_call = call_exdate.Midpoint.values
        spreads_call = call_exdate.Spread.values
        strikes_put = put_exdate.Strike.values
        prices_put = put_exdate.Midpoint.values
        spreads_put = put_exdate.Spread.values
        IV_actual_call = call_exdate.IV.values
        IV_actual_put = put_exdate.IV.values

        moneyness_call = strikes_call / S0
        moneyness_put = strikes_put / S0
        
        # Concatenate call and put data into single arrays
        strikes_all = np.concatenate([strikes_put, strikes_call])
        prices_all = np.concatenate([prices_put, prices_call])
        spreads_all = np.concatenate([spreads_put, spreads_call])
        IV_actual_all = np.concatenate([IV_actual_put, IV_actual_call])
        moneyness_all = np.concatenate([moneyness_put, moneyness_call])
        
        # Create single feature matrix for combined data
        X_all = np.column_stack([np.ones_like(moneyness_all), moneyness_all, moneyness_all**2, moneyness_all**3])
        
        # Fit single model to combined call and put data
        beta_all = np.linalg.lstsq(X_all, IV_actual_all, rcond=None)[0]
        beta_0, beta_1, beta_2, beta_3 = beta_all
        params_list.append(beta_all)
        
        prices_model = []
        IV_model = []
        for i in range(len(strikes_all)):
            strike = strikes_all[i]
            vol = beta_0 + beta_1*moneyness_all[i] + beta_2*moneyness_all[i]**2 + beta_3*moneyness_all[i]**3
            prices_model.append(bs_price(S0, strike, T, r, vol))
            IV_model.append(vol)

        MSE = np.mean((IV_model - IV_actual_all)**2)
        MSE_list.append(MSE)

        # Calculate MSE for different moneyness ranges
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
            plt.plot(strikes_all, IV_model, label='AHBS IV')
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
#     print(f"Running AHBS for {ticker}")
#     data_1dte_2023_ticker = pd.read_csv(f"data/source/{ticker}_1dte_2023.csv")
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.Close.notna()]
#     data_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.IV.notna()]
#     CALL_1dte_2023_ticker = data_1dte_2023_ticker[data_1dte_2023_ticker.cp_flag == "C"].reset_index(drop=True)
#     MSE_AHBS_1_ticker, params_AHBS_1_ticker = calibrate_model(CALL_1dte_2023_ticker, T=1/252, model="AHBS")
#     pd.DataFrame(pd.DataFrame(MSE_AHBS_1_ticker)).to_csv(f"data/results/MSE_AHBS_1_{ticker}.csv", index=False)
#     rmse_values = [np.sqrt(mse) for mse in MSE_AHBS_1_ticker["MSE_list"]]
#     print(f"{np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")

MSE_AHBS_1, params_AHBS_1 = calibrate_model(CALL_1dte_2023, PUT_1dte_2023, T=1/252, model="AHBS")
for column in MSE_AHBS_1.keys():
    rmse_values = [np.sqrt(mse) for mse in MSE_AHBS_1[column]]
    print(f"{column}: {np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")
pd.DataFrame(pd.DataFrame(MSE_AHBS_1)).to_csv("data/results/MSE_AHBS_1_copy.csv", index=False)

# MSE_AHBS_7, params_AHBS_7 = calibrate_model(CALL_7dte, PUT_7dte, T=5/252, model="AHBS")
# for column in MSE_AHBS_7.keys():
#     rmse_values = [np.sqrt(mse) for mse in MSE_AHBS_7[column]]
#     print(f"{column}: {np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")
# pd.DataFrame(pd.DataFrame(MSE_AHBS_7)).to_csv("data/results/MSE_AHBS_7.csv", index=False)

# MSE_AHBS_30, params_AHBS_30 = calibrate_model(CALL_30dte, PUT_30dte, T=22/252, model="AHBS")
# for column in MSE_AHBS_30.keys():
#     rmse_values = [np.sqrt(mse) for mse in MSE_AHBS_30[column]]
#     print(f"{column}: {np.nanmean(rmse_values):.3f} ({np.nanstd(rmse_values):.3f})")
# pd.DataFrame(pd.DataFrame(MSE_AHBS_30)).to_csv("data/results/MSE_AHBS_30.csv", index=False)



from hedging import simulate_hedging
from greeks_helper import delta_BS, gamma_BS, theta_BS


T = 2/252
def greeks_function(strikes, prices, spreads, S0, IV_actual):
    deltas_BS = delta_BS(IV_actual, S0, strikes, T, r)
    gammas_BS = gamma_BS(IV_actual, S0, strikes, T, r)
    thetas_BS = theta_BS(IV_actual, S0, strikes, T, r)
    return deltas_BS, gammas_BS, thetas_BS


def get_model_predicted(strikes, prices, spreads, r, S0, T, IV_actual):
    moneyness = strikes / S0
    X = np.column_stack([np.ones_like(moneyness), moneyness, moneyness**2, moneyness**3])
    beta = np.linalg.lstsq(X, IV_actual, rcond=None)[0]
    beta_0, beta_1, beta_2, beta_3 = beta
    prices_model = []
    IV_model = []
    for i in range(len(strikes)):
        strike = strikes[i]
        vol = beta_0 + beta_1*moneyness[i] + beta_2*moneyness[i]**2 + beta_3*moneyness[i]**3
        prices_model.append(bs_price(S0, strike, T, r, vol))
        IV_model.append(vol)
    return np.array(prices_model), np.array(IV_model)


# def greeks_function(strikes, prices, spreads, S0, IV_actual):
#     moneyness = strikes / S0
#     X = np.column_stack([np.ones_like(moneyness), moneyness, moneyness**2, moneyness**3])
#     beta = np.linalg.lstsq(X, IV_actual, rcond=None)[0]
#     beta_0, beta_1, beta_2, beta_3 = beta
#     IV_model = beta_0 + beta_1*moneyness + beta_2*moneyness**2 + beta_3*moneyness**3

#     deltas = []
#     gammas = []
#     thetas = []
#     for i, k in enumerate(strikes):
#         vol = IV_model[i]
#         d1 = (np.log(S0/k) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
#         delta = norm.cdf(d1)
#         deltas.append(delta)
#         gamma = norm.pdf(d1) / (S0 * vol * np.sqrt(T))
#         gammas.append(gamma)
#         theta = -S0 * vol * norm.pdf(d1) / (2 * np.sqrt(T))
#         thetas.append(theta)

#     return deltas, gammas, thetas

# ----------------------------- Hedging -----------------------------

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
# greeks_df.to_csv("data/hedging_results/greeks_df_ahbs.csv", index=False)
# pd.DataFrame(pd.DataFrame(
#     {
#         "delta_pnl_list": delta_pnl_list,
#         "delta_gamma_pnl_list": delta_gamma_pnl_list,
#         "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
#     }
#     )).to_csv("data/hedging_results/greeks_pnl_list_ahbs_2023.csv", index=False)



# deltas, gammas, thetas = greeks_function(CALL_1dte_2023.Strike.values, CALL_1dte_2023.Midpoint.values, CALL_1dte_2023.Spread.values, CALL_1dte_2023.Close.values[0], CALL_1dte_2023.IV.values)
# plt.plot(CALL_1dte_2023.Strike.values, deltas)
# print(gammas)
# plt.show()
# plt.plot(CALL_1dte_2023.Strike.values, gammas)
# plt.show()
# plt.plot(CALL_1dte_2023.Strike.values, thetas)
# plt.show()


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
    greeks_df.to_csv(f"data/hedging_results/greeks/greeks_df_ahbs_2023_{ticker}.csv", index=False)
    pd.DataFrame(pd.DataFrame(
        {
            "delta_pnl_list": delta_pnl_list,
            "delta_gamma_pnl_list": delta_gamma_pnl_list,
            "delta_gamma_theta_pnl_list": delta_gamma_theta_pnl_list
        }
        )).to_csv(f"data/hedging_results/delta/delta_pnl_list_2023_ahbs_{ticker}.csv", index=False)