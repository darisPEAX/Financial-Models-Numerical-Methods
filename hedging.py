import numpy as np
import pandas as pd

def simulate_hedging(df_1dte, df_2_dte, greeks_function, get_model_predicted, disp=False):
    common_exdates = set(df_1dte['exdate']).intersection(set(df_2_dte['exdate']))
    print("Number of common exdates: ", len(common_exdates))
    df_1dte = df_1dte[df_1dte['exdate'].isin(common_exdates)]
    df_2_dte = df_2_dte[df_2_dte['exdate'].isin(common_exdates)]

    delta_pnl_list = []
    delta_gamma_pnl_list = []
    delta_gamma_theta_pnl_list = []
    greeks_df = pd.DataFrame(columns=['moneyness', 'delta', 'gamma', 'theta'])
    for exdate in df_2_dte['exdate'].unique():
        print("Doing exdate: ", exdate)
        option_2dte = df_2_dte[df_2_dte['exdate'] == exdate]
        sort_idx = np.argsort(option_2dte.Strike.values)
        option_2dte = option_2dte.iloc[sort_idx]
        option_2dte = option_2dte[option_2dte.IV.notna()]
        strikes = option_2dte.Strike.values
        prices = option_2dte.Midpoint.values
        spreads = option_2dte.Spread.values
        S0 = option_2dte.Close.values[0]
        IV_actual = option_2dte.IV.values
        prices_model, IV_model = get_model_predicted(strikes, prices, spreads, 0.04, S0, 2/252, IV_actual)

        rounded_strike = np.floor(S0/10) * 10

        if len(option_2dte) < 20:
            print("Not enough data for 2DTE exdate: ", exdate)
            continue

        moneyness = strikes / S0
        print('prices_model: ', prices_model)
        print('IV_model: ', IV_model)
        deltas, gammas, thetas = greeks_function(strikes, prices_model, spreads, S0, IV_model)
        option_2dte['delta'] = deltas
        option_2dte['gamma'] = gammas
        option_2dte['theta'] = thetas
        delta_across_moneyness = list(zip(moneyness, deltas, gammas, thetas))
        greeks_df = pd.concat([greeks_df, pd.DataFrame(delta_across_moneyness, columns=['moneyness', 'delta', 'gamma', 'theta'])])

        option_1dte = df_1dte[df_1dte['exdate'] == exdate]
        sort_idx = np.argsort(option_1dte.Strike.values)
        option_1dte = option_1dte.iloc[sort_idx]
        option_1dte = option_1dte[option_1dte.IV.notna()]
        strikes = option_1dte.Strike.values
        prices = option_1dte.Midpoint.values
        spreads = option_1dte.Spread.values
        S0 = option_1dte.Close.values[0]
        IV_actual = option_1dte.IV.values

        if len(option_1dte) < 20:
            print("Not enough data for 1DTE exdate: ", exdate)
            continue

        rounded_strike_up = rounded_strike + 10
        rounded_strike_down = rounded_strike - 10
        # Find a strike that is both in option_1dte and option_2dte and is far in the money
        # "Far in the money" for a call means strike << S0, so start well below S0
        strikes_common = np.intersect1d(option_1dte['Strike'].values, option_2dte['Strike'].values)
        # print('S0: ', S0)
        # print('strikes_common: ', strikes_common)
        # Sort strikes ascending, so lowest (most ITM for call) is first
        strikes_common_sorted = np.sort(strikes_common)
        # print('strikes_common_sorted: ', strikes_common_sorted)
        # Pick the lowest strike that is at least 10% below S0, or just the lowest if none meet that
        far_itm_candidates = strikes_common_sorted[strikes_common_sorted < 0.98 * S0]
        print('far_itm_candidates: ', far_itm_candidates)
        if len(far_itm_candidates) > 0:
            strike_delta = far_itm_candidates[-1]
        else:
            strike_delta = strikes_common_sorted[0]

        print('strike_delta: ', strike_delta)
        
        option_1dte_delta = option_1dte[option_1dte['Strike'] == strike_delta].iloc[0]
        option_2dte_delta = option_2dte[option_2dte['Strike'] == strike_delta].iloc[0]
        while True:
            if rounded_strike < option_1dte['Strike'].values[0] or rounded_strike < option_2dte['Strike'].values[0]:
                print("Strike out of bounds")
                rounded_strike = -1
                break
            if rounded_strike in option_1dte['Strike'].values and rounded_strike in option_2dte['Strike'].values:
                option_1dte_up = option_1dte[option_1dte['Strike'] == rounded_strike_up].iloc[0]
                option_2dte_up = option_2dte[option_2dte['Strike'] == rounded_strike_up].iloc[0]
                option_1dte_down = option_1dte[option_1dte['Strike'] == rounded_strike_down].iloc[0]
                option_2dte_down = option_2dte[option_2dte['Strike'] == rounded_strike_down].iloc[0]
                option_1dte = option_1dte[option_1dte['Strike'] == rounded_strike].iloc[0]
                option_2dte = option_2dte[option_2dte['Strike'] == rounded_strike].iloc[0]
                break
            rounded_strike -= 10
            rounded_strike_up += 10
            rounded_strike_down -= 10
        if rounded_strike == -1:
            continue

        underlying_2dte = float(option_2dte['Close'])
        underlying_1dte = float(option_1dte['Close'])

        price_2dte = float(option_2dte['Midpoint'])
        price_1dte = float(option_1dte['Midpoint'])
        price_2dte_up = float(option_2dte_up['Midpoint'])
        price_1dte_up = float(option_1dte_up['Midpoint'])
        price_2dte_down = float(option_2dte_down['Midpoint'])
        price_1dte_down = float(option_1dte_down['Midpoint'])
        price_2dte_delta = float(option_2dte_delta['Midpoint'])
        price_1dte_delta = float(option_1dte_delta['Midpoint'])
        delta = float(option_2dte['delta'])
        gamma = float(option_2dte['gamma'])
        theta = float(option_2dte['theta'])
        delta_up = float(option_2dte_up['delta'])
        gamma_up = float(option_2dte_up['gamma'])
        theta_up = float(option_2dte_up['theta'])
        delta_down = float(option_2dte_down['delta'])
        gamma_down = float(option_2dte_down['gamma'])
        theta_down = float(option_2dte_down['theta'])
        weight = - gamma / gamma_up
        # delta_pnl = (float(price_2dte - delta*underlying_2dte)
        #        - float(price_1dte - delta*underlying_1dte))
        # delta_pnl = (float(price_2dte_down - delta*underlying_2dte)
        #        - float(price_1dte_down - delta*underlying_1dte))
        delta_pnl = (float(price_2dte_delta - delta*underlying_2dte)
               - float(price_1dte_delta - delta*underlying_1dte))
        delta_gamma_pnl = (price_2dte + weight*price_2dte_up
                           - (delta + weight*delta_up)*underlying_2dte
                           - (price_1dte + weight*price_1dte_up)
                           + (delta + weight*delta_up)*underlying_1dte)
        
        weight_t = ((gamma*theta_up/gamma_up) - theta) / (theta_down - gamma_down*theta_up/gamma_up)
        weight_g = -(gamma - weight_t*gamma_down) / gamma_up
        delta_gamma_theta_pnl = (price_2dte + weight_g*price_2dte_up + weight_t*price_2dte_down
                                 - (delta + weight_g*delta_up + weight_t*delta_down)*underlying_2dte
                                 - (price_1dte + weight_g*price_1dte_up + weight_t*price_1dte_down)
                                 + (delta + weight_g*delta_up + weight_t*delta_down)*underlying_1dte)

        print('delta_pnl', delta_pnl)
        print('delta_gamma_pnl', delta_gamma_pnl)
        print('delta_gamma_theta_pnl', delta_gamma_theta_pnl)
        delta_pnl_list.append(delta_pnl)
        delta_gamma_pnl_list.append(delta_gamma_pnl)
        delta_gamma_theta_pnl_list.append(delta_gamma_theta_pnl)
    return delta_pnl_list, delta_gamma_pnl_list, delta_gamma_theta_pnl_list, greeks_df