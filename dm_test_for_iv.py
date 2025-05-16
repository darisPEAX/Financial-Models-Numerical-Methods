#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diebold-Mariano test for comparing predictive accuracy of option pricing models
Using the formula where d_t is the difference between squared errors of implied volatilities
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def diebold_mariano_test_iv(bsiv_market, bsiv_model1, bsiv_model2, h=1):
    """
    Performs the Diebold-Mariano test to compare the predictive accuracy of two forecasting models
    using the difference in squared errors of implied volatilities.
    
    Parameters:
    -----------
    bsiv_market : array-like
        The actual observed implied volatilities from the market.
    bsiv_model1 : array-like
        Implied volatilities from the first model.
    bsiv_model2 : array-like
        Implied volatilities from the second model.
    h : int, optional (default=1)
        The forecast horizon.
    
    Returns:
    --------
    stat : float
        The DM test statistic.
    p_value : float
        The p-value of the test.
    conclusion : str
        Interpretation of the test result.
    """
    # Ensure inputs are numpy arrays
    bsiv_market = np.asarray(bsiv_market)
    bsiv_model1 = np.asarray(bsiv_model1)
    bsiv_model2 = np.asarray(bsiv_model2)
    
    # Calculate squared errors for each model
    se1 = (bsiv_market - bsiv_model1)**2
    se2 = (bsiv_market - bsiv_model2)**2
    
    # Calculate the difference in squared errors (d_t in the formula)
    d = se1 - se2
    
    # Calculate mean of the loss differential
    d_bar = np.mean(d)
    
    # Calculate long-run variance of d using Newey-West estimator
    n = len(d)
    
    # Calculate autocovariances up to h-1 lags
    gamma_0 = np.mean(d**2) - d_bar**2  # Variance
    
    gamma = []
    for k in range(1, h):
        if k < n:
            gamma_k = np.mean(d[k:] * d[:-k]) - d_bar**2
            gamma.append(gamma_k)
    
    # Calculate long-run variance with Bartlett kernel weights
    v_d = gamma_0
    if len(gamma) > 0:
        v_d += 2 * sum([(1 - k/h) * gamma[k-1] for k in range(1, min(h, n))])
    
    # Calculate the test statistic (DM in the formula)
    dm_stat = d_bar / np.sqrt(v_d / n)
    
    # Calculate the p-value (two-tailed test)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n-1)
    
    # Interpret the result
    if p_value <= 0.01:
        significance = "at 1% significance level"
    elif p_value <= 0.05:
        significance = "at 5% significance level"
    elif p_value <= 0.10:
        significance = "at 10% significance level"
    else:
        significance = "not statistically significant"
    
    if dm_stat > 0:
        conclusion = f"Model 2 is more accurate than Model 1 {significance}"
    elif dm_stat < 0:
        conclusion = f"Model 1 is more accurate than Model 2 {significance}"
    else:
        conclusion = "Both models have equal predictive accuracy"
        
    return dm_stat, p_value, conclusion

def compare_models_with_mse(mse_list_model1, mse_list_model2, h=5, model1_name="Model 1", model2_name="Model 2"):
    """
    Compares two models using their MSE lists and performs the Diebold-Mariano test.
    
    Parameters:
    -----------
    mse_list_model1 : list or array
        List of MSE values for the first model.
    mse_list_model2 : list or array
        List of MSE values for the second model.
    h : int, optional (default=5)
        The forecast horizon.
    model1_name : str, optional
        Name of the first model.
    model2_name : str, optional
        Name of the second model.
    
    Returns:
    --------
    dm_stat : float
        The DM test statistic.
    p_value : float
        The p-value of the test.
    conclusion : str
        Interpretation of the test result.
    """
    # Ensure lists are of equal length
    if len(mse_list_model1) != len(mse_list_model2):
        raise ValueError("MSE lists must be of equal length")
    
    # For the DM test using MSE, we need to calculate d as the difference between MSEs
    # Here we're assuming each MSE in the list is already a squared error measurement
    # If MSEs represent aggregate measures over multiple options, this would require adjustment
    d = np.array(mse_list_model1) - np.array(mse_list_model2)
    
    # Calculate mean of the loss differential
    d_bar = np.mean(d)
    
    # Calculate long-run variance of d using Newey-West estimator
    n = len(d)
    
    # Calculate autocovariances up to h-1 lags
    gamma_0 = np.mean(d**2) - d_bar**2  # Variance
    
    gamma = []
    for k in range(1, h):
        if k < n:
            gamma_k = np.mean(d[k:] * d[:-k]) - d_bar**2
            gamma.append(gamma_k)
    
    # Calculate long-run variance with Bartlett kernel weights
    v_d = gamma_0
    if len(gamma) > 0:
        v_d += 2 * sum([(1 - k/h) * gamma[k-1] for k in range(1, min(h, n))])
    
    # Calculate the test statistic
    dm_stat = d_bar / np.sqrt(v_d / n)
    
    # Calculate the p-value (two-tailed test)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n-1)
    
    # Interpret the result
    if p_value <= 0.01:
        significance = "at 1% significance level"
    elif p_value <= 0.05:
        significance = "at 5% significance level"
    elif p_value <= 0.10:
        significance = "at 10% significance level"
    else:
        significance = "not statistically significant"
    
    if dm_stat > 0:
        conclusion = f"{model2_name} is more accurate than {model1_name} {significance}"
    elif dm_stat < 0:
        conclusion = f"{model1_name} is more accurate than {model2_name} {significance}"
    else:
        conclusion = f"Both {model1_name} and {model2_name} have equal predictive accuracy"
    
    # Print results
    print(f"\nDiebold-Mariano Test Results: {model1_name} vs {model2_name}")
    print(f"DM Statistic: {dm_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Conclusion: {conclusion}")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(mse_list_model1, 'b-', label=f'{model1_name} MSE')
    plt.plot(mse_list_model2, 'r-', label=f'{model2_name} MSE')
    plt.axhline(y=np.mean(mse_list_model1), color='b', linestyle='--', 
                label=f'{model1_name} Mean: {np.mean(mse_list_model1):.4f}')
    plt.axhline(y=np.mean(mse_list_model2), color='r', linestyle='--', 
                label=f'{model2_name} Mean: {np.mean(mse_list_model2):.4f}')
    plt.xlabel('Observation')
    plt.ylabel('MSE')
    plt.title(f'MSE Comparison: {model1_name} vs {model2_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'dm_test_{model1_name}_vs_{model2_name}.png')
    plt.close()
    
    return dm_stat, p_value, conclusion

def example_usage():
    """
    Example showing how to use the DM test with existing MSE lists.
    """
    # Example MSE lists from the SVI model
    MSE_SVI_1 = [0.0043, 0.0039, 0.0041, 0.0038, 0.0042, 0.0044, 0.0040, 0.0037]
    
    # Example MSE lists from the AHBS model
    MSE_AHBS_1 = [0.0047, 0.0045, 0.0044, 0.0046, 0.0043, 0.0048, 0.0042, 0.0041]
    
    # Compare models using MSE lists
    dm_stat, p_value, conclusion = compare_models_with_mse(
        MSE_SVI_1, MSE_AHBS_1, h=2, model1_name="SVI", model2_name="AHBS"
    )
    
    # Example with real data from the codebase
    # Replace this with your actual MSE lists
    """
    # Import your MSE lists from previous analysis
    from A_svi import MSE_SVI_1_oos, MSE_SVI_7_oos, MSE_SVI_30_oos
    from A_ahbs import MSE_AHBS_1_oos, MSE_AHBS_7_oos, MSE_AHBS_30_oos
    
    # Compare 1-day options
    compare_models_with_mse(
        MSE_SVI_1_oos, MSE_AHBS_1_oos, h=1, model1_name="SVI-1d", model2_name="AHBS-1d"
    )
    
    # Compare 7-day options
    compare_models_with_mse(
        MSE_SVI_7_oos, MSE_AHBS_7_oos, h=5, model1_name="SVI-7d", model2_name="AHBS-7d"
    )
    
    # Compare 30-day options
    compare_models_with_mse(
        MSE_SVI_30_oos, MSE_AHBS_30_oos, h=22, model1_name="SVI-30d", model2_name="AHBS-30d"
    )
    """

if __name__ == "__main__":
    example_usage()

    # Import your MSE lists from previous analysis
    MSE_SVI_1_oos = pd.read_csv("data/MSE_SVI_1_oos.csv")["MSE_SVI_1_oos"].values
    MSE_SVI_7_oos = pd.read_csv("data/MSE_SVI_7_oos.csv")["MSE_SVI_7_oos"].values
    MSE_SVI_30_oos = pd.read_csv("data/MSE_SVI_30_oos.csv")["MSE_SVI_30_oos"].values

    MSE_AHBS_1_oos = pd.read_csv("data/MSE_AHBS_1_oos.csv")["MSE_AHBS_1_oos"].values
    MSE_AHBS_7_oos = pd.read_csv("data/MSE_AHBS_7_oos.csv")["MSE_AHBS_7_oos"].values
    MSE_AHBS_30_oos = pd.read_csv("data/MSE_AHBS_30_oos.csv")["MSE_AHBS_30_oos"].values

    # Compare models using MSE lists
    dm_stat, p_value, conclusion = compare_models_with_mse(
        MSE_SVI_1_oos, MSE_AHBS_1_oos, h=1, model1_name="SVI-1d", model2_name="AHBS-1d"
    )
    print(dm_stat, p_value, conclusion)

    dm_stat, p_value, conclusion = compare_models_with_mse(
        MSE_SVI_7_oos, MSE_AHBS_7_oos, h=5, model1_name="SVI-7d", model2_name="AHBS-7d"
    )
    print(dm_stat, p_value, conclusion)

    dm_stat, p_value, conclusion = compare_models_with_mse(
        MSE_SVI_30_oos, MSE_AHBS_30_oos, h=22, model1_name="SVI-30d", model2_name="AHBS-30d"
    )
    print(dm_stat, p_value, conclusion)
    
    
    
    
    