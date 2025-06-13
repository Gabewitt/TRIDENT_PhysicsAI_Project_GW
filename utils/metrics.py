import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error



def compute_mae_rmse(true_vals, pred_vals): # Compute Mean Absolute Error and Root Mean Squared Error
    
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    return mae, rmse

def compute_relative_errors(true_vals, pred_vals): # Compute Relative Errors as a percentage of true values
    
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    return (pred_vals - true_vals) / true_vals * 100

def count_error_ranges(relative_errors, thresholds=[1, 5, 10, 20]): # Count the number of predictions within specified error ranges
   
    counts = {}
    for t in thresholds:
        counts[f"within_{t}percent"] = np.sum(np.abs(relative_errors) <= t)
    return counts
