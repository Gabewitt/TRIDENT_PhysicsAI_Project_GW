import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import compute_mae_rmse, compute_relative_errors, count_error_ranges
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
import inspect



def evaluate_model(model, test_loader):
    model.eval()
    true_vals, pred_vals = [], []

    device = next(model.parameters()).device


    forward_params = inspect.signature(model.forward).parameters

    with torch.no_grad():
        for batch in test_loader:
            x     = batch["features"].to(device)
            y_log = batch["truth"].to(device)

            
            if "mask" in batch and "mask" in forward_params:
                mask = batch["mask"].to(device)
                pred_log = model(x, mask)
            else:
                pred_log = model(x)

        
            y    = 10 ** y_log
            pred = 10 ** pred_log

            true_vals.extend(y.cpu().numpy())
            pred_vals.extend(pred.cpu().numpy())

    return np.array(true_vals), np.array(pred_vals)




def plot_prediction_scatter(true_vals, pred_vals, save_path, title):
    plt.figure(figsize=(7, 6), dpi=1000)
    plt.scatter(true_vals, pred_vals, s=10, alpha=0.5)
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--', label="Perfect Prediction")
    plt.xlabel("True Energy (GeV)")
    plt.ylabel("Predicted Energy (GeV)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()

def plot_error_histogram(true_vals, pred_vals, save_path):
    relative_errors = compute_relative_errors(true_vals, pred_vals)
    mae, rmse = compute_mae_rmse(true_vals, pred_vals)

    plt.figure(figsize=(10, 6), dpi=1000)
    plt.hist(relative_errors, bins=500, alpha=0.7, color='blue', edgecolor='black',
             label=f"Number of Hits: {len(true_vals)}")
    plt.axvline(0, color='red', linestyle='--', label="Perfect Prediction")
    plt.text(30, plt.ylim()[1] * 0.9, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Number of Events")
    plt.xlim(-100, 300)
    plt.title("Distribution of Prediction Errors (Relative to True Energy)\nMLP Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()

def save_accuracy_breakdown(true_vals, pred_vals, output_path, model_name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d at %H")
    relative_errors = compute_relative_errors(true_vals, pred_vals)
    mae, rmse = compute_mae_rmse(true_vals, pred_vals)
    counts = count_error_ranges(relative_errors)
    total_predictions = len(true_vals)

    results_text = f"""
{model_name} Model Evaluation Results:
Generated on {timestamp}

Total Predictions: {total_predictions}
MAE (Mean Absolute Error): {mae:.2f}
RMSE (Root Mean Squared Error): {rmse:.2f}

Predictions within:
"""
    for key, count in counts.items():
        percentage = count / total_predictions * 100
        results_text += f"- {key.replace('_', ' ')}: {count} ({percentage:.2f}%)\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(results_text)

    print(f"\nResults saved to {output_path}")