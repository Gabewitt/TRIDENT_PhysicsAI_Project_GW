import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from models.deepset import DeepSets
from utils.utils import TridentDataset
from config_deep_sets import *
from system import *


# Load dataset
dataset = TridentDataset(
    feature_path=feature_path,
    truth_path=truth_path,
    truth_keys=truth_key,
    maximum_length=maximum_length
)
dataset._truth = torch.log10(dataset.truth)

# Splitting the dartaset into train, validation, and test sets 60% train, 20% validation, 20% test
_, _, test_set = torch.utils.data.random_split(
    dataset, [int(0.6 * len(dataset)), int(0.2 * len(dataset)), len(dataset) - int(0.6 * len(dataset)) - int(0.2 * len(dataset))]
)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Load model
model = DeepSets(
    phi_dim=phi_Dim,
    rho_dim=rho_Dim,
    num_phi_layers=num_Phi_layers,
    num_rho_layers=num_Rho_layers,
    dropout_rate=Droupout_Rate
)


model_path = f"savedmodels/DeepSets_Layers_{Layers}_Phi_{phi_Dim}_Rho_{rho_Dim}_Epoch_{num_epochs}_Batch_{batch_size}.pth"
model.load_state_dict(torch.load(model_path))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate
true_vals, pred_vals = evaluate_model(model, test_loader)


# Saving results
timestamp = datetime.now().strftime("%Y-%m-%d_%H")
timestamp2 = datetime.now().strftime("%Y-%m-%d")


folder_name = f"DeepSets_Layers_{Layers}_Phi_{phi_Dim}_Rho_{rho_Dim}_Epochs_{num_epochs}_Batch_{batch_size}_{timestamp2}"

os.makedirs(f"TridentMLResults/DeepSets/{folder_name}", exist_ok=True)

base_name = f"DeepSets_{Layers}Layers_{phi_Dim}_phi_{rho_Dim}_rho_{num_epochs}epochs_{timestamp}"


plot_prediction_scatter(
    true_vals, pred_vals,
    f"TridentMLResults/DeepSets/{folder_name}/{base_name}_Plot.png",
    f"Deep Sets: Predicted vs True Energy (Log Scale) - {Layers} Layers, {phi_Dim} phi, {rho_Dim} rho, {num_epochs} epochs"
)


plot_error_histogram(
    true_vals, pred_vals,
    f"TridentMLResults/DeepSets/{folder_name}/{base_name}_Histogram.png"
)


save_accuracy_breakdown(
    true_vals, pred_vals,
    output_path=f"TridentMLResults/DeepSets/{folder_name}/{base_name}_metrics.txt",
    model_name=model_path
)