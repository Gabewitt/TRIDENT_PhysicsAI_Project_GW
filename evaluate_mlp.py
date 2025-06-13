from models.mlp import TridentRegressor
from utils.utils import TridentDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
import torch, numpy as np, os
from system import *
from config_mlp import *
from utils.utils import TridentDataset
from torch.utils.data import DataLoader



# Load the model and dataset
dataset = TridentDataset(
    feature_path=feature_path,
    truth_path=truth_path,
    truth_keys=truth_key,
    maximum_length=maximum_length
)
dataset._truth = torch.log10(dataset.truth)

# Splitting the dataset into training, validation, and test sets 60% train, 20% validation, 20% test
_, _, test_set = torch.utils.data.random_split(dataset, [int(0.6 * len(dataset)), int(0.2 * len(dataset)), len(dataset) - int(0.6 * len(dataset)) - int(0.2 * len(dataset))])
test_loader = DataLoader(test_set, batch_size=batch_size)


model = TridentRegressor(Layers, max_Dimension, input_Dimension, Droupout_Rate)

model_save_path = f"savedmodels/MLP_Layers_{Layers}_InputDim{input_Dimension}_MaxDim_{max_Dimension}_Epochs_{num_epochs}_Batch_{batch_size}.pth"

model.load_state_dict(torch.load(f"savedmodels/MLP_Layers_{Layers}_InputDim{input_Dimension}_MaxDim_{max_Dimension}_Epochs_{num_epochs}_Batch_{batch_size}.pth"))

model = model.to("cuda" if torch.cuda.is_available() else "cpu")



true_vals, pred_vals = evaluate_model(model, test_loader)

# Saving results and plotting

timestamp = datetime.now().strftime("%Y-%m-%d_%H")
timestamp2 = datetime.now().strftime("%Y-%m-%d")

folder_name = f"MLP_Layers_{Layers}_InputDim_{input_Dimension}_MaxDim_{max_Dimension}_Epochs_{num_epochs}_Batch_{batch_size}_{timestamp2}"

os.makedirs(f"TridentMLResults/MLP/{folder_name}", exist_ok=True)

base_name = f"MLP_{Layers}_Layers_{max_Dimension}_max_dim_{input_Dimension}_input_dim_{num_epochs}_epochs_{timestamp}"

plot_prediction_scatter(
    true_vals, pred_vals,
    f"TridentMLResults/MLP/{folder_name}/{base_name}_Plot.png",
    "MLP: Predicted vs True Neutrino Energy (LogLog)"
)

plot_error_histogram(
    true_vals, pred_vals,
    f"TridentMLResults/MLP/{folder_name}/{base_name}_Histogram.png"
)



save_accuracy_breakdown(
    true_vals, pred_vals,
    output_path=f"TridentMLResults/MLP/{folder_name}/{base_name}_metrics.txt",
    model_name=model_save_path
)