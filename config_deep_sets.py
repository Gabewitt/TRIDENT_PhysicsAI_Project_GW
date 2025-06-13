# Be sure to configure this file before running the model.


Layers = 4                        # Total number of layers for both phi and rho
phi_Dim = 64                      # Dimension for per-hit 
rho_Dim = 128                      # Dimension for event 
num_Phi_layers = Layers          
num_Rho_layers = Layers           
Droupout_Rate = 0.1           

# Training parameters
num_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 32


maximum_length = 64                # Maximum number of hits per event, must be a power of 2 
input_Dimension = maximum_length * 4          # Input dimension per hit (x, y, z, t), do not change unless hit format changes

# Data paths
feature_path = "trident_data/features.parquet"
truth_path = "trident_data/truth_minimal.parquet"
truth_key = "initial_state_energy"

# Output paths
results_dir = "TridentMLResults"
saved_models_dir = "savedmodels"