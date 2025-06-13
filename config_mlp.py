# Be sure to configure this file before running the model.


Layers = 8                   # Number of layers in the model
max_Dimension = 4096          # Must be a power of 2
Droupout_Rate = 0.2
 

# Training parameters
num_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 32


maximum_length = 64

input_Dimension = maximum_length * 4      # * 4 do not change unless hit format changes


# Data
feature_path = "trident_data/features.parquet"
truth_path = "trident_data/truth_minimal.parquet"
truth_key = "initial_state_energy"


# Output paths
results_dir = "TridentMLResults"
saved_models_dir = "savedmodels"


