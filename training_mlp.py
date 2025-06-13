from models.mlp import TridentRegressor
from utils.utils import TridentDataset
from config_mlp import feature_path, truth_path, truth_key, maximum_length, batch_size, Layers, max_Dimension, input_Dimension, Droupout_Rate, num_epochs, learning_rate, weight_decay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import os
import numpy as np


dataset = TridentDataset(
    feature_path=feature_path,
    truth_path=truth_path,
    truth_keys=truth_key,
    maximum_length=maximum_length
)

# Log-transform the energy labels
dataset._truth = torch.log10(dataset.truth)

# Split into train/val/test
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)




model = TridentRegressor(num_layers=Layers, max_dim=max_Dimension, input_dim= input_Dimension, dropout_rate=Droupout_Rate).to("cuda" if torch.cuda.is_available() else "cpu") #initializing the model and moves it to the GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #Adaptive Moment Estimation optimizer
loss_fn = nn.MSELoss() #MEan Squared Error Loss function

#schedular for learning rate - if the validation loss does not improve for 5 epochs, the learning rate is reduced by a factor of 10
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_losses, val_losses = [], []
log_lines = []


for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for batch in train_loader: #takes training data in batches
        x = batch["features"].to(model.model[0].weight.device) #hits
        y = batch["truth"].to(model.model[0].weight.device) #truth log

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward() #calculates the gradients of the loss with respect to the model parameters
        optimizer.step() #updates the weights of the model
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    
    model.eval() #sees how the model performs on unseen data aka validation data
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["features"].to(model.model[0].weight.device)
            y = batch["truth"].to(model.model[0].weight.device)
            val_loss += loss_fn(model(x), y).item()
    
    val_losses.append(val_loss / len(val_loader))
    
    scheduler.step(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} — Train Loss: {train_loss:.4f}, Val Loss: {val_losses[-1]:.4f}, LR: {current_lr:.6f}")
    log_lines.append(f"Epoch {epoch+1} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}\n")


output_folder = "TridentMLResults/MLP/"
os.makedirs(output_folder, exist_ok=True)


now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H")
name_of_file = "MLP_Traning_Log_" + str(Layers) + "_Layers_" + str(max_Dimension) + "_MaxDim_" + str(input_Dimension) + "_InputDim_" + str(Droupout_Rate) + "_DropoutRate_" + str(timestamp) + ".txt"

output_file = os.path.join(output_folder, name_of_file)

model_save_path = f"savedmodels/MLP_Layers_{Layers}_InputDim{input_Dimension}_MaxDim_{max_Dimension}_Epochs_{num_epochs}_Batch_{batch_size}.pth"


with open(output_file, "w") as f:
    f.writelines(log_lines)
    f.write(f"\nmodel_save_path: {model_save_path}\n")

print(f"\nTraining log saved to {output_file}")




torch.save(model.state_dict(), model_save_path)