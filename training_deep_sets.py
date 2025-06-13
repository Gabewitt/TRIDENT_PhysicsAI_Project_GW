from models.deepset import DeepSets
from utils.utils import TridentDataset
from config_deep_sets import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import os

# Load dataset
dataset = TridentDataset(
    feature_path=feature_path,
    truth_path=truth_path,
    truth_keys=truth_key,
    maximum_length=maximum_length
)

# Log-transform the energy 
dataset._truth = torch.log10(dataset.truth)

# Split into train/val/test

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Initialize model and optimizer (ADAM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSets(
    phi_dim=phi_Dim,
    rho_dim=rho_Dim,
    num_phi_layers=num_Phi_layers,
    num_rho_layers=num_Rho_layers,
    dropout_rate=Droupout_Rate
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # Reduce learning rate when validation loss plateaus

train_losses = []
val_losses = []
log_lines = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        x = batch["features"].to(device)
        y = batch["truth"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        pred = model(x, mask)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["features"].to(device)
            y = batch["truth"].to(device)
            mask = batch["mask"].to(device)

            pred = model(x, mask)
            loss = loss_fn(pred, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr'] 
    print(f"Epoch {epoch+1:2d} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
    log_lines.append(f"Epoch {epoch+1} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}\n")


# Save training log

os.makedirs(results_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H")
log_name = f"DeepSets_Traning_Log_{Layers}_Layers_{phi_Dim}_PhiDim_{rho_Dim}_RhoDim_{Droupout_Rate}_DropoutRate_{timestamp}.txt"
output_file = os.path.join(results_dir, "DeepSets", log_name)
os.makedirs(os.path.dirname(output_file), exist_ok=True)


model_file_name = f"DeepSets_Layers_{Layers}_Phi_{phi_Dim}_Rho_{rho_Dim}_Epoch_{num_epochs}_Batch_{batch_size}.pth"

with open(output_file, "w") as f:
    f.writelines(log_lines)
    f.write(f"\nModel save path: {model_file_name}\n")
print(f"\nTraining log saved to {output_file}")



# Save model
model_save_path = os.path.join(saved_models_dir, model_file_name)
torch.save(model.state_dict(), model_save_path)

