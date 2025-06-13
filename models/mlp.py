import torch
import torch.nn as nn


class TridentRegressor(nn.Module):
    def __init__(self, num_layers, max_dim, input_dim, dropout_rate):
        super().__init__()


        # Setting up the MLP architecture depending on config input
        
        layers = []


        layers.append(nn.Linear(input_dim, max_dim))
        layers.append(nn.BatchNorm1d(max_dim))  # Normalize the input - see report for more details
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        current_dim = max_dim


        for _ in range(num_layers - 2):  
            next_dim = max(current_dim // 2, 1)  
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = next_dim

       
        layers.append(nn.Linear(current_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.model(x).squeeze()