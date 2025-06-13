import torch
import torch.nn as nn




class DeepSets(nn.Module):
    def __init__(self, phi_dim, rho_dim, num_phi_layers, num_rho_layers, dropout_rate):
        super().__init__()

        # Build phi = varible-level network depening on config input
        phi_layers = []
       
        phi_layers.append(nn.Linear(4, phi_dim))
        phi_layers.append(nn.BatchNorm1d(phi_dim)) # Normalize the input - see report for more details
        phi_layers.append(nn.ReLU())

        for _ in range(num_phi_layers - 1):
            phi_layers.append(nn.Linear(phi_dim, phi_dim))
            phi_layers.append(nn.BatchNorm1d(phi_dim))
            phi_layers.append(nn.ReLU())

        self.phi = nn.Sequential(*phi_layers)

        # Build rho = variable-set-level network depending on config input
        rho_layers = []
        rho_layers.append(nn.Linear(2 * phi_dim, rho_dim))
        rho_layers.append(nn.BatchNorm1d(rho_dim))
        rho_layers.append(nn.ReLU())
        rho_layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_rho_layers - 2):
            rho_layers.append(nn.Linear(rho_dim, rho_dim))
            rho_layers.append(nn.BatchNorm1d(rho_dim))
            rho_layers.append(nn.ReLU())
            rho_layers.append(nn.Dropout(dropout_rate))

     
        rho_layers.append(nn.Linear(rho_dim, 1))

        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x, mask=None):

        batch_size, max_hits, _ = x.shape

       
        hits_flat = x.view(batch_size * max_hits, -1)
        phi_flat  = self.phi(hits_flat)                         
        phi_out   = phi_flat.view(batch_size, max_hits, -1)    

       
        if mask is not None: # masking - see report for more details
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, phi_out.size(-1))  
            phi_out = phi_out * mask_expanded

       
        summed = phi_out.sum(dim=1) # Sum pooling across the variable dimension

        maxed, _ = phi_out.max(dim=1) # Max pooling across the variable dimension

    
        pooled = torch.cat([summed, maxed], dim=1)  # Concatenate along the feature dimension - Pooling

        out = self.rho(pooled).squeeze(-1)     
        return out