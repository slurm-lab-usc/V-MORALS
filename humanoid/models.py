import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D Conv encoder for 6 binary frames of size 128x128.
# Input:  (B, 6, 128, 128)
class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(6,4,4), stride=(1,2,2), padding=(0,1,1)) 
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))  
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) 
        self.conv4 = nn.Conv3d(64, 128, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) 
        
        self.fc_enc = nn.Linear(128 * 1 * 8 * 8, latent_dim)  

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))  
        x = F.relu(self.conv4(x)) 
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return torch.tanh(z)

# Output: (B, 6, 128, 128)
class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc_dec = nn.Linear(latent_dim, 128 * 1 * 8 * 8)
        
        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.deconv4 = nn.ConvTranspose3d(16, 1,  kernel_size=(6,4,4), stride=(1,2,2), padding=(0,1,1))

    def forward(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(x.size(0), 128, 1, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x)) 
        x = F.relu(self.deconv3(x)) 
        x = torch.sigmoid(self.deconv4(x))  
        return x.squeeze(1) 

# A simple feedforward dynamics model that predicts the next latent state from the current latent state.
class DynamicsModel(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z):
        return torch.tanh(self.net(z))