import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Encoder
from pend_dataloader import PendulumSequenceDataset
from torch.utils.data import DataLoader
import os

# Config
LATENT_DIM = 2
MODEL_DIR = "saved_models"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DATA_DIR = "pendulum_data_images"
STACK_SIZE = 3
BATCH_SIZE = 64
NUM_SAMPLES = 10000  # Number of points to plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder
encoder = Encoder(latent_dim=LATENT_DIM).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()

# Load dataset and dataloader
dataset = PendulumSequenceDataset(data_dir=DATA_DIR, stack_size=STACK_SIZE, tau=1)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

latent_points1 = []
latent_points2 = []
labels = []

with torch.no_grad():
    samples_collected = 0
    for stack1, stack2, label in dataloader:
        stack1 = stack1.to(device).float()
        stack2 = stack2.to(device).float()
        z1 = encoder(stack1)
        z2 = encoder(stack2)
        latent_points1.append(z1.cpu().numpy())
        latent_points2.append(z2.cpu().numpy())
        labels.append(label.numpy())
        samples_collected += stack1.size(0)
        if samples_collected >= NUM_SAMPLES:
            break

latent_points1 = np.concatenate(latent_points1, axis=0)[:NUM_SAMPLES]
latent_points2 = np.concatenate(latent_points2, axis=0)[:NUM_SAMPLES]
labels = np.concatenate(labels, axis=0)[:NUM_SAMPLES]

# 2D plot (first two latent dimensions)
fig2d = plt.figure(figsize=(8, 8))
ax2d = fig2d.add_subplot(111)
colors = ['purple' if l == 0 else 'green' for l in labels]
ax2d.scatter(latent_points1[:, 0], latent_points1[:, 1], c=colors, edgecolor='k', marker='o', label='stack1')
ax2d.scatter(latent_points2[:, 0], latent_points2[:, 1], c=colors, edgecolor='k', marker='^', label='stack2')
ax2d.set_xlabel('Latent dim 1')
ax2d.set_ylabel('Latent dim 2')
ax2d.set_title('Encoded Stacks1 (o) and Stacks2 (^) in 2D Latent Space')
ax2d.legend(['stack1', 'stack2'])
plt.show()