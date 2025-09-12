import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Encoder, DynamicsModel
from pend_dataloader import PendulumSequenceDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from datetime import datetime
import dytop.CMGDB_util as CMGDB_util
import dytop.Grid as Grid
import dytop.dyn_tools as dyn_tools
#import PlotRoA
from dytop.PlotRoA import PlotRoA
import dytop.RoA as RoA       
from mpl_toolkits.mplot3d import Axes3D

# Config
LATENT_DIM = 4
MODEL_DIR = "saved_models_4D"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DYNAMICS_MODEL_PATH = os.path.join(MODEL_DIR, "dynamics_model.pt")
DATA_DIR = "pendulum_data_images"
STACK_SIZE = 5
BATCH_SIZE = 1
NUM_SAMPLES = 50000
GRID_SIZE = 10
ROLLOUT_STEPS = 24
sb = 16
subdiv_init = subdiv_min = subdiv_max = sb 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lower_bounds = [-1]*LATENT_DIM
upper_bounds = [1]*LATENT_DIM

grid = Grid.Grid(lower_bounds, upper_bounds, sb)
MG_util = CMGDB_util.CMGDB_util()

# Load models
encoder = Encoder(latent_dim=LATENT_DIM).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()

dynamics_model = DynamicsModel(latent_dim=LATENT_DIM).to(device)
dynamics_model.load_state_dict(torch.load(DYNAMICS_MODEL_PATH, map_location=device))
dynamics_model.eval()

# Load dataset and dataloader
dataset = PendulumSequenceDataset(data_dir=DATA_DIR, stack_size=STACK_SIZE, tau=1)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_SAMPLES = len(dataset)

latent_space_samples = []

print("Randomly sampling latent space points...")
latent_space_samples = np.random.uniform(
    low=-1, high=1, size=(NUM_SAMPLES, LATENT_DIM)
)

#latent_space_samples = np.concatenate(latent_space_samples, axis=0)

lower_bounds = np.min(latent_space_samples, axis=0).tolist()
upper_bounds = np.max(latent_space_samples, axis=0).tolist()

print(lower_bounds, upper_bounds)

valid_grid_latent_space = grid.valid_grid(latent_space_samples)

# Compute ROA
def compute_roa(map_graph, morse_graph, lower_bounds, upper_bounds, output_dir):

    startTime = datetime.now()

    roa = RoA.RoA(map_graph, morse_graph)

    print(f"Time to build the regions of attraction = {datetime.now() - startTime}")

    roa.dir_path = os.path.join(os.getcwd(), output_dir)

    roa.save_file("MG")

    fig, ax = PlotRoA(lower_bounds, upper_bounds, from_file="MG", dir_path=output_dir)

    out_pic = os.path.join(os.getcwd(), output_dir, "MG_RoA_")

    plt.savefig(out_pic, bbox_inches='tight')


def g(X):
    X = np.array(X, dtype=np.float32)   
    X_tensor = torch.from_numpy(X).to(device) 
    with torch.no_grad():
        for i in range(ROLLOUT_STEPS):
            next_z = dynamics_model(X_tensor)
            X_tensor = next_z
    return next_z.cpu().numpy().squeeze().tolist()

# print(g(latent_space_samples).shape)

phase_periodic = [False, False]

K = [1.1 * (1 + 10/100)] * LATENT_DIM

K = [0.5] * LATENT_DIM
  
def F(rect):
    return MG_util.BoxMapK_valid(g, rect, K, valid_grid_latent_space, grid.point2cell)

base_name = "4Doutput"

morse_graph, map_graph = MG_util.run_CMGDB(
    subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)

compute_roa(map_graph, morse_graph, lower_bounds, upper_bounds, base_name)