import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm
from models import Encoder
from humanoid_dataloader import TrajectoryRollout

# Config
LATENT_DIM = 3

print(f"--- Running analysis for a {LATENT_DIM}D Latent Space ---")

if LATENT_DIM == 2:
    MODEL_DIR = "saved_models_2D"
    ROA_FILE = "output/MG_RoA_.csv"
elif LATENT_DIM == 3:
    MODEL_DIR = "saved_models_3D"
    ROA_FILE = "3Doutput/MG_RoA_.csv"
else:
    MODEL_DIR = f"saved_models_{LATENT_DIM}D"
    ROA_FILE = f"{LATENT_DIM}Doutput/MG_RoA_.csv"

ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DATA_DIR = "humanoid_data_images"
LABEL_FILE = "humanoid_data_success.txt"
STACK_SIZE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DESIRABLE_LABEL = 1


# Load ROA Map from MG_ROA file
def load_roa_map_nd(roa_csv_path, latent_dim):
    print(f"--- Loading {latent_dim}D ROA map from {roa_csv_path} ---")
    try:
        df = pd.read_csv(roa_csv_path, skiprows=3, header=None)
        
        headers = ['Tile', 'Morse_node']
        min_headers = [f'x{i}_min' for i in range(latent_dim)]
        max_headers = [f'x{i}_max' for i in range(latent_dim)]
        headers.extend(min_headers)
        headers.extend(max_headers)
        
        if len(df.columns) != len(headers):
             raise ValueError(f"Expected {len(headers)} columns for {latent_dim}D data, but file has {len(df.columns)} columns.")

        df.columns = headers
        
        for col in headers[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(inplace=True)
        print(f"Successfully loaded and parsed {latent_dim}D ROA map.")
        return df
    except Exception as e:
        print(f"!!! ERROR: Could not process the ROA file: {e}")
        return None

# Find Morse node for a given latent vector in N-D ROA map
def find_morse_node_nd(latent_vector, roa_map, latent_dim):
    
    query = pd.Series(True, index=roa_map.index)
    
    for i in range(latent_dim):
        coord = latent_vector[i]
        min_col = f'x{i}_min'
        max_col = f'x{i}_max'
        query &= (roa_map[min_col] <= coord) & (coord < roa_map[max_col])
    
    result = roa_map.loc[query]
    
    if not result.empty:
        return int(result.iloc[0]['Morse_node'])
    return None

# Encoder, Dataset, ROA map
print("--- Loading necessary files ---")
encoder = Encoder(latent_dim=LATENT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=True))
encoder.eval()
dataset = TrajectoryRollout(data_dir=DATA_DIR, stack_size=STACK_SIZE, label_file=LABEL_FILE, max_trajectories=199)
roa_df = load_roa_map_nd(ROA_FILE, LATENT_DIM)

if roa_df is None:
    print("\nCould not proceed. Please check file paths and formats.")
else:
    # Find the Morse node for the final state of every trajectory
    final_state_nodes = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Analyzing final states"):
            _, final_stack, label, _ = dataset[i]
            final_stack_tensor = final_stack.unsqueeze(0).to(DEVICE).float()
            final_latent = encoder(final_stack_tensor).cpu().numpy().squeeze()
            
            if final_latent.ndim == 0:
                final_latent = np.array([final_latent])
                
            node = find_morse_node_nd(final_latent, roa_df, LATENT_DIM)
            if node is not None:
                final_state_nodes.append({"label": label, "final_node": node})

    # Separate nodes based on the ground-truth label
    successful_final_nodes = [item['final_node'] for item in final_state_nodes if item['label'] == DESIRABLE_LABEL]
    failed_final_nodes = [item['final_node'] for item in final_state_nodes if item['label'] != DESIRABLE_LABEL]

    # Nodes Final States belong to (to find attractors)
    print("\n" + "="*40)
    print("--- ATTRACTOR ANALYSIS RESULTS ---")
    print("="*40)

    if successful_final_nodes:
        success_counts = pd.Series(successful_final_nodes).value_counts()
        print("\n Frequency of nodes for FINAL states of successful runs:")
        print(success_counts)
        print(f"\n   Possible Desirable Nodes: {sorted(list(success_counts.index))}")
    else:
        print("\nNo successful final states were found within the ROA map.")

    if failed_final_nodes:
        failure_counts = pd.Series(failed_final_nodes).value_counts()
        print("\n Frequency of nodes for FINAL states of failed runs:")
        print(failure_counts)
        print(f"\n   Possible Undesirable Nodes: {sorted(list(failure_counts.index))}")
    else:
        print("\nNo failed final states were found within the ROA map.")