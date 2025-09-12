import torch
import numpy as np
import pandas as pd
from models import Encoder, DynamicsModel
from cart_dataloader import TrajectoryRollout
import os
from tqdm import tqdm

# Config
LATENT_DIM = 3
print(f"--- Running analysis for a {LATENT_DIM}D Latent Space ---")

if LATENT_DIM == 2:
    MODEL_DIR = "saved_models_2D"
    ROA_FILE = "output/MG_RoA_.csv"
    DESIRABLE_NODE = {11, 23}
    FAILURE_NODES = {0, 1, 3, 9, 5, 6, 10}
    node_mappings = {
        31:[9, 10, 11, 6, 23], 30:[10, 23], 29:[10, 23], 28:[10, 23], 27:[10, 23], 26:[23], 25:[6, 11, 23], 24:[6, 11, 23], 23:[23],
        22:[6, 11], 21:[0, 3, 1, 9, 5, 6, 10, 11], 20:[1, 5, 6], 19:[3, 9, 5, 10, 11], 18:[3, 9, 5, 10, 11], 17:[3, 9], 16:[3, 9], 15:[9, 5, 6, 10, 11], 14:[9, 10], 13:[10, 11], 12:[6, 11], 11:[11],
        10:[10],
        9:[9],
        8:[1, 5 , 6], 7:[5, 6], 6:[6],
        5:[5],
        4:[0, 3],
        2:[0, 1],
        1:[1],
        0:[0]

    }
elif LATENT_DIM == 3:
    MODEL_DIR = "saved_models_3D"
    ROA_FILE = "3Doutput/MG_RoA_.csv"
    DESIRABLE_NODE = {3}
    FAILURE_NODES = {0, 1}
    node_mappings = {
        4:[0,1,3], 3:[3],
        2:[0,1], 1:[1],
        0:[0]
    }
else:
    raise ValueError(f"LATENT_DIM of {LATENT_DIM} is not configured.")


ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DATA_DIR = "cartpole_data_images"
LABEL_FILE = "cartpole_data_success.txt"
STACK_SIZE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DESIRABLE_LABEL = 1


# Load ROA Map and Node Mappings for N-Dimensional Latent Space
def load_roa_map_nd(roa_csv_path, latent_dim):
    print(f"--- Loading {latent_dim}D ROA map from {roa_csv_path} ---")
    try:
        df = pd.read_csv(roa_csv_path, skiprows=3, header=None)
        
        headers = ['Tile', 'Morse_node']
        coord_names = ['x', 'y', 'z', 'a', 'b', 'c'] 
        min_headers = [f'{coord_names[i]}_min' for i in range(latent_dim)]
        max_headers = [f'{coord_names[i]}_max' for i in range(latent_dim)]
        
        headers = ['Tile', 'Morse_node']
        headers.extend(min_headers)
        headers.extend(max_headers)

        if len(df.columns) != len(headers):
             df.columns = headers[:len(df.columns)] 
        else:
             df.columns = headers

        cols_to_convert = headers[1:] 
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(inplace=True)
        print(f"Successfully loaded and parsed {latent_dim}D ROA map.")
        return df
    except Exception as e:
        print(f"!!! ERROR: Could not process the ROA file: {e}")
        return None

# Finds the Morse node for a given N-dimensional latent vector.
def find_morse_node_nd(latent_vector, roa_map, latent_dim):
    coord_names = ['x', 'y', 'z', 'a', 'b', 'c']
    query = pd.Series(True, index=roa_map.index)
    
    for i in range(latent_dim):
        coord = latent_vector[i]
        min_col = f'{coord_names[i]}_min'
        max_col = f'{coord_names[i]}_max'
        query &= (roa_map[min_col] <= coord) & (coord < roa_map[max_col])
    
    result = roa_map.loc[query]
    if not result.empty:
        return int(result.iloc[0]['Morse_node'])
    return None

# Load Models and Dataset
print("--- Loading necessary files ---")
encoder = Encoder(latent_dim=LATENT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=True))
encoder.eval()
dataset = TrajectoryRollout(data_dir=DATA_DIR, stack_size=STACK_SIZE, label_file=LABEL_FILE, max_trajectories=288)
roa_df = load_roa_map_nd(ROA_FILE, LATENT_DIM)

if roa_df is None:
    print("\nCould not proceed. Please check file paths and formats.")
else:
    latent_pairs = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Encoding all trajectories"):
            initial_stack, final_stack, label, _ = dataset[i]
            initial_stack_tensor = initial_stack.unsqueeze(0).to(DEVICE).float()
            final_stack_tensor = final_stack.unsqueeze(0).to(DEVICE).float()
            
            initial_latent = encoder(initial_stack_tensor).cpu().numpy().squeeze()
            if initial_latent.ndim == 0: # Ensure it's an array
                initial_latent = np.array([initial_latent])

            final_latent = encoder(final_stack_tensor).cpu().numpy().squeeze()
            if final_latent.ndim == 0: # Ensure it's an array
                final_latent = np.array([final_latent])
            
            latent_pairs.append({
                "initial_latent": initial_latent,
                "final_latent": final_latent,
                "label": label
            })

    # Evaluation
    print("\n" + "="*40)
    print("--- EVALUATION OF ROA PREDICTIONS ---")
    print("="*40)
    tp, fp, fn, tn = 0, 0, 0, 0
    unknown_states = 0
    
    for pair in tqdm(latent_pairs, desc="Evaluating trajectories"):
        initial_node = find_morse_node_nd(pair['initial_latent'], roa_df, LATENT_DIM)

        if initial_node is None:
            unknown_states += 1
            continue
        
        if initial_node == -1:
            continue

        possible_destinations = set(node_mappings.get(initial_node, []))

        # Case 1: The trajectory was a success
        if pair['label'] == DESIRABLE_LABEL:
            if not possible_destinations.isdisjoint(DESIRABLE_NODE):
                tp += 1
            else:
                fn += 1
        # Case 2: The trajectory was a failure
        else:
            if not possible_destinations.isdisjoint(FAILURE_NODES):
                tn += 1
            else:
                fp += 1

    # P, R, F-score
    print("\n--- Evaluation Results ---")
    print(f"Total Test Samples: {len(latent_pairs)}")
    print(f"  - True Positives (TP): {tp}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - True Negatives (TN): {tn}")
    print(f"  - False Negatives (FN): {fn}")
    
    if unknown_states > 0:
        print(f"  - Unknown States (out of map bounds): {unknown_states}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Performance Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F-score:   {f_score:.4f}")