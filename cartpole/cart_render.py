import os
os.environ["MUJOCO_GL"] = "glfw"

import cv2
import numpy as np
from dm_control import suite
import tqdm
from PIL import Image

# Image rendering script for Cartpole from cartpole_data txt files

# Config
trajectory_dir = "cartpole_data"  
output_dir = "cartpole_data_images"
num_trajectories = 1799
threshold_val = 130

# Load cartpole environment
env = suite.load(domain_name="cartpole", task_name="swingup")
physics = env.physics
nq = physics.model.nq
nv = physics.model.nv

# Helper to read states
def load_states(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [list(map(float, line.split(','))) for line in lines]

# Iterate over all trajectories
for num in tqdm.tqdm(range(1, num_trajectories + 1), desc="Trajectories"):
    input_file = os.path.join(trajectory_dir, f"trajectory_{num}.txt")
    if not os.path.exists(input_file):
        print(f"Skipping {input_file}, file not found!")
        continue

    # Prepare output directory for this trajectory
    final_output_dir = os.path.join(output_dir, f"cartpole_trajectory_{num}")
    os.makedirs(final_output_dir, exist_ok=True)

    # Load states
    states = load_states(input_file)

    for idx, state in enumerate(states):
        qpos = np.array(state[:nq])
        qvel = np.array(state[nq:nq+nv])
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
        physics.forward()

        crop_top = 20   
        crop_bottom = 20  

        # Frame rendering
        frame = physics.render(height=120, width=170, camera_id=0)

        # Crop top and bottom
        frame = frame[crop_top:frame.shape[0]-crop_bottom, :, :]

        # Convert to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Threshold to create a mask
        mask = (img_gray > threshold_val).astype(np.uint8)

        # Apply mask and set all non-zero pixels to 255 (white)
        masked_img = mask * 255 

        # Double the size of the binary mask
        masked_pil = Image.fromarray(masked_img).convert('L')

        output_path = os.path.join(final_output_dir, f"frame_{idx:04d}.png")
        masked_pil.save(output_path)

print("Done")