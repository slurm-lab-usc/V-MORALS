import os
os.environ["MUJOCO_GL"] = "egl"

import cv2
import numpy as np
from dm_control import suite
from PIL import Image

#Renders Trajectories for Humanoid from humanoid_data txt files

def load_qpos_qvel_trajectory(filename):
    data = np.loadtxt(filename, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return data

def center_crop(img, crop_size=(384, 384)):
    h, w = img.shape[:2]
    ch, cw = crop_size
    start_h = max((h - ch) // 2, 0)
    start_w = max((w - cw) // 2, 0)
    return img[start_h:start_h+ch, start_w:start_w+cw]

def filter_orange_pixels(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_orange = np.array([10, 100, 100], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower_orange, upper_orange)
    filtered = cv2.bitwise_and(img, img, mask=mask)
    return filtered

def render_trajectory(traj_file, out_dir):
    env = suite.load(domain_name="humanoid", task_name="run")
    traj = load_qpos_qvel_trajectory(traj_file)
    os.makedirs(out_dir, exist_ok=True)

    qpos_dim = env.physics.data.qpos.shape[0]
    qvel_dim = env.physics.data.qvel.shape[0]

    for i, state in enumerate(traj):
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:qpos_dim+qvel_dim]
        env.physics.data.qpos[:] = qpos
        env.physics.data.qvel[:] = qvel
        env.physics.forward()
        img = env.physics.render(camera_id=0, height=480, width=640)

        # Center crop
        img_cropped = center_crop(img, crop_size=(384, 384))

        # Filter for orange pixels
        img_orange = filter_orange_pixels(img_cropped)

        # Convert to grayscale (single channel)
        img_gray = cv2.cvtColor(img_orange, cv2.COLOR_RGB2GRAY)

        # Resize to 128x128
        img_resized = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_AREA)

        # Save as single-channel PNG
        img_path = os.path.join(out_dir, f"frame_{i:04d}.png")
        Image.fromarray(img_resized).convert('L').save(img_path)

def process_all_trajectories(traj_dir="humanoid_data", out_dir="humanoid_data_images"):
    os.makedirs(out_dir, exist_ok=True)
    traj_files = sorted(
        [f for f in os.listdir(traj_dir) if f.startswith("state_trajectory_") and f.endswith(".txt")],
        key=lambda x: os.path.getmtime(os.path.join(traj_dir, x))
    )
    for idx, traj_file in enumerate(traj_files):
        traj_path = os.path.join(traj_dir, traj_file)
        final_subdir = os.path.join(out_dir, f"humanoid_trajectory_{idx}")
        if os.path.exists(final_subdir):
            for old_file in os.listdir(final_subdir):
                os.remove(os.path.join(final_subdir, old_file))
        else:
            os.makedirs(final_subdir, exist_ok=True)
        render_trajectory(traj_path, final_subdir)
        print(f"Saved frames for {traj_file} in {final_subdir}")

process_all_trajectories("humanoid_data", "humanoid_data_images")