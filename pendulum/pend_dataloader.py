import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

traj_success_file = "pendulum_success.txt"

# Load labels from success.txt
def load_trajectory_labels(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip():
                traj, label = line.strip().split(',')
                label_dict[traj] = int(label)
    return label_dict


# Returns the initial and final stack of each trajectory, with its label and folder name.
class TrajectoryRollout(Dataset):
    def __init__(self, data_dir, stack_size=3, transform=None, label_file=traj_success_file, max_trajectories=None):
        self.data_dir = data_dir
        self.stack_size = stack_size
        self.transform = transform
        self.samples = []
        self.label_dict = load_trajectory_labels(label_file)

        print("Initializing TrajectoryRollout dataset...")

        traj_folders = sorted(
            [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))],
            key=lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else -1
        )
        if max_trajectories is not None:
            traj_folders = traj_folders[-max_trajectories:] 

        for traj_folder in tqdm(traj_folders, desc="Processing trajectories"):
            traj_path = os.path.join(data_dir, traj_folder)
            img_files = sorted([
                os.path.join(traj_path, f)
                for f in os.listdir(traj_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if len(img_files) >= self.stack_size:
                initial_stack_files = img_files[:self.stack_size]
                final_stack_files = img_files[-self.stack_size:]
                self.samples.append((initial_stack_files, final_stack_files, traj_folder))

    def __len__(self):
        return len(self.samples)

    def _load_stack(self, file_list):
        images = []
        for img_path in file_list:
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, dtype=np.float32) / 255.0
            images.append(img_np)
        return torch.from_numpy(np.stack(images, axis=0))

    def __getitem__(self, idx):
        initial_stack_files, final_stack_files, traj_folder = self.samples[idx]
        initial_stack = self._load_stack(initial_stack_files)
        final_stack = self._load_stack(final_stack_files)

        match = re.search(r'(\d+)$', traj_folder)
        if match:
            label_key = f"{match.group(1)}.txt"
        else:
            label_key = traj_folder + ".txt"

        label = self.label_dict.get(label_key, 0)

        sample = (initial_stack, final_stack, label, traj_folder)
        if self.transform:
            sample = self.transform(sample)
        return sample

#Returns two stacks separated by tau frames, with its label.
class PendulumSequenceDataset(Dataset):
    def __init__(self, data_dir, stack_size=2, tau=1, transform=None, label_file=traj_success_file, max_trajectories=None):
        self.data_dir = data_dir
        self.stack_size = stack_size
        self.tau = tau
        self.transform = transform
        self.samples = []
        self.label_dict = load_trajectory_labels(label_file)

        print("Initializing dataset...")
        traj_folders = sorted(os.listdir(data_dir))
        if max_trajectories is not None:
            traj_folders = traj_folders[:max_trajectories]
        for traj_folder in tqdm(traj_folders, desc="Processing trajectories"):
            traj_path = os.path.join(data_dir, traj_folder)
            if not os.path.isdir(traj_path):
                continue

            img_files = sorted([
                os.path.join(traj_path, f)
                for f in os.listdir(traj_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

            if len(img_files) >= self.stack_size + self.tau:
                for i in range(len(img_files) - self.stack_size - self.tau + 1):
                    stack1_files = img_files[i : i + self.stack_size]
                    stack2_files = img_files[i + self.tau : i + self.tau + self.stack_size]
                    self.samples.append((stack1_files, stack2_files, traj_folder))

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def _load_stack(self, file_list):
        """Loads a list of image files into a single tensor stack."""
        images = []
        for img_path in file_list:
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, dtype=np.float32) / 255.0
            images.append(img_np)
        return torch.from_numpy(np.stack(images, axis=0))

    
    def __getitem__(self, idx):
        stack1_files, stack2_files, traj_folder = self.samples[idx]
        stack1 = self._load_stack(stack1_files)
        stack2 = self._load_stack(stack2_files)

        match = re.search(r'(\d+)$', traj_folder)
        if match:
            label_key = f"{match.group(1)}.txt"
        else:
            label_key = traj_folder + ".txt" 

        label = self.label_dict.get(label_key, 0)

        sample = (stack1, stack2, label)
        if self.transform:
            sample = self.transform(sample)
        return sample

# Double Check Sequence Dataset
def test_label_reading():
    data_dir = "pendulum_data_images"
    label_file = "pendulum_success.txt"
    dataset = PendulumSequenceDataset(data_dir=data_dir, stack_size=3, tau=1, label_file=label_file)
    label_counts = {0: 0, 1: 0}
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label counts: {label_counts}")
    stack1, stack2, label = dataset[0]
    print(f"Example label: {label}")
    print(f"Stack1 shape: {stack1.shape}, Stack2 shape: {stack2.shape}")

    import matplotlib.pyplot as plt
    # Display stack1
    plt.figure(figsize=(6,3))
    for i in range(stack1.shape[0]):
        plt.subplot(2, stack1.shape[0], i+1)
        plt.imshow(stack1[i], cmap='gray')
        plt.axis('off')
        plt.title(f"stack1[{i}]")
    # Display stack2
    for i in range(stack2.shape[0]):
        plt.subplot(2, stack2.shape[0], stack2.shape[0]+i+1)
        plt.imshow(stack2[i], cmap='gray')
        plt.axis('off')
        plt.title(f"stack2[{i}]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_label_reading()