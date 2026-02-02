import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import numpy as np
from models import Encoder, Decoder, DynamicsModel
from humanoid_dataloader import HumanoidSequenceDataset
import wandb

# Config
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-4
BATCH_SIZE = 1024  # 256 success + 256 failure
LATENT_DIM = 2

# Loss Weights
W_RECON_T = 100
W_DYN = 20
W_RECON_T_TAU = 100
W_LABEL = 1.0

# Data Parameters
DATA_DIR = "humanoid_data_images"
label_file = "humanoid_data_success.txt"
STACK_SIZE = 6
TAU = 1

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

class LabelsLoss(nn.Module):
    """
    A contrastive loss function that pushes latent vectors from different classes apart
    while pulling latent vectors from the same class together.
    """
    def __init__(self, margin=2.0, reduction='mean'):
        super(LabelsLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, z_pos, z_neg):
        # Handle cases where one class is not present in the batch
        if z_pos.size(0) == 0 or z_neg.size(0) == 0:
            return torch.tensor(0.0, device=z_pos.device, requires_grad=True)

        # Inter-class loss
        inter_class_dist = torch.cdist(z_pos, z_neg, p=2)
        inter_class_loss = torch.relu(self.margin - inter_class_dist).pow(2)

        # Intra-class loss
        intra_class_loss_pos = torch.pdist(z_pos, p=2).pow(2)
        intra_class_loss_neg = torch.pdist(z_neg, p=2).pow(2)

        # Combine the losses
        # We take the mean of each component.
        total_loss = inter_class_loss.mean() + intra_class_loss_pos.mean() + intra_class_loss_neg.mean()

        # Handle cases where there's only one sample of a class, resulting in NaN for pdist
        if torch.isnan(total_loss):
            return torch.tensor(0.0, device=z_pos.device, requires_grad=True)

        return total_loss

def main():
    #wandb.init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = Encoder(latent_dim=LATENT_DIM).to(device)
    decoder = Decoder(latent_dim=LATENT_DIM).to(device)
    dynamics_model = DynamicsModel(latent_dim=LATENT_DIM).to(device)

    # Load Models
    encoder_path = os.path.join(MODEL_DIR, "encoder.pt")
    decoder_path = os.path.join(MODEL_DIR, "decoder.pt")
    dynamics_model_path = os.path.join(MODEL_DIR, "dynamics_model.pt")

    if os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(dynamics_model_path):
        print("Found existing models, loading them to continue training...")
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        dynamics_model.load_state_dict(torch.load(dynamics_model_path, map_location=device))
        print("Models loaded successfully.")
    else:
        print("No existing models found, starting training from scratch.")

    dataset = HumanoidSequenceDataset(data_dir=DATA_DIR, stack_size=STACK_SIZE, tau=TAU, label_file=label_file, max_trajectories=796)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Use standard BCELoss for reconstruction loss
    reconstruction_loss_fn = nn.BCELoss(reduction='mean')
    dynamics_loss_fn = nn.MSELoss()
    label_criterion = LabelsLoss(reduction='mean')

    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics_model.parameters())
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE)

    print("Starting joint training...")
    for epoch in range(NUM_EPOCHS):
        # Balance Batch: equal number of successful and unsuccessful trajectories
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)

        selected_successful = []
        selected_unsuccessful = []

        # Iterate through the shuffled dataset to find samples for our balanced batch
        for idx in all_indices:
            if len(selected_successful) == BATCH_SIZE // 2 and len(selected_unsuccessful) == BATCH_SIZE // 2:
                break # Found enough for the batch
            _, _, label = dataset[idx]
            if label == 1 and len(selected_successful) < BATCH_SIZE // 2:
                selected_successful.append(idx)
            elif label == 0 and len(selected_unsuccessful) < BATCH_SIZE // 2:
                selected_unsuccessful.append(idx)
            #print(f"Selected {len(selected_successful)} successful and {len(selected_unsuccessful)} unsuccessful samples so far.")

        selected_indices = selected_successful + selected_unsuccessful
        np.random.shuffle(selected_indices)

        # Create subset and dataloader
        subset = Subset(dataset, selected_indices)
        dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        encoder.train()
        decoder.train()
        dynamics_model.train()

        total_recon_loss_t = 0.0
        total_recon_loss_t_plus_tau = 0.0
        total_dyn_loss = 0.0
        total_label_loss = 0.0

        # For wandb logging
        orig_stack = None
        recon_stack = None

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for i, (stack1_batch, stack2_batch, label_batch) in enumerate(loop):
            X_t = stack1_batch.to(device)
            X_t_plus_tau = stack2_batch.to(device)
            labels = label_batch.float().to(device)

            z_t = encoder(X_t)
            z_t_plus_tau_true = encoder(X_t_plus_tau)
            z_t_plus_tau_pred = dynamics_model(z_t)
            X_t_recon = decoder(z_t)
            X_t_plus_tau_recon = decoder(z_t_plus_tau_pred)

            # Save the first batch's original and reconstructed stack for logging
            # if i == 0:
            #     orig_stack = X_t[0].detach().cpu().numpy()  
            #     recon_stack = X_t_recon[0].detach().cpu().numpy() 

            # Loss Calculation
            recon_loss_t = reconstruction_loss_fn(X_t_recon, X_t)
            recon_loss_t_plus_tau = reconstruction_loss_fn(X_t_plus_tau_recon, X_t_plus_tau)
            dyn_loss = dynamics_loss_fn(z_t_plus_tau_pred, z_t_plus_tau_true)

            # Label separation loss: split z_t by label
            pos_mask = labels == 1
            neg_mask = labels == 0
            z_pos = z_t[pos_mask]
            z_neg = z_t[neg_mask]
            label_loss = label_criterion(z_pos, z_neg)

            total_loss = (W_RECON_T * recon_loss_t) + \
                         (W_DYN * dyn_loss) + \
                         (W_RECON_T_TAU * recon_loss_t_plus_tau) + \
                         (W_LABEL * label_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_recon_loss_t += recon_loss_t.item()
            total_dyn_loss += dyn_loss.item()
            total_recon_loss_t_plus_tau += recon_loss_t_plus_tau.item()
            total_label_loss += label_loss.item()

        if len(dataloader) == 0:
            print("No data in dataloader for this epoch. Skipping.")
            continue


        # Log original and reconstructed stack and total loss to wandb
        # wandb_log_dict = {"epoch": epoch + 1}
        # if orig_stack is not None and recon_stack is not None:
        #     orig_imgs = [wandb.Image(orig_stack[j], caption=f"Original t {j}") for j in range(orig_stack.shape[0])]
        #     recon_imgs = [wandb.Image(recon_stack[j], caption=f"Reconstructed t {j}") for j in range(recon_stack.shape[0])]
        #     wandb_log_dict["Original Stack"] = orig_imgs
        #     wandb_log_dict["Reconstructed Stack"] = recon_imgs
        # # Log total loss for graphing
        # wandb_log_dict["Total Weighted Loss"] = (
        #     W_RECON_T * total_recon_loss_t +
        #     W_DYN * total_dyn_loss +
        #     W_RECON_T_TAU * total_recon_loss_t_plus_tau +
        #     W_LABEL * total_label_loss
        # )
        # wandb.log(wandb_log_dict)

        print(
            f"Epoch {epoch+1} Summary:\n"
            f"  Raw Recon Loss (t):      {total_recon_loss_t:.6f}\n"
            f"  Raw Dyn Loss:            {total_dyn_loss:.6f}\n"
            f"  Raw Recon Loss (t+tau):  {total_recon_loss_t_plus_tau:.6f}\n"
            f"  Raw Label Loss:          {total_label_loss:.6f}\n"
            f"  Total Raw Loss:          {(total_recon_loss_t + total_dyn_loss + total_recon_loss_t_plus_tau + total_label_loss):.6f}\n"
            f"  Weighted Recon Loss (t):      {(W_RECON_T * total_recon_loss_t):.6f}\n"
            f"  Weighted Dyn Loss:            {(W_DYN * total_dyn_loss):.6f}\n"
            f"  Weighted Recon Loss (t+tau):  {(W_RECON_T_TAU * total_recon_loss_t_plus_tau):.6f}\n"
            f"  Weighted Label Loss:          {(W_LABEL * total_label_loss):.6f}\n"
            f"  Total Weighted Loss:          {(W_RECON_T * total_recon_loss_t + W_DYN * total_dyn_loss + W_RECON_T_TAU * total_recon_loss_t_plus_tau + W_LABEL * total_label_loss):.6f}\n"
        )

        torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "encoder.pt"))
        torch.save(decoder.state_dict(), os.path.join(MODEL_DIR, "decoder.pt"))
        torch.save(dynamics_model.state_dict(), os.path.join(MODEL_DIR, "dynamics_model.pt"))
        print(f"Models saved to {MODEL_DIR}")

if __name__ == '__main__':
    main()
