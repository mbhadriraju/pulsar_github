import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
class SimDataset(Dataset):
    def __init__(self, npy_path, num_examples, scrunch_to=16):
        self.data = np.load(npy_path, mmap_mode="r")
        self.n = num_examples
        self.scrunch_to = scrunch_to

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[2 * idx]).float().unsqueeze(0)  # [1, 256, 1024]
        y = torch.from_numpy(self.data[2 * idx + 1]).float().unsqueeze(0)

        if self.scrunch_to is not None:
            n_chan = x.shape[1]  # original number of channels
            group_size = n_chan // self.scrunch_to
            # reshape to [batch, new_chan, group_size, phase]
            x = x.view(1, self.scrunch_to, group_size, -1).mean(dim=2)
            y = y.view(1, self.scrunch_to, group_size, -1).mean(dim=2)
        return x, y
    

def loss_nn(out, x_noisy, x_clean, beta=0.05):
    recon = x_noisy - out

    l1 = F.l1_loss(recon, x_clean)
    mse = F.mse_loss(recon, x_clean)
    spatial_loss = 0.7 * l1 + 0.3 * mse

    recon_f = torch.fft.rfft(recon, dim=-1, norm='ortho')
    target_f = torch.fft.rfft(x_clean, dim=-1, norm='ortho')

    freq_loss = F.l1_loss(torch.angle(recon_f), torch.angle(target_f))

    return spatial_loss + beta * freq_loss


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="circular")
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="circular")
        self.gn2 = nn.GroupNorm(8, channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(x + out)



class Net(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__()
        self.entry = nn.Conv2d(in_channels, hidden_channels, 3, padding=1, padding_mode="circular")

        self.body = nn.Sequential(
            ResNetBlock(hidden_channels),
            ResNetBlock(hidden_channels),
            ResNetBlock(hidden_channels),
            ResNetBlock(hidden_channels),
            ResNetBlock(hidden_channels),
        )

        self.exit = nn.Conv2d(hidden_channels, in_channels, 3, padding=1, padding_mode="circular")

    def forward(self, x):
        identity = x  # ADD (for stability, even in noise prediction)
        x = self.entry(x)
        x = checkpoint(self.body, x, use_reentrant=False)
        noise = self.exit(x)
        return noise  # still predicting noise
    

import os

def train(dataset_path, num_epochs=100, batch_size=10, learning_rate=1e-4, num_examples=10000, save_path="wideband_denoise.pth"): 
    dataset = SimDataset(dataset_path, num_examples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_arr = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            recon = x - out
            loss = loss_nn(out, x, y) 
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
        loss_arr.append(avg_loss)
        torch.save(model, save_path)