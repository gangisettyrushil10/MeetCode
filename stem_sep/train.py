import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import StemDataset
from model import UNet

dataset = StemDataset("stem_sep/data/train")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet(in_channels=1, out_channels=1)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()

for epoch in range(50):
    for mix, vocal in loader:
        mix = mix.unsqueeze(1)
        vocal = vocal.unsqueeze(1)
        mask = model(mix)
        pred = mask * mix

        loss = criterion(pred, vocal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}  Loss {loss.item()}")

torch.save(model.state_dict(), "stem_sep/model.pth")