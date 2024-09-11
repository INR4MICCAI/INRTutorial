from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as pl

from medmnist import RetinaMNIST, BreastMNIST
from data.datasets import RandomPointsDataset, MedMNISTDataset
from networks.decoders import MLP
from networks.layers import SineLayer, ReluLayer, WIRELayer
from networks.heads import ReconstructionHead
from networks.pos_encoders import FourierFeatPosEncoder, IdentityPosEncoder

IMAGE_SIZE = 224
train_dataset = MedMNISTDataset(BreastMNIST(split="val", download=True, size=IMAGE_SIZE, ), num_points=2048)


class INR(pl.LightningModule):
    def __init__(self, coord_size, out_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.pos_enc = IdentityPosEncoder(coord_size)
        # self.pos_enc = FourierFeatPosEncoder(coord_size, freq_num=hidden_size//2)
        self.network = MLP(self.pos_enc.out_size, hidden_size=hidden_size, num_layers=num_layers, layer_class=ReluLayer)
        self.head = ReconstructionHead(self.network.out_size, out_size)
        self.progress_ims = []

    def configure_optimizers(self):
        return torch.optim.Adam((*self.network.parameters(), *self.head.parameters()), lr=0.001)

    def forward(self, coords):
        coords_enc = self.pos_enc(coords)
        out = self.network(coords_enc)
        out = self.head(out)
        return out

    def training_step(self, batch, batch_idx):
        coords, values, _ = batch
        coords = coords.view((-1, coords.shape[-1]))
        outputs = self.forward(coords)
        loss = nn.functional.mse_loss(outputs, values)
        if any([self.current_epoch == i for i in [0, 10, 50, 200, 1000, 5000, 10000]]):
            pred_im = self.sample_at_resolution([IMAGE_SIZE] * coords.shape[-1])
            self.progress_ims.append(pred_im.cpu().numpy())
        return loss

    @torch.no_grad()
    def sample_at_resolution(self, resolution: Tuple[int, ...]):
        meshgrid = torch.meshgrid([torch.arange(0, i, device=self.device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=self.device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, len(meshgrid))
        predictions_ = self.forward(coords_norm_)
        predictions = predictions_.reshape((*resolution, -1))
        return predictions

inr = INR(train_dataset.coord_size, train_dataset.value_size)
trainer = pl.Trainer(max_epochs=10001)
trainer.fit(inr, train_dataloaders=DataLoader(train_dataset, batch_size=1))


orig = train_dataset.load_image(0)[0].numpy()
vis = np.concatenate((orig, *inr.progress_ims), 1)
import matplotlib.pyplot as plt
plt.imshow(vis)
plt.show()

