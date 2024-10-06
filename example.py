from typing import Tuple, List
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
import lightning as pl
from medmnist import BreastMNIST, PneumoniaMNIST

from dataset import RandomPointsDataset
from networks.inrs import WIRENMLP, SIRENMLP, ReLUMLP
from networks.pos_encoders import FourierFeatPosEncoder
from utils import psnr, plot_reconstructions


# ----------  Download image and create dataset  -------------------------
IMAGE_SIZE = 224  # ChestMNIST offers sizes: 28, 64, 128, 224
chest_dataset = PneumoniaMNIST(split="val", download=True, size=IMAGE_SIZE)

# INRs are trained on only 1 scene. We only want 1 image.
pil_image, _ = chest_dataset[1]

gt_image = pil_to_tensor(pil_image)
gt_image = gt_image.moveaxis(0, -1)  # Convert to torch.Tensor
gt_image = gt_image.to(torch.float32) / 255.0  # Normalize image between [0.0, 1.0]
print("Image shape:", gt_image.shape)
print("Max:", gt_image.max(), "Min:", gt_image.min())


POINTS_PER_SAMPLE = 2048
dataset = RandomPointsDataset(gt_image, points_num=POINTS_PER_SAMPLE)
# We set a batch_size of 1 since our dataloader is already returning a batch of points.
dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)


# ------------  Create network (Uncomment to use) -------------------------------
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# ReLU
# pos_encoder = None
# inr = ReLUMLP(dataset.coord_size, dataset.value_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

# Positional encoder + ReLU
# pos_encoder = FourierFeatPosEncoder(dataset.coord_size, HIDDEN_SIZE//2)
# inr = ReLUMLP(pos_encoder.out_size, dataset.value_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

# SIREN
# SIREN_OMEGA = 30.
# pos_encoder = None
# inr = SIRENMLP(dataset.coord_size, dataset.value_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
#                siren_factor=SIREN_OMEGA)

# WIRE
WIRE_OMEGA = 10.
WIRE_SIGMA = 20.
pos_encoder = None
inr = WIRENMLP(dataset.coord_size, dataset.value_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
               wire_omega=WIRE_OMEGA, wire_sigma=WIRE_SIGMA)


# ------------  Pytorch Lightning module for training  -------------------------------
class INRLightningModule(pl.LightningModule):
    def __init__(self,
                 network: nn.Module,
                 gt_im: torch.Tensor,
                 pos_encoder: nn.Module = None,
                 lr: float = 0.001,
                 name: str = "",
                 eval_interval: int = 100,
                 visualization_intervals: List[int] = [0, 100, 500, 1000, 5000, 10000, 20000],
                 ):
        super().__init__()
        self.lr = lr
        self.network = network
        self.pos_encoder = pos_encoder

        # Logging
        self.name = name
        self.gt_im = gt_im
        self.eval_interval = eval_interval
        self.visualization_intervals = visualization_intervals
        self.progress_ims = []
        self.scores = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, coords):
        if self.pos_encoder is not None:
            coords = self.pos_encoder(coords)
        out = self.network(coords)
        return out

    def training_step(self, batch, batch_idx):
        coords, values = batch
        coords = coords.view(-1, coords.shape[-1])
        values = values.view(-1, values.shape[-1])
        outputs = self.forward(coords)
        loss = nn.functional.mse_loss(outputs, values)
        return loss

    def on_train_epoch_end(self):
        """ At each visualization interval, reconstruct the image using our INR """
        if (self.current_epoch + 1) % self.eval_interval == 0 or self.current_epoch == 0:
            pred_im = self.sample_at_resolution(self.gt_im.shape[:-1])
            pred_im = pred_im.reshape(self.gt_im.shape)
            psnr_value = psnr(pred_im, self.gt_im.to(pred_im.device)).cpu().item()
            self.scores.append((self.current_epoch + 1, psnr_value))  # Log PSNR
            if self.current_epoch + 1 in self.visualization_intervals:
                self.progress_ims.append((self.current_epoch + 1, pred_im.cpu(), psnr_value))

    @torch.no_grad()
    def sample_at_resolution(self, resolution: Tuple[int, ...]):
        """ Evaluate our INR on a grid of coordinates in order to obtain an image. """
        meshgrid = torch.meshgrid([torch.arange(0, i, device=self.device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=self.device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
        predictions_ = self.forward(coords_norm_)
        predictions = predictions_.reshape(resolution)
        return predictions

pl_module = INRLightningModule(inr, gt_image, pos_encoder=pos_encoder)


# ------------  Training  -------------------------------
TRAINING_EPOCHS = 10_000
trainer = pl.Trainer(max_epochs=TRAINING_EPOCHS)
s = datetime.now()
trainer.fit(pl_module, train_dataloaders=DataLoader(dataset, batch_size=1))
print(f"Fitting time: {datetime.now()-s}s.")


# ------------  Result visualization  -------------------------------
vis = plot_reconstructions(pl_module.progress_ims, gt_image)
