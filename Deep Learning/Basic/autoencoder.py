import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
import torchvision 
from torchvision.datasets.mnist import MNIST 
from torchvision import transforms 

import pytorch_lightning as pl 

class AutoEncoder(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )
    self.decoder = nn.Sequential(
        nn.Linear(3, 128),
        nn.ReLU(),
        nn.Linear(128, 28 * 28)
    )

  def forward(self, x):
    out = self.encoder(x)
    return out

  def training_step(self, batch, batch_idx):
    x, _ = batch 
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    y = self.decoder(z)
    loss = F.mse_loss(y, x)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, _ = batch 
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    y = self.decoder(z)
    loss = F.mse_loss(y, x)
    self.log("val_loss", loss)
    return {"val_loss" : loss}

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    print(avg_loss)
    self.log("val_epoch_loss", avg_loss)
    return avg_loss
  
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ = "__main__":

    train_val = MNIST("", train=True, download=True, transform=transforms.ToTensor()) 
    test = MNIST("", train=False, download=True, transform=transforms.ToTensor()) 
    train, val = random_split(train_val, [55000, 5000])

    model = AutoEncoder() 

    trainer = pl.Trainer(gpus=1, max_epochs=5)
    trainer.fit(model, train_dataloader, val_dataloader)

    '''
    colab 上で tensorboad log の確認
    '''
    %load_ext tensorboard
    %tensorboard --logdir ./lightning_logs/version_0/
