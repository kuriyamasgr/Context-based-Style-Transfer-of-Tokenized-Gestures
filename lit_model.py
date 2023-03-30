# -*- coding: utf-8 -*-

from abc import ABC
import torch
import pytorch_lightning as pl
from dataset.mocap import MocapSet
from torch.utils.data import DataLoader, WeightedRandomSampler


class LitModel(pl.LightningModule, ABC):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = hparams['batch_size']
        self.learning_rate = hparams['learning_rate']
        self.data_path = hparams['data_path']
        self.num_frames = hparams.get('num_frames', -1)
        self.multiplex = hparams.get('multiplex', False)
        self.fps = hparams.get('fps', 30)
        self.num_epochs = 0

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_function(y_hat, batch)
        self.log('train_loss', loss)
        return loss

    # @pl.data_loader
    def train_dataloader(self):
        _mocap_set = MocapSet(data_path=self.data_path, num_frames=self.num_frames,
                              multiplex=self.multiplex, fps=self.fps, verbose=True)
        weights, n_samples = _mocap_set.sampling_weight()
        weighted_sampler = WeightedRandomSampler(weights, num_samples=n_samples)
        if torch.cuda.is_available():
            n_worker = 10
        else:
            n_worker = 0

        return DataLoader(_mocap_set, batch_size=self.batch_size, num_workers=n_worker,
                          sampler=weighted_sampler, drop_last=True)

    def configure_optimizers(self):
        # Generate the optimizers.
        print('Learning rate =', self.learning_rate)
        return torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
