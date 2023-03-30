# -*- coding: utf-8 -*-
import torch
import glob

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger


def train(model, hparams):
    """
    Training a NN model, which is used in gsxf.py and autoencoder.py
    :param model: neural network model to be trained
    :param hparams: hyperparameters for this model
    :return:
    """
    ckpt_files = glob.glob('./pretrain/' + hparams['version'] + '/checkpoints/*.ckpt')

    if not ckpt_files:  # first training !
        _model = model(hparams=hparams)
        ckpt_file = None
    else:  # resume of training from the previous state
        ckpt_files.sort()
        ckpt_file = ckpt_files[-1]
        _ckpt = torch.load(ckpt_file)
        _model = model(_ckpt['hyper_parameters']['hparams'])

    _logger = CSVLogger(save_dir='./', name='pretrain', version=hparams['version'])
    trainer = Trainer(resume_from_checkpoint=ckpt_file, log_every_n_steps=10,
                      max_epochs=hparams['max_epochs'] + 1, logger=_logger,
                      gpus=1 if torch.cuda.is_available() else None)

    trainer.fit(_model)
