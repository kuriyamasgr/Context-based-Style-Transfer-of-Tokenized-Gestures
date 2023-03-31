# -*- coding: utf-8 -*-
# @author: kuriyama

from abc import ABC
import glob
import torch
import numpy as np
import torch.nn as nn
import lit_model
from dataset.quaternion import expq, qdifangle
from dataset.mocap import MocapSet


class Encoder(nn.Module):
    """
    Encoder for converting tensor of rotational values represented by exponential map into latent variables
    """

    def __init__(self, input_channel, channels):
        """
        :param input_channel: dimension of pose vectors (number of joints x 3)
        :param channels: dimensions of latent variables for every layers
        """
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=channels[0], kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1, stride=2),
            nn.GELU()
        )

    def forward(self, z):
        """
        :param z: input tensor representing pose vectors (joints x exponential maps)
        :return: embedded latent variables
        """
        _z = z.transpose(1, 2)  # batch, frame, pose => batch, pose, frame
        _z = self.layers(_z)
        return _z.transpose(1, 2)  # batch, pose, frame => batch, frame, pose


class Decoder(nn.Module):
    """
        Decoder for converting latent variables into tensor of rotational values represented by exponential map
    """

    def __init__(self, output_channel, channels):
        """
        :param output_channel: dimension of pose vectors (number of joints x 3)
        :param channels: dimensions of latent variables for every layers
        """
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.ConvTranspose1d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.ConvTranspose1d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(in_channels=channels[2], out_channels=output_channel, kernel_size=1, padding=0)
        )

    def forward(self, z):
        """
        :param z: embedded latent variables
        :return: output tensor representing pose vectors (joints x exponential maps)
        """
        _z = z.transpose(1, 2)  # batch, frame, pose => batch, pose, frame
        _z = self.layers(_z)
        return _z.transpose(1, 2)  # batch, pose, frame => batch, frame, pose


class AutoEncoder(lit_model.LitModel, ABC):
    """
    This class inherit LitModel of pytorch-lightening package
    """

    def __init__(self, hparams):
        """
        :param hparams: disctionary of hyper parameters including design parameters of the network
        """
        super(AutoEncoder, self).__init__(hparams=hparams)

        config = hparams['config']
        _enc_chs = config['encode_chs']
        _dec_chs = config['decode_chs']
        _leg_chs = config['leg_decode_chs']
        self.rot_dim = 3  # dimension of rotational representation

        _torso_dim = 5 * self.rot_dim  # root + four spine joints
        _arms_dim = 8 * self.rot_dim  # total eight joints for both arms
        _legs_dim = 8 * self.rot_dim  # total eight joints for both legs
        self.pose_dim = _torso_dim + _arms_dim

        self.encoder = Encoder(input_channel=self.pose_dim, channels=_enc_chs)
        self.decoder = Decoder(output_channel=self.pose_dim, channels=_dec_chs)
        self.leg_decoder = Decoder(output_channel=_legs_dim, channels=_leg_chs)

    def encode(self, tensor):
        """
        Encoding tensors
        :param tensor: input tensor representing pose vectors (joints x exponential maps)
        :return: embedded latent variables
        """
        return self.encoder(tensor[:, :, :self.pose_dim])

    def decode(self, z):
        """
        :param z: embedded latent variables
        :return: output tensor representing pose vectors (joints x exponential maps)
        """
        _upper_pose = self.decoder(z)
        _lower_pose = self.leg_decoder(z)
        return torch.cat((_upper_pose, _lower_pose), dim=2)

    def forward(self, data):
        """
        :param data: batch of tensors representing a sequence of gestural poses
        :return: auto-encoded tensors representing a sequence of gestural poses
        """
        return self.decode(self.encode(data))

    def loss_function(self, output, data):
        """
        :param output: output tensor of this autoencoder
        :param data: original tensors representing a sequence of gestural poses
        :return: loss value
        """
        _frames = data.shape[1]
        _b, _f, _dim = output.shape
        _out_rot = output[:, :_frames, :].view(_b, _frames, _dim // self.rot_dim, self.rot_dim)
        _gt_rot = data.view(_b, _frames, _dim // self.rot_dim, self.rot_dim)
        return torch.mean(torch.norm(_out_rot - _gt_rot, dim=3))

    def evaluation(self, output, reference):
        """
        evaluate restoration errors using the mean and maximum of angular differences of every joints
        :param output:
        :param reference:
        :return:
        """
        _frames = reference.shape[1]
        _b, _f, _dim = output.shape
        _out_rot = output[:, :_frames, :].view(_b, _frames, _dim // self.rot_dim, self.rot_dim)
        _gt_rot = reference.view(_b, _frames, _dim // self.rot_dim, self.rot_dim)
        _dif = qdifangle(expq(_out_rot, torch=True), expq(_gt_rot, torch=True))
        _difnp = _dif.detach().numpy()
        return np.mean(_difnp), np.max(_difnp)

    def test_accuracy(self, data_path, fps):
        """
        Test the restoration accuracy of this autoencoder
        :param data_path: path to the testing dataset
        :param fps: frame per second of the dataset
        """
        _mocap = MocapSet(data_path, num_frames=-1, multiplex=False, fps=fps)
        _means = []
        _maxs = []
        for _clip in _mocap.clips:
            _input = torch.unsqueeze(torch.tensor(_clip.tensor), dim=0)
            _output = self(_input)
            _mean, _max = self.evaluation(_output, _input)
            _means.append(_mean)
            _maxs.append(_max)

        _means = np.array(_means)
        _maxs = np.array(_maxs)
        print(np.mean(_means), np.max(_maxs), np.mean(_maxs))


if __name__ == '__main__':
    """
    Training/Testing the autoencoder
    """
    from trainer import train

    hyper_parameters = {
        'config': {
            'encode_chs': [64, 32, 16],
            'decode_chs': [16, 32, 64],
            'leg_decode_chs': [16, 16, 32]
        },
        'version': 'AE',
        'batch_size': 32, 'learning_rate': 0.0001, 'num_frames': 256,
        'fps': 60, 'max_epochs': 3000,
        'data_path': glob.glob('./dataset/bvh/training/*/*.bvh') + glob.glob('./dataset/bvh/training_add/*.bvh')
    }

    is_train = True

    if is_train:
        train(AutoEncoder, hyper_parameters)
    else:
        files = glob.glob('./pretrain/' + hyper_parameters['version'] + '/checkpoints/*.ckpt')
        files.sort()
        _file = files[0]  # latest version
        _autoenc = AutoEncoder.load_from_checkpoint(checkpoint_path=_file)
        _autoenc.test_accuracy(data_path=hyper_parameters['data_path'], fps=hyper_parameters['fps'])
