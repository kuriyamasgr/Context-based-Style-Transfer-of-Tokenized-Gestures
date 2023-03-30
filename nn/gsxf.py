# -*- coding: utf-8 -*-
# @author: kuriyama

import glob
from abc import ABC
import torch
import torch.nn.functional as F
import numpy as np

from lit_model import LitModel
from nn.autoencoder import AutoEncoder
from dataset.mocap import MocapSet
from torch.utils.data import DataLoader, WeightedRandomSampler
from nn.transformer import TransformerEncoder, TransformerDecoder


class GestureStyleTransformer(LitModel, ABC):
    """
    main class for transferring the style of gestures
    """
    def __init__(self, hparams):
        """
        :param hparams: dictionary of hyper-parameters for configuring style transformer
        """
        self.args = hparams
        config = hparams['config']
        self.token_width = config['token_width']
        self.dim_z = config['dim_z']
        self.loss_type = config.get('loss_type', 'all')
        self.num_frames = config.get('num_frames', 1280)
        self.fps = config.get('fps', 60)

        super(GestureStyleTransformer, self).__init__(hparams=hparams)

        _dim_token = self.dim_z * self.token_width
        self.tr_encoder = TransformerEncoder(hid_dim=_dim_token)
        self.tr_decoder = TransformerDecoder(hid_dim=_dim_token)

        self.c_token = None
        self.s_token = None
        self.s_token_list = None

        self.ae = self.set_autoencoder(config)

    def setup_style_tokens(self, style_set):
        """
        Set up the list of style tokens for transfer
        :param style_set: set of style gesture clips
        """
        style_set.setup_mirror_motions()  # data augumentation by taking minnor motions
        self.s_token_list = []
        for _style_clip in style_set.clips:  # for each style samples
            _style_z = self.ae.encode(torch.unsqueeze(torch.tensor(_style_clip.tensor), dim=0))
            _s_token = self.tensor2token(_style_z)
            self.s_token_list.append(_s_token)

        # print(len(self.s_token_list), 'style files are used...')

    @staticmethod
    def set_autoencoder(config):
        """
        Set up autoencoder for converting a gesture motion representation into corresnponding latent variables
        :param config: dictionary for configuring the path to the checkpoint file of pre-trained autoencoder
        :return: class object of autoencoder
        """
        ckpt_paths = glob.glob(config['ae_path'] + '/checkpoints/*.ckpt')
        ckpt_paths = np.sort(ckpt_paths)
        # print('AutoEncoder of ', ckpt_paths[-1], 'is loaded...')
        if torch.cuda.is_available():
            _mapl = torch.device('cuda')
        else:
            _mapl = torch.device('cpu')
        _ae = AutoEncoder.load_from_checkpoint(checkpoint_path=ckpt_paths[-1], map_location=_mapl)
        for param in _ae.parameters():
            param.requires_grad_(False)
        return _ae

    def training_step(self, batch, batch_idx):
        """
        derived from pytorch-lightening (LitModel)
        :param batch: batch of training samples
        :param batch_idx: not used for this code
        :return: loss value of this transformer
        """
        y_hat = self(batch)
        loss = self.loss_function(y_hat, batch)
        return loss

    def train_dataloader(self):
        """
        derived from pytorch-lightening (LitModel)
        :return: dataloader for training
        """
        _content_path = glob.glob(self.data_path + '/content/*.bvh')
        _style_path = glob.glob(self.data_path + '/style/*.bvh')

        _content_set = MocapSet(data_path=_content_path, num_frames=self.num_frames, fps=self.fps)
        _style_set = MocapSet(data_path=_style_path, num_frames=self.num_frames, fps=self.fps)

        weights, n_samples = _content_set.sampling_weight()
        weighted_sampler = WeightedRandomSampler(weights, num_samples=n_samples)

        self.setup_style_tokens(_style_set)

        if torch.cuda.is_available():
            n_worker = 2
        else:
            n_worker = 0
        return DataLoader(_content_set, batch_size=self.batch_size, num_workers=n_worker,
                          sampler=weighted_sampler, drop_last=True)

    def tensor2token(self, tensor):
        """
        Convert tokenize gesture motion into tokens
        :param tensor: the tensor representation of gesture motion
        :return: batch of gesture tokens
        """
        _num_token = int(tensor.shape[1] // self.token_width)
        _step = self.token_width // 2
        _n_batch = tensor.shape[0]
        _n_split = int(self.token_width // _step)
        _total = (_num_token - 1) * _n_split + 1
        _token_dim = self.token_width * self.dim_z
        _tens_pool = torch.zeros(_n_batch, _total, _token_dim)
        for offset, s in enumerate(range(0, self.token_width, _step)):
            if s == 0:
                _tens = tensor[:, :_num_token * self.token_width, :]
                _tmp = torch.reshape(_tens, (_n_batch, _num_token, _token_dim))
                _tens_pool[:, 0::_n_split] = torch.reshape(_tens, (_n_batch, _num_token, _token_dim))
            else:
                _tens = tensor[:, s:(_num_token - 1) * self.token_width + s, :]
                _tens_pool[:, offset::_n_split] = torch.reshape(_tens, (_n_batch, _num_token - 1, _token_dim))

        return _tens_pool.transpose(0, 1)  # n_token, batch, token_dim

    def token2tensor(self, token):
        """
        Convert tokenize gesture tokens into tensor representation
        :param token: batch of gesture tokens (n_token, batch, token_dim)
        :return:
        """
        _batch = token.shape[1]
        _tmp = token.transpose(0, 1)
        return torch.reshape(_tmp, (_batch, -1, self.dim_z))

    def xf_from_encoded_tokens(self, content, style, style_token, multi_stream=False):
        """
        Style transfer from encoded gesture tokens
        :param content: encoded tokens of content gesture
        :param style: encoded tokens of style gesture
        :param style_token: (non-encoded) tokens of style gesture
        :param multi_stream: set True when obtaining multi-stream swapping
        :return: style-transferred tokens
        """
        _c_decoded, _s_encoded = self.tr_decoder(content=content, style=style)
        return swapper(decoded_content=_c_decoded, encoded_style=_s_encoded,
                       style_token=style_token, multi_stream=multi_stream)

    def forward(self, data):
        """
        :param data: input tensors (gesture motion)
        :return: style-transferred tokens, encoded content tokens, and encoded style tokens
        """
        self.c_token = self.tensor2token(self.ae.encode(data))
        _rnd_index = np.random.randint(0, len(self.s_token_list))
        self.s_token = self.s_token_list[_rnd_index]  # training is done for each style sample randomly extracted !

        _c_enc = self.tr_encoder(self.c_token)
        _s_enc = self.tr_encoder(self.s_token)
        _xf_token = self.xf_from_encoded_tokens(content=_c_enc, style=_s_enc, style_token=self.s_token)

        return _xf_token, _c_enc, _s_enc

    def identity_loss(self, c_enc, s_enc):
        """
        Evaluate identity loss
        :param c_enc: encoded content tokens
        :param s_enc: encoded style tokens
        :return: sum of identity losses of content and style tokens (gestures)
        """
        c_identity = self.xf_from_encoded_tokens(content=c_enc, style=c_enc, style_token=self.c_token, multi_stream=True)
        s_identity = self.xf_from_encoded_tokens(content=s_enc, style=s_enc, style_token=self.s_token, multi_stream=True)

        c_ident_loss = torch.mean(torch.linalg.norm(c_identity - self.c_token, dim=2))
        s_ident_loss = torch.mean(torch.linalg.norm(s_identity - self.s_token, dim=2))
        return c_ident_loss + s_ident_loss

    def cyclic_loss(self, c_enc, xf_token):
        """
        Evaluate cyclic loss
        :param c_enc: encoded content tokens
        :param xf_token: style transferred tokens
        :return: cyclic loss
        """
        _c2s_enc = self.tr_encoder(xf_token)
        _cyclic_token = self.xf_from_encoded_tokens(content=_c2s_enc, style=c_enc, style_token=self.c_token, multi_stream=True)
        return torch.mean(torch.linalg.norm(_cyclic_token - self.c_token, dim=2))

    def appearance_loss(self, xf_token, s_enc):
        """
        Evaluate appearance loss
        :param xf_token: style transferred tokens
        :param s_enc: encoded style tokens
        :return: appearance loss
        """
        _xf_enc = self.tr_encoder(xf_token)
        _s_tr_token = s_enc[0, :, :].expand(_xf_enc.shape[1], -1)  # n_batch, token_dim
        return torch.mean(torch.linalg.norm(_s_tr_token - _xf_enc[0, :, :], dim=1))  # between token tokens

    def structure_loss(self, xf_token):
        """
        Evaluate structure loss
        :param xf_token: style transferred tokens
        :return: structure loss
        """
        _xf_struc = self.tr_encoder(xf_token, structure=True)
        _c_struc = self.tr_encoder(self.c_token, structure=True)
        return torch.mean(torch.linalg.norm(_c_struc[:, 1:, 1:] - _xf_struc[:, 1:, 1:], dim=(-2, -1)))

    def loss_function(self, output, data):  # data is dummy for API compatibility
        """
        Summing up all kinds of loss functions
        :param output: the output of this transformer (style-transferred gesture)
        :param data: not used in this code (leave for compatibility)
        :return: loss values for training this transformer
        """
        _xf_token = output[0]  # style-transferred content
        _c_enc = output[1]  # encoded content (overlapped)
        _s_enc = output[2]  # encoded style (overlapped)
        loss = 0.0
        if self.loss_type != 'woId' and self.loss_type != 'woReg':
            _identity_loss = self.identity_loss(_c_enc, _s_enc)
            self.log('identity_loss', _identity_loss)
            loss += _identity_loss
        if self.loss_type != 'woCy' and self.loss_type != 'woReg':
            _cyclic_loss = self.cyclic_loss(_c_enc, _xf_token)
            self.log('cyclic_loss', _cyclic_loss)
            loss += _cyclic_loss
        if self.loss_type != 'woAp' and self.loss_type != 'woViT':
            _appear_loss = self.appearance_loss(_xf_token, _s_enc)
            self.log('appearance_loss', _appear_loss)
            loss += _appear_loss
        if self.loss_type != 'woSt' and self.loss_type != 'woViT':
            _struct_loss = self.structure_loss(_xf_token)
            self.log('structure_loss', _struct_loss)
            loss += _struct_loss

        self.log('train_loss', loss)
        return loss

    def transfer(self, data, hard_swap=False):
        """
        Evaluate style transfer using our method
        :param data: dictionaly of input tensors of both content and style gestures
        :param hard_swap: set True when testing hard swap model
        :return: tensors representing style transferred gesture
        """
        _c_z = self.ae.encode(data['content'])
        _s_z = self.ae.encode(data['style'])
        _c_token = self.tensor2token(_c_z)
        _s_token = self.tensor2token(_s_z)

        _c_enc = self.tr_encoder(_c_token)  # ovrelapped
        _s_enc = self.tr_encoder(_s_token)  # inclusive mirror + ovrelap tokens
        _c_decoded, _s_encoded = self.tr_decoder(content=_c_enc, style=_s_enc)
        _c2s_token = swapper(decoded_content=_c_decoded, encoded_style=_s_encoded,
                             style_token=_s_token, hard_swap=hard_swap)
        _c2s_z = linear_blending(_c2s_token, self.token_width, self.dim_z)
        return self.ae.decode(_c2s_z)

    def adain(self, data):
        """
        Evaluate style transfer using adain mechanism
        :param data: dictionaly of input tensors of both content and style gestures
        :return: tensors representing style transferred gesture
        """
        _c_z = self.ae.encode(data['content'])
        _s_z = self.ae.encode(data['style'])
        # print('AE =', _c_z[0, 300, :], _s_z[0, 200, :])
        _c_mean = torch.mean(_c_z, dim=1, keepdim=True)
        _s_mean = torch.mean(_s_z, dim=1, keepdim=True)
        _c_std = torch.std(_c_z, dim=1, keepdim=True)
        _s_std = torch.std(_s_z, dim=1, keepdim=True)
        _c2s_z = _s_std * (_c_z - _c_mean) / _c_std + _s_mean
        return self.ae.decode(_c2s_z)

    def avatar(self, data):
        """
        Evaluate style transfer using the similar mechanism of Avatar net
        :param data: dictionaly of input tensors of both content and style gestures
        :return: tensors representing style transferred gesture
        """
        _c_z = self.ae.encode(data['content'])
        _s_z = self.ae.encode(data['style'])
        _c_mean = torch.mean(_c_z, dim=1, keepdim=True)
        _s_mean = torch.mean(_s_z, dim=1, keepdim=True)
        _c_std = torch.std(_c_z, dim=1, keepdim=True)
        _s_std = torch.std(_s_z, dim=1, keepdim=True)
        _c_white = (_c_z - _c_mean) / _c_std
        _s_white = (_s_z - _s_mean) / _s_std
        _c_token = self.tensor2token(_c_white)
        _s_token = self.tensor2token(_s_white)
        _c2s_token = simple_swapper(_c_token, _s_token)
        _c2s_white = linear_blending(_c2s_token, self.token_width, self.dim_z)
        _c2s_z = _s_std * _c2s_white + _s_mean
        return self.ae.decode(_c2s_z)

def swapper(decoded_content, encoded_style, style_token, multi_stream=False, hard_swap=False):
    """
    Swapping between decoded content tokens and encoded style tokens based on the (hard/soft) maximum of cosine similarity
    :param decoded_content: tokens of decoded content gesture
    :param encoded_style: tokens of encoded style gesture
    :param style_token: token of style gestures (not encoded)
    :param multi_stream: set True when obtaing multi-stream swapping
    :param hard_swap: set True when obtaining hard maximum of cosine similarity in swapping
    :return: style-transferred tokens through token-swapping mechanism
    """
    if multi_stream:
        _matrix = torch.einsum('ibk,jbk->bij', decoded_content, encoded_style)
    else:
        _matrix = torch.einsum('ibk,jsk->bij', decoded_content, encoded_style)

    if hard_swap:
        _indices = torch.argmax(_matrix, dim=2)  # batch, num_c_token, num_s_token
        _weights = torch.zeros(_matrix.shape)
        for b in range(_weights.shape[0]):
            for c in range(_weights.shape[1]):
                _weights[b, c, _indices[b, c]] = 1.0
    else:
        _weights = torch.softmax(_matrix, dim=2)  # batch x num_c_token x num_s_token

    if multi_stream:
        _xf_tokens = torch.einsum('bij,jbk->bik', _weights, style_token)
    else:
        _xf_tokens = torch.einsum('bij,jsk->bik', _weights, style_token)

    return _xf_tokens.transpose(0, 1)

def simple_swapper(c_token, s_token):
    """
    Swapping between content and style tokens based on the maximum of cosine similarity,
    here the tokens are not encoded/decoded by style transformer
    :param c_token: tokens of content gesture
    :param s_token: tokens of style gesture
    :return:
    """
    _c_nrm_token = F.normalize(c_token, dim=2)
    _s_nrm_token = F.normalize(s_token, dim=2)
    _matrix = torch.tensordot(_c_nrm_token, _s_nrm_token, dims=[[1, 2], [1, 2]])
    _indices = torch.argmax(_matrix, dim=1)
    _xf_tokens = torch.zeros(_c_nrm_token.shape[0], c_token.shape[2])
    for n, _ind in enumerate(_indices):
        _xf_tokens[n, :] = s_token[_ind, 0, :]
    return torch.unsqueeze(_xf_tokens, dim=1)

def linear_blending(tokens, width, dim_z):
    """
    Merge two sequences of tokens whose periods are halfway overlapped, using a linear interpolation
    :param tokens: gesture tokens where every other are halfway overlapped along frames (time)
    :param width: the number frames included in each token
    :param dim_z: dimension of each token per each frame
    :return:
    """
    _token_size = width * dim_z  # full channel size per each token
    _half_w = int(_token_size // 2)
    _wgt = torch.arange(0., 1.0001, 1. / (_half_w - 1))
    _blended = []
    _prev = None
    for _token in tokens:
        if _prev is None:
            _blended.append(_token[:, :_half_w])
        else:
            _blended.append((1. - _wgt) * _prev + _wgt * _token[:, :_half_w])
        _prev = _token[:, _half_w:]
    _blended.append(_prev)

    _n_tokens = tokens.shape[0] // 2 + 1
    _n_batch = tokens.shape[1]
    _tmp = torch.stack(_blended, dim = 0).transpose(0, 1)
    _tmp = torch.reshape(_tmp, (_n_batch, -1))
    return torch.reshape(_tmp, (_n_batch, -1, dim_z))


if __name__ == '__main__':
    """
    Used in training this transformer
    """
    import os
    import time
    from trainer import train

    hyper_parameters = {
        'config': {
            'dim_z': 16,
            'token_width': 4,
            'ae_path': './pretrain/AE',
            'class_token': True,
            'loss_type': 'all'
        },
        'data_path': './dataset/bvh/training',
        'batch_size': 8, 'learning_rate': 0.0001, 'num_frames': 1280,
        'fps': 60, 'max_epochs': 1000, 'version': 'GSX4ep1K'
    }

    os.chdir('../')
    start = time.time()
    train(GestureStyleTransformer, hyper_parameters)
    print('Training time =', time.time() - start)
