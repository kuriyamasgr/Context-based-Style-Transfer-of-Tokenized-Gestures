# -*- coding: utf-8 -*-
# @author: kuriyama

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """
    Decoder of style transformer
    """
    def __init__(self, hid_dim):
        """
        :param hid_dim: dimension of latent variables (hidden states)
        """
        super().__init__()
        self.layer_nrm = nn.LayerNorm(normalized_shape=hid_dim)
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_kv = nn.Linear(hid_dim, 2 * hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.tensor([hid_dim]))
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.GELU(),
            nn.Linear(in_features=hid_dim, out_features=hid_dim)
        )

    def forward(self, content, style):
        """
        :param content: tokens of content gestures
        :param style: tokens of style gestures
        :return: style transferred content gesture, encoded style gesture
        """
        _content = content[1:, :, :]  # omit class token
        _style = style[1:, :, :]  # omit class token
        if _content.shape[1] == 1:
            _content = _content.expand(-1, _style.shape[1], -1)
        elif _style.shape[1] == 1:
            _style = _style.expand(-1, _content.shape[1], -1)

        _z = self.layer_nrm(_content.transpose(0, 1))
        query = self.fc_q(_z)
        _zs = self.layer_nrm(_style.transpose(0, 1))
        (key, value) = self.fc_kv(_zs).chunk(2, dim=-1)
        # cross-attention
        dot_prod = torch.matmul(query, key.transpose(1, 2)) / self.scale
        _weight = torch.softmax(dot_prod, dim=-1)
        _xa = self.fc_o(torch.matmul(_weight, value))
        _zxa = self.layer_nrm(_z + _xa)
        return (_zxa + self.mlp(_zxa)).transpose(0, 1), _style


class TransformerEncoder(nn.Module):
    """
        Encoder of style transformer
    """
    def __init__(self, hid_dim):
        """
        :param hid_dim: dimension of latent variables (hidden states)
        """
        super().__init__()

        self._init_mask()
        self.pe = ConstantPE(d_model=hid_dim, max_len=150)
        self.layer_nrm = nn.LayerNorm(normalized_shape=hid_dim)
        self.fc_qkv = nn.Linear(hid_dim, 3 * hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.tensor([hid_dim], dtype=torch.float))
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.GELU(),
            nn.Linear(in_features=hid_dim, out_features=hid_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, hid_dim))

    def forward(self, z, structure=False):
        """
        :param z: latent variables fed from autoencoder
        :param structure: set True when obtaining structural similarity metric
        :return: encoded gesture tokens (structure==False), or structural similarity metric (structure==True)
        """
        _z = z.transpose(0, 1)
        _cls_tokens = self.cls_token.repeat(_z.shape[0], 1, 1)  # repeat along batch
        _z = torch.cat((_cls_tokens, _z), dim=1)

        _z = self.layer_nrm(self.pe(_z))
        # self-attention
        (query, key, value) = self.fc_qkv(_z).chunk(3, dim=-1)

        if structure:
            n_k1 = nn.functional.normalize(key, dim=2)
            return torch.matmul(n_k1, n_k1.transpose(1, 2))

        attn = torch.matmul(query, key.transpose(1, 2)) / self.scale
        _size = z.shape[0] + 1  # plus cls token !
        attn = attn.masked_fill(self.src_mask[:_size, :_size] == 1, -1e10)

        attn = torch.softmax(attn, dim=-1)
        _sa = self.fc_o(torch.matmul(attn, value))
        _zsa = self.layer_nrm(_z + _sa)
        return (_zsa + self.mlp(_zsa)).transpose(0, 1)

    def _init_mask(self):
        """
        Initialize the mask of attention mechanism
        """
        _mask_size = 2560
        if torch.cuda.is_available():
            self.src_mask = torch.ones((_mask_size, _mask_size), dtype=torch.bool, device=torch.device('cuda:0'))
        else:
            self.src_mask = torch.ones((_mask_size, _mask_size), dtype=torch.bool)

        _mask_range = 20  # int(80 / config['token_width'])
        for row in range(_mask_size):
            if row > 0:  # for gesture tokens !
                head = row - _mask_range
                if head < 0:
                    head = 0
                tail = row + _mask_range
                if tail > _mask_size:
                    tail = _mask_size
                self.src_mask[row, head:tail] = 0
            else:  # for CLS token !
                self.src_mask[0, :] = 0


class ConstantPE(nn.Module):
    """
    Constant Positional Encoding
    """
    def __init__(self, d_model, max_len, period=1):
        """
        :param d_model: dimension of each token
        :param max_len: maximum length of the positional encoding
        :param period: period of trigonometric functions
        """
        super(ConstantPE, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = 2.0 * torch.pi * torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / period
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # _z:batch(=1), num_token, token_dim (8x16=128)
        self.register_buffer('pe', pe)

    def forward(self, z, batch_first=False):
        """
        :param z: latent variables
        :param batch_first: True when the first dimension of z is batch dimension
        :return: positional-encoded latent variables
        """
        if not batch_first:
            _z = z.transpose(0, 1)  # _z:batch, num_token, token_dim (8x16=128)
        else:
            _z = z
        _out = _z + self.pe[:, :_z.size(1), :]
        if not batch_first:
            return _out.transpose(0, 1)  # _z:num_token, batch, token_dim
        else:
            return _out
