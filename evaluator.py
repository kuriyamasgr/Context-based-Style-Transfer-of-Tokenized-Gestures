# -*- coding: utf-8 -*-
# @author: kuriyama

import copy
import glob
import numpy as np

from dataset.mocap import MocapSet
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from nn.gsxf import GestureStyleTransformer


def evaluation(model, version, type='intra'):
    """
    evaluation of style transfer using our own similarity metrics
    :param model: model of style transfer
    :param version: version name of pre-training
    :param type: condition name of experiment
    """
    _content_set = MocapSet(data_path=glob.glob('./dataset/bvh/evaluation/content/*.bvh'), fps=60)
    _style_set = MocapSet(data_path=glob.glob('./dataset/bvh/evaluation/style/*.bvh'), fps=60)
    _style_set.setup_mirror_motions()

    for n, _ver in enumerate(version):
        # print('Version =', _ver)
        _content_er = np.array([0.0, 0.0, 0.0])
        _style_er = np.array([0.0, 0.0, 0.0])
        _data_counter = 0
        lit_model = _setup_model(model, './pretrain/' + _ver)
        for _content in _content_set.clips:
            _c_label = _content.label.split('_')[0]
            for _style in _style_set.clips:
                _s_label = _style.label.split('_')[0]
                if _c_label == _s_label and type == 'inter':
                    continue
                elif _c_label != _s_label and type == 'intra':
                    continue

                _data = {'content': _content.one_batch_tensor(), 'style': _style.one_batch_tensor()}

                _output = lit_model.transfer(_data)
                # _output = lit_model.transfer(_data, hard_swap=True)
                # _output = lit_model.avatar(_data)
                # _output = lit_model.adain(_data)

                _content_update = copy.deepcopy(_content)
                _content_update.restore(_output[0, :, :])

                _content.set_position(flatten=False)
                _style.set_position(flatten=False)
                _content_update.set_position(flatten=False)

                _cnt, _sty = evaluate_distance(content=_content.position,
                                               style=_style.position,
                                               transfer=_content_update.position)
                _content_er += _cnt
                _style_er += _sty
                _data_counter += 1

        _content_er /= _data_counter
        _style_er /= _data_counter
        print(type, '=> content: mean {:.3g}'.format(_content_er[0]), 'max {:.3g}'.format(_content_er[1]), 'worst25 {:.3g},'.format(_content_er[2]),
              'style: mean {:.3g}'.format(_style_er[0]), 'max {:.3g}'.format(_style_er[1]), 'worst25 {:.3g}'.format(_style_er[2]))


def _setup_model(model_class, log_path):
    files = glob.glob(log_path + '/checkpoints/*.ckpt')
    files.sort()
    ckpt_path = files[-1]
    return model_class.load_from_checkpoint(checkpoint_path=ckpt_path)


def _pad_terminal(tensor, width):
    _head = np.repeat(tensor[:1], width - 1)
    _tail = np.repeat(tensor[-1:], width)
    return np.concatenate([_head, tensor, _tail])


def evaluate_distance(content, style, transfer):
    """
    Evalution for transferred results
    :param content: joint positions of content genstures
    :param style: joint positions of style genstures
    :param transfer: joint positions of stylized gestures by proposed method
    :return: various distance metrics
    """
    _effector_indices = [7, 11]  # root joint (0) non-inclusive !
    _upper_indices = [4, 5, 6, 7, 8, 9, 10, 11]
    _cut = 60

    _clip = content.shape[0] if content.shape[0] < transfer.shape[0] else transfer.shape[0]
    content = content[:_clip, :, :]
    transfer = transfer[:_clip, :, :]

    _cont_errors = similar_content_metric(content[:, _effector_indices, :],
                                          transfer[_cut:-_cut, _effector_indices, :], _cut)

    _style_errors = similar_style_metric(transfer[_cut:-_cut, _upper_indices, :],
                                         style[_cut:-_cut, _upper_indices, :])

    return _get_metrics(_cont_errors), _get_metrics(_style_errors)


def _get_metrics(errors):
    _sorted = np.sort(errors)[::-1]
    n_25p = errors.shape[0] // 4
    return np.array([np.mean(errors), np.max(errors), np.mean(_sorted[:n_25p])])


def _get_statistics(tensor, width):
    """
    Calculate statistical values within a period of width frames
    :param tensor: tensor whose statistical feature is computed
    :param width: the number of frames in computing statistical values
    :return:
    """
    _means = []
    _stds = []
    for _head in range(0, tensor.shape[0] - width, width // 2):
        _dist = []
        _seg = tensor[_head:_head+width, :, :]
        _mean = np.mean(_seg, axis=0).astype(dtype=np.double)
        _means.append(_mean)
        _var = np.var(_seg, axis=0).astype(dtype=np.double)
        _stds.append(np.sqrt(_var))

    return np.array(_means), np.array(_stds)


def similar_content_metric(content, transfer, cut_offset):
    """
    Compute content similarity metric
    :param content: clipped joint positions of content gesture
    :param transfer: clipped joint positions of tansferred gesture
    :param cut_offset: cut off frames for omitting starting motions
    :return:
    """
    _content = content[cut_offset:transfer.shape[0] + cut_offset, :, :]
    _content = _content.reshape(_content.shape[0], -1)
    _transfer = transfer.reshape(transfer.shape[0], -1)
    distance, path = fastdtw(_content, _transfer, dist=euclidean)

    _dists = []
    for _p in path:
        _dists.append(np.linalg.norm(_content[_p[0], :] - _transfer[_p[1], :]))
    return np.array(_dists)


def similar_style_metric(transfer, reference, width = 128):  # input sensors of transfer and referenced style gestures
    """
    Compute style similarity metric
    :param transfer: clipped joint positions of tansferred gesture
    :param reference: clipped joint positions of style gesture
    :param width: frame length (period) in computing statistical features of style
    :return:
    """
    stylized_veloc = 60. * (transfer[1:, :, :] - transfer[:-1, :, :])
    reference_veloc = 60. * (reference[1:, :, :] - reference[:-1, :, :])

    xf_mean, xf_std = _get_statistics(stylized_veloc, width)
    gt_mean, gt_std = _get_statistics(reference_veloc, width)

    _errors = []
    for _mean, _std in zip(xf_mean, xf_std):
        _dif_mean = np.sum(np.linalg.norm(gt_mean - _mean, axis=2, ord=2), axis=1)
        _dif_std = np.sum(np.linalg.norm(gt_std - _std, axis=2, ord=2), axis=1)
        _errors.append(np.min(_dif_mean + _dif_std))
    _errors = np.array(_errors)

    return _errors


if __name__ == '__main__':
    evaluation(model=GestureStyleTransformer, version=['SCA'], type='inter')
