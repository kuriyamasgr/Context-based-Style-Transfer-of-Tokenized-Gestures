# -*- coding: utf-8 -*-
# @author: kuriyama

import numpy as np
import copy
import torch
from torch.utils import data
from .quaternion import qexp, expq, quat_euler, euler_quat, qmul, slerp, qrot
from .bvh import BVH


def _fetch_site_indices(linkage):
    site_indices = []
    for n in range(len(linkage)):
        if n not in linkage:  # n-th joint has no children
            site_indices.append(n)

    return site_indices


def get_joint_positions(rotations, offsets, linkage, root_positions=None, flatten=False, use_torch=False):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where B = number of batches, L = sequence length, J = number of joints):
     -- rotations: ([B], L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: ([B], L, 3) tensor describing the root joint positions.
    """
    if use_torch and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if root_positions is not None and rotations.shape[0] != root_positions.shape[0]:
        rotations = rotations[:root_positions.shape[0], :, :]

    batch_mode = (len(rotations.shape) == 4)
    if batch_mode:
        num_batches = rotations.shape[0]
        frames_per_batch = rotations.shape[1]
        total_frames = num_batches * frames_per_batch
    else:
        total_frames = rotations.shape[0]

    null_rotations = None
    if root_positions is None:
        _omit_root = True
        if use_torch:
            root_positions = torch.zeros(total_frames, 3).double().to(device)
            null_rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).double().to(device)  # We reset orientation of root !
            null_rotations = null_rotations.expand(total_frames, 4)
        else:
            root_positions = np.zeros([total_frames, 3])
            null_rotations = np.array([1.0, 0.0, 0.0, 0.0])  # We also reset orientation of root !
            null_rotations = np.expand_dims(null_rotations, axis=0)
            null_rotations = np.repeat(null_rotations, total_frames, axis=0)
    else:
        _omit_root = False
        if batch_mode:
            root_positions = root_positions.view(-1, 3)

    if batch_mode:  # this always uses torch
        rotations = rotations.contiguous().view(-1, rotations.shape[2], rotations.shape[3])
        if len(offsets.shape) == 3:  # static along time
            offsets = offsets.expand(total_frames, offsets.shape[1], offsets.shape[2])
        else:
            offsets = offsets.view(-1, offsets.shape[2], offsets.shape[3])
    else:
        if len(offsets.shape) == 2:  # static along time
            offsets = np.expand_dims(offsets, axis=0)
            offsets = np.repeat(offsets, total_frames, axis=0)

    # Parallelize along the batch and time dimensions
    positions_world = []
    rotations_world = []
    sample_positions = []
    _site_indices = _fetch_site_indices(linkage)

    ridx = 0  # row indices of rotations
    for i in range(offsets.shape[1]):
        if linkage[i] == -1:  # for root joint...
            positions_world.append(root_positions)
            if _omit_root:
                rotations_world.append(null_rotations)
            else:
                rotations_world.append(rotations[:, 0, :])
            ridx += 1
        elif i not in _site_indices:
            _position = qrot(rotations_world[linkage[i]], offsets[:, i, :], torch=use_torch) \
                        + positions_world[linkage[i]]
            positions_world.append(_position)
            if ridx < rotations.shape[1]:
                rotations_world.append(qmul(rotations_world[linkage[i]], rotations[:, ridx, :], torch=use_torch))
            sample_positions.append(_position)  # sample only is_sampled nodes
            ridx += 1
        else:  # fixed joints such as sites or fingures
            rotations_world.append(None)
            positions_world.append(None)

    if use_torch:
        _positions = torch.stack(sample_positions, dim=0).permute(1, 0, 2)
    else:
        _positions = np.transpose(np.array(sample_positions), (1, 0, 2))

    if flatten:
        return np.reshape(_positions, (_positions.shape[0], -1))
    else:
        return _positions


class MocapClip:
    """
    Class of motion capture clip, which corresponds to the data of each bvh file in our implementation
    """

    def __init__(self, npdata, index=None, fps=60):
        """
        :param npdata: numpy data representing each motion clip
        :param index: used when multiply sampling clips for every offset against higher time resolutions (fps)
        :param fps: the number of frames recorded per each second (frames per second)
        """
        if index is None:
            self.position = npdata['position']
            self.rotation = npdata['rotation']
        else:  # multiplex mode
            self.position = npdata['position'][index]
            self.rotation = npdata['rotation'][index]

        self.offset = npdata['offset']
        self.linkage = npdata['linkage']
        self.joints = npdata['joints']  # name list of joints
        self.label = npdata.get('label', None)  # name of this clip

        self.tensor = self.pack_tensor()
        self.tensor_label = np.array([_joint for _joint in self.joints if _joint != 'Site'])
        self.fps = fps

    def __len__(self):
        """
        :return: the number of frames included in this clip
        """
        return self.rotation.shape[0]

    def pack_tensor(self):
        """
        Convert the motion representation of rotations (quaternions) per joint into the tensor of exponential map
        :return: the sequence of poses represented as exponential maps of all joints along frames
        """
        _expmap_tensor = qexp(self.rotation)
        _expmap_tensor[:, 0, :] = regularize_root_rotation(self.rotation[:, 0, :])
        return _expmap_tensor.reshape(_expmap_tensor.shape[0], -1).astype(np.float32)

    def one_batch_tensor(self):
        """
        :return: unsqueeze the tensor data to construct one-batch representation
        """
        return torch.unsqueeze(torch.tensor(self.tensor), dim=0)

    def clip_tensor(self, frames):
        """
        Clips tensor from randomly selected head to have the length (period) of frames
        :param frames:
        :return: clipped tensors (frames, pose)
        """
        if frames > 0:
            _head = np.random.randint(0, self.tensor.shape[0] - frames + 1)
            return self.tensor[_head:_head+frames, :]
        else:
            return self.tensor  # No clipping

    def restore(self, tensor):
        """
        Restore motion representation of rotations (quaternions) per joint from the tensor of exponential map
        :param tensor: tensor of exponential map
        """
        if type(tensor) is torch.Tensor:
            tensor = tensor.detach().numpy()
        if len(tensor.shape) == 3:
            tensor = tensor[0, :, :]
        _origin_root = self.rotation[:, 0, :]
        _update_rotation = expq(tensor.reshape(tensor.shape[0], -1, 3))  # exponential_map to quaternion

        if _origin_root.shape[0] < _update_rotation.shape[0]:
            _update_rotation = _update_rotation[:_origin_root.shape[0], :, :]
        else:
            _origin_root = _origin_root[:_update_rotation.shape[0], :]

        self.rotation = _update_rotation
        self.rotation[:, 0, :] = restore_root_rotation(_origin_root, _update_rotation[:, 0, :])

        _updated_frames = self.rotation.shape[0]
        self.position = self.position[:_updated_frames, :]

    def to_numpy(self, x_offset=None):
        """
        Convert the variables of this class into the numpy dictionary format
        :param x_offset: offset value to shift the position along x-axis
        :return: numpy dictionary representing the local variables of this class
        """
        _position = copy.deepcopy(self.position)
        if x_offset is not None:
            _position[:, 0] += x_offset
        return {'position': _position, 'rotation': self.rotation,
                'offset': self.offset, 'linkage': self.linkage, 'label': self.label}

    def set_position(self, flatten=False):
        """
        Calculate all joint positions
        :param flatten: True when flattens joints' dimension
        """
        self.position = get_joint_positions(rotations=self.rotation, offsets=self.offset, linkage=self.linkage, flatten=flatten)

    def search_joint_indices(self, labels, tensor_index=False):
        """
        Seach joint indices from the given labels
        :param labels: list of the joint names
        :param tensor_index: set True when detecting the indices among target joints included in a tensor, if False among all joints
        :return: list of indices of the joints whose names correspond to labels
        """
        if tensor_index:
            return [np.where(self.tensor_label == _label)[0][0] for _label in labels]
        else:
            return [np.where(self.joints == _label)[0][0] for _label in labels]

    def mirror_joints(self):
        """
        :return: labels of left and right joints of arms
        """
        if 'LShoulder' in self.joints:
            joints_left = self.search_joint_indices(labels=['LShoulder', 'LUArm', 'LFArm', 'LHand'], tensor_index=True)
            joints_right = self.search_joint_indices(labels=['RShoulder', 'RUArm', 'RFArm', 'RHand'], tensor_index=True)
        else:
            joints_left = self.search_joint_indices(labels=['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'], tensor_index=True)
            joints_right = self.search_joint_indices(labels=['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'], tensor_index=True)

        return joints_left, joints_right

    def add_mirror_motion(self):
        """
        :return: mirroring motion of this clip
        """
        joints_left, joints_right = self.mirror_joints()

        mirrored = copy.deepcopy(self)
        mirrored.rotation[:, joints_left, :] = self.rotation[:, joints_right, :]
        mirrored.rotation[:, joints_right, :] = self.rotation[:, joints_left, :]
        mirrored.rotation[:, :, [2, 3]] *= -1
        mirrored.position[:, 0] *= -1

        self.append(mirrored)

    def append(self, target, transit=True, period=30):
        """
        Append pose of other clip to this clip
        :param target: appended clip
        :param transit: if True, use linear transition in transition
        :param period: frame period of transition
        """
        _position = target.position
        _rotation = target.rotation
        if transit:
            step = 1. / period
            ratio = 0.0
            for idx in range(period):
                self.position[-period+idx, :] = (1. - ratio) * self.position[-period+idx, :] + ratio * _position[idx, :]
                self.rotation[-period+idx, :, :] = slerp(self.rotation[-period+idx, :, :], _rotation[idx, :, :], ratio)
                ratio += step
            self.position = np.append(self.position, _position[period:, :], axis=0)
            self.rotation = np.append(self.rotation, _rotation[period:, :, :], axis=0)
        else:
            self.position = np.append(self.position, _position, axis=0)
            self.rotation = np.append(self.rotation, _rotation, axis=0)


class MocapSet(data.Dataset):
    """
    Class of the set of motion data (clip)
    """
    def __init__(self, data_path, num_frames=-1, fps=60, multiplex=False, verbose=False):
        """
        :param data_path: list of paths for every motion (BVH) files
        :param num_frames: the number of frames clipped in training
        :param fps: the number of frames recorded per each second (frames per second)
        :param multiplex: if True, multiple MocapClip are sampled for a single file by shifting frame offset when down-sampling
        :param verbose: if True, message display is activated in parsing BVH files
        """
        if type(data_path) is not list:
            data_path = [data_path]

        self.num_frames = num_frames
        self.fps = fps
        self.multiplex = multiplex

        self.clips = []
        for _path in data_path:
            self.load_bvh(_path, fps=self.fps, verbose=verbose)

    def __len__(self):
        """
        :return: the number of motion clips (MocapClip) included in this class
        """
        return len(self.clips)

    def __getitem__(self, index):
        """
        :param index: index of motion clips (MocapClip)
        :return: the tensor of motion clips (this function is called by dataloader to allocated to construct a mini-batch
        """
        return self.clips[index].clip_tensor(self.num_frames)

    def clip(self, query):
        """
        Retrieve the moction clip by giving index or string (label)
        :param query:
        :return:
        """
        _type = type(query)
        if _type is str:
            for _clip in self.clips:
                if _clip.label == query:
                    return _clip
            print('Clip of', query, 'does not exists in this MocapSet! None clip is returned')
            return None
        elif _type is int:
            if 0 <= query < len(self.clips):
                return self.clips[query]
            else:
                print('Clip index', query, 'is out of range! None clip is returned')
                return None
        else:
            print('Query type of', _type, 'is undefined! None clip is returned')
            return None

    def clip_by_labels(self, labels):
        """
        Retrieve a list of motion clips from a list of labels
        :param labels: labels of motion clips
        :return: numpy array of motion clips
        """
        _clip_list = []
        for _label in labels:
            _clip_list.append(self.clip(_label))

        return np.array(_clip_list)

    def load_bvh(self, path, fps, verbose):
        """
        Load a BVH file
        :param path: path to a BVH file
        :param fps: frames per second used when sampling a BVH file
        :param verbose: if True, message display is activated when parsing a BVH file
        """
        _bvh = BVH()
        _bvh.parse(path, verbose=verbose)
        _npdata = _bvh.to_numpy(fps=self.fps, multiplex=self.multiplex)
        if self.multiplex is False:
            self.clips.append(MocapClip(_npdata, fps=fps))
        else:
            for i in range(len(_npdata['position'])):
                self.clips.append(MocapClip(_npdata, i))

    def sampling_weight(self):
        """
        Adjust the probability of sampling according to the length of frames (period) of each motion clip
        :return: list of probability sampled from this set as training samples
        """
        probs = []
        num_clips = 0
        for _clip in self.clips:
            frames = _clip.rotation.shape[0]
            probs.append(frames)
            num_clips += frames // self.num_frames

        return np.array(probs)/np.sum(probs), num_clips

    def setup_mirror_motions(self):
        """
        Caluculation mirroring motions for all motion clips included
        """
        _mirrors = []
        for _clip in self.clips:
            _clip.add_mirror_motion()


def regularize_root_rotation(root_rotation):
    """
    Regularize the rotation of root joint by resetting the rotational components anlog a vertical (Y-) axis
    :param root_rotation: sequence of rotations (quaternions) of a root joint
    :return: sequence of regularized rotations converted into exponential map representation
    """
    eulers = quat_euler(root_rotation, order='yzx')
    eulers[:, 1] = 0.0  # reset y-axis rotations
    return qexp(euler_quat(eulers, order='yzx'))  # return exponential map!


def restore_root_rotation(origin, updated):
    """
    Restore rotations of a root joint
    :param origin: original data for implanting rotational component along vertical (Y-) axis
    :param updated: updated rotations whose rotational component along vertical (Y-) axis is implanted from original data
    :return: restored rotations whose rotational component along vertical axis is copied from original data
    """
    eulers = quat_euler(origin, order='yzx')
    eulers[:, ::2] = 0.0  # preserve only y-axis rotations
    _v_rot_quat = euler_quat(eulers, order='yzx')
    return qmul(_v_rot_quat, updated)
