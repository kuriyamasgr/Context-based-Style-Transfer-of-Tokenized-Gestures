# -*- coding: utf-8 -*-
# @author: kuriyama

import numpy as np
from .quaternion import euler_quat, quat_euler

def str2floats(val, idx, is_position=False):
    """
    Convert string to the numpy array of floats
    :param val: string values
    :param idx: offset index to extract float values
    :param is_position: if True, return positional representation
    :return: numpy array of float values representing positions or rotations
    """
    if is_position:
        k = 1.0
    else:
        k = np.pi / 180.
    try:
        return np.array([k * float(val[idx]), k * float(val[idx+1]), k * float(val[idx+2])])
    except ValueError:
        print('Motion value format is unreadable !...')
        return np.array([0.0, 0.0, 0.0])


def set_channel_order(eulers, from_order):
    """
    Permute the order of 3D vector
    :param eulers: array of 3D vectors representing Euler angles
    :param from_order: order of 3D (XYZ) channels
    :return: permuted 3D vector of Euler angles
    """
    assert eulers.shape[-1] == 3

    order = from_order.lower()
    new_order = []
    for c in order:
        if c == 'x':
            new_order.append(0)
        elif c == 'y':
            new_order.append(1)
        elif c == 'z':
            new_order.append(2)
        else:
            print('Illegal axis label in set_channel_order !...')
            raise Exception

    return eulers[..., new_order]


class BVH:
    """
    Class of BVH data
    """
    def __init__(self):
        self.num_frames = 0
        self.positions = None
        self.rotations = None
        self.file_path = None
        self.root_node = None
        self.node_list = []
        self.channel_order = None
        self.fps = None
        self.label = None
        self.scale_factor = 1.0
        self.num_channels = []

    def parse(self, file_path, skeleton_only=False, verbose=False):
        """
        Parsing a BVH file
        :param file_path: path to a BVH file
        :param skeleton_only: if True, only load the data of skeleton while skipping the information of motions
        :param verbose: if True, display the message for monitoring this process
        """
        with open(file_path, 'r') as bvh:
            self.file_path = file_path
            self.label = file_path[file_path.rfind('/')+1:file_path.rfind('.')]
            tokenlist = []
            line = True
            while line:
                line = bvh.readline()
                tokens = line.split()
                if tokens[0] == 'MOTION':
                    break
                tokenlist.extend(tokens)
            self._analyze_hierarchy(tokenlist)

            if not skeleton_only:
                tokens = bvh.readline().split()
                tok = tokens.pop(0)
                if tok != "Frames:":
                    return False
                else:
                    self.num_frames = int(tokens.pop(0))
                    if verbose:
                        print("Total number of frames:", self.num_frames)
                    
                tokens = bvh.readline().split()
                tok = tokens.pop(0)
                tok2 = tokens.pop(0)
                if tok != "Frame" or tok2 != "Time:":
                    return False 
                else:
                    self.fps = int(1.0 / float(tokens.pop(0)))
                    if verbose:
                        print("Captured fps:", self.fps)
                self._parse_motion(bvh)

    def _parse_motion(self, bvh):
        """
        Parsing the data about motions
        :param bvh: string data obtained from a BVH file
        """
        frame = 0
        self.positions = np.zeros([self.num_frames, 3], dtype=np.float32)
        self.rotations = np.zeros([self.num_frames, len(self.num_channels), 3], dtype=np.float32)
        while True:
            line = bvh.readline()
            if line is None:
                break
            eulers = line.split()
            if len(eulers) == 0:  # end of the file
                break
            total_channels = np.sum(self.num_channels)
            if len(eulers) != total_channels:
                print('Data dimension not consistent', len(eulers), ' <-> ', total_channels)
                return

            head = 0
            for n, ch in enumerate(self.num_channels):
                if n == 0:
                    self.positions[frame, :3] = str2floats(eulers, 0, is_position=True)
                    self.rotations[frame, 0, :] = str2floats(eulers, 3)
                else:
                    if ch == 6:
                        self.rotations[frame, n, :] = str2floats(eulers, head+3)
                    else:  # ch == 3
                        self.rotations[frame, n, :] = str2floats(eulers, head)
                head += ch
            frame += 1

        if 30.0 < np.mean(self.positions[:, 1]) < 150.:  # position is given in centimeters
            self.scale_factor = 0.01  # scale to meter size

    def _analyze_hierarchy(self, tokenlist):
        """
        Parsing the data about a body skeleton
        :param tokenlist: string data obtained from a BVH file
        """
        tok = tokenlist.pop(0)
        while True:
            if tok == "HIERARCHY":
                tok = tokenlist.pop(0)
                if tok != "ROOT":
                    return
                if not self._make_node(None, tokenlist, is_end=False):
                    return
            if len(tokenlist) == 0:
                return
    
    def _parse_node(self, node, tokenlist):
        """
        Paring the string data representing each node of a body skeleton
        :param node: node object (BVHNode class)
        :param tokenlist: string data obtained from a BVH file
        :return: return True if the paring is processed successfully, False otherwise
        """
        tok = tokenlist.pop(0)
        while tok is not None:
            if tok == "ROOT" or tok == "JOINT" or tok == "End":
                if not self._make_node(node, tokenlist, is_end=(tok == "End")):
                    print("Error in Node generation")
                    return False			
            elif tok == "OFFSET":
                val = [0, 0, 0]
                for i in range(3):
                    tok = tokenlist.pop(0)
                    if tok is None:
                        print("OFFSET Not exist")
                        return False
                    val[i] = float(tok)
                node.offset = val
            elif tok == "CHANNELS":
                self.num_channels.append(node.set_channel_order(tokenlist, self))
            elif tok == "}":
                return True
            tok = tokenlist.pop(0)
        return True
    
    def _make_node(self, node, tokenlist, is_end):
        """
        Construct a node object (BVHNode)
        :param node: parent node
        :param tokenlist: string data obtained from a BVH file
        :param is_end: set True if the newly created node is a terminal node
        :return:
        """
        _child = self.BVHNode()
        _child.is_end = is_end
        tok = tokenlist.pop(0)
        _child.name = tok
        if node is None:
            _child.is_root = True
            _child.usePos = True
            self.root_node = _child
        else:  
            node.add_child(_child)
            _child.parent = node
        # if _child.name != 'Site':
        self.node_list.append(_child)
        tok = tokenlist.pop(0)
        if tok != "{":
            print(tok + " Error in _make_node")
            return False
        if not self._parse_node(_child, tokenlist):
            print("cannot parse node")
            return False
        _child.set_orientation()
        
        return True

    def to_quats(self):
        """
        Convert Euler angle representations into quaternion
        :return: quaternion values
        """
        return euler_quat(set_channel_order(self.rotations, self.channel_order), order=self.channel_order)

    def remove_node(self, omit_labels):
        """
        Remove the node from the skeleton information
        :param omit_labels: names of nodes (joints) to be removed
        """
        n_list = []
        for nd in self.node_list:
            is_omit = False
            for label in omit_labels:
                if label in nd.name:
                    is_omit = True
                    break
            if not is_omit:
                n_list.append(nd)

        self.node_list = n_list

    def remove_tensor(self, omit_labels):
        """
        Reconstruct tensor of rorations by removing corresponding entries of removed joints
        :param omit_labels: names of nodes (joints) to be removed
        """
        use_flag = []
        for nd in self.node_list:
            if not nd.isEnd:
                is_omit = False
                for label in omit_labels:
                    if label in nd.name:
                        is_omit = True
                        break
                if is_omit:
                    use_flag.append(False)
                else:
                    use_flag.append(True)

        self.rotations = self.rotations[:, use_flag, :]

    def save_updated_motion(self, save_path, clip_range=None, num_skip=None, update=None):
        """
        Save updated BVH file by replacing motion data
        :param save_path: path to a file to be saved
        :param clip_range: frame range of clipping
        :param num_skip: skip number in downsampling
        :param update: MocapClip class object for updating motion
        """
        save_file = open(save_path, 'w')
        if update is not None:
            update_lines = self.convert_motion_linetext(update)

        with open(self.file_path, 'r') as bvh:
            line = True
            while line:
                line = bvh.readline()
                save_file.write(line)
                tokens = line.split()
                if tokens[0] == 'MOTION':
                    break
            tokens = bvh.readline().split()
            tok = tokens.pop(0)
            if tok != "Frames:":
                return False
            else:
                _frms = tokens.pop(0)  # pop num_frames
                if clip_range is not None:
                    _frames = int(clip_range[1] - clip_range[0])
                else:
                    _frames = int(_frms)
                if num_skip is not None:
                    _num_frms = _frames // num_skip
                    if _frames > num_skip * _num_frms:
                        _num_frms += 1
                elif update is not None:
                    _num_frms = len(update)
                else:
                    _num_frms = _frames

                save_file.write("Frames: " + str(_num_frms) + "\n")

            line = bvh.readline()
            tokens = line.split()
            tok = tokens.pop(0)
            tok2 = tokens.pop(0)
            if tok != "Frame" or tok2 != "Time:":
                return False
            else:
                float(tokens.pop(0))  # pop frame rate
                _fr_time = line.split()
                if num_skip is not None:
                    save_file.write('Frame Time: ' + str(float(_fr_time[2]) * float(num_skip)) + '\n')
                else:
                    save_file.write('Frame Time: ' + _fr_time[2] + '\n')

            idx = 0
            while True:
                line = bvh.readline()
                if line is None or len(line) == 0 or (clip_range is not None and idx == clip_range[1]):
                    save_file.close()
                    return
                if num_skip is None or idx % num_skip == 0:
                    if clip_range is None or idx > clip_range[0]:
                        if update is not None and idx < len(update):
                            save_file.write(update_lines[idx])
                        elif update is None:
                            save_file.write(line)
                idx += 1

    def convert_motion_linetext(self, clip):
        """
        Convert motion data of the MocapClip to BVH's motion representation
        :param clip: MocapClip object for updating motion
        :return: strings of motion date
        """
        line_buffer = []
        _root_position = clip.position
        if len(_root_position.shape) == 3:
            _root_position = _root_position[:, 0, :]

        for _rpos, _jrots in zip(_root_position, clip.rotation):
            rp = _rpos / self.scale_factor
            line_pos_str = str(rp[0]) + '    ' + str(rp[1]) + '    ' + str(rp[2])
            line_rot_str = ''
            jeulers = set_channel_order(quat_euler(_jrots, order= self.channel_order), from_order=self.channel_order)
            for je in jeulers:
                jedeg = 180.0 * je / np.pi  # convert from radian to degree
                line_rot_str += '    ' + str(jedeg[0]) + '    ' + str(jedeg[1]) + '    ' + str(jedeg[2])
            line_buffer.append(line_pos_str + line_rot_str + '\n')

        return np.stack(line_buffer, axis=0)

    def to_numpy(self, fps=30, multiplex=False, verbose=False):
        """
        Convert local varoables into numpy format
        :param fps: frames per second of cnverted data
        :param multiplex: if True, frames are sampled by shifting the offset when down-sampling
        :param verbose: if True, message display is activated
        :return: dictionary of numpy arrays
        """
        if self.fps > fps:
            _skip = int(self.fps / fps)
            _pos_list, _rot_list = self.down_sampling(skip=_skip, multiplex=multiplex)
            if verbose:
                print('Numpy data is down-sampled from', self.fps, 'to', fps, 'by visiting every', _skip, 'frames')
        else:
            _pos_list = [self.positions]
            _rot_list = [self.rotations]

        _root_positions = []
        for _positions in _pos_list:
            _root_positions.append(self.scale_factor * _positions)

        _joint_rorations = []
        for _rotations in _rot_list:
            _rots = np.zeros(_rotations.shape)
            for n, ax in enumerate(self.channel_order):
                if ax == 'X':
                    idx = 0
                elif ax == 'Y':
                    idx = 1
                else:
                    idx = 2
                _rots[:, :, idx] = _rotations[:, :, n]
            _joint_rorations.append(euler_quat(_rots, order=self.channel_order))  # get quaternion

        _offset = [node.offset[:] for node in self.node_list]
        _offset = self.scale_factor * np.array(_offset)  # 骨格長の正規化
        _parent = []
        _joints = []  # List of joint's name (text)
        for node in self.node_list:
            _parent.append(node.get_parent_index(self.node_list))
            _joints.append(node.name)

        if multiplex:
            return {'position': _root_positions, 'rotation': _joint_rorations, 'joints': np.array(_joints),
                    'offset': _offset, 'linkage': np.array(_parent), 'label': self.label}
        else:
            return {'position': _root_positions[0], 'rotation': _joint_rorations[0], 'joints': np.array(_joints),
                    'offset': _offset, 'linkage': np.array(_parent), 'label': self.label}

    def down_sampling(self, skip=2, multiplex=False):
        """
        Down sampling motion data
        :param skip: the number of skipped frames
        :param multiplex: if True, frames are sampled by shifting the offset when down-sampling
        :return: list of arrays representing positions and rotations of joints
        """
        _positions = []
        _rotations = []
        if multiplex:
            for offset in range(skip):
                _positions.append(self.positions[offset::skip, :])
                _rotations.append(self.rotations[offset::skip, :])
            self.num_frames = _positions[0].shape
        else:
            _positions.append(self.positions[::skip, :])
            _rotations.append(self.rotations[::skip, :, :])
            self.num_frames = self.positions.shape[0]
        return _positions, _rotations

    class BVHNode:
        """
        Node class representing each joint of a body skeleton
        """
        def __init__(self):
            self.name = ""
            self.child = []
            self.offset = []
            self.orientation = []
            self.rot_order = None
            self.pos_order = None
            self.is_root = False
            self.is_end = False
            self.use_pos = False
            self.length = 0.0

        def add_child(self, c):
            """
            Append the child node
            :param c: node object to be appended
            """
            self.child.append(c)

        def get_parent_index(self, node_list):
            """
            Obtain parent index of this node
            :param node_list: list of parent indices
            """
            for node in node_list:
                if self in node.child:
                    return node_list.index(node)
            return -1

        def set_orientation(self):
            """
            Compute the rotation of the node (joint)
            """
            self.orientation = np.array([0., 0., 1., 0.])
            ydown = np.array([0., -1., 0.])
            ofs = np.array(self.offset)
            self.length = np.linalg.norm(ofs)
            if self.length > 0.0001:
                ofs /= self.length
            if ofs[1] > -0.999:
                if ofs[1] > 0.999:
                    self.orientation = np.array([0., 0., 1., np.pi])
                else:
                    nrm = np.cross(ydown, ofs)
                    nlen = np.linalg.norm(nrm)
                    if nlen > 0.00001:
                        self.orientation[0:3] = nrm / nlen
                    else:
                        self.orientation[0:3] = nrm
                    self.orientation[3] = np.arccos(-ofs[1])

        def set_channel_order(self, tokenlist, bvh):
            """
            Set the order of XYZ channels
            :param tokenlist: string data obtained from a BVH file
            :param bvh: object of a BVH class
            """
            tok = tokenlist.pop(0)
            chnum = int(tok)
            axis_order = ''
            position_order = ''
            for _ in range(chnum):
                tok = tokenlist.pop(0)
                if 'position' in tok:
                    position_order += tok[0]
                    self.use_pos = True
                else:
                    axis_order += tok[0]  # cocatenate X, Y, or Z
            self.rot_order = axis_order
            if bvh.channel_order is None:
                bvh.channel_order = axis_order
                # print('Order of rotation axes:', bvh.channel_order)
            if self.use_pos:
                self.pos_order = position_order
                return 6  # channels are position(3) + rotation(3)
            else:
                return 3  # channels are rotation(3)
