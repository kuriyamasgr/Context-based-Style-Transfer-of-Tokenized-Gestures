# -*- coding: utf-8 -*-
# @author: kuriyama

import numpy as np
import torch as tc


def identity():
    return np.array([1., 0., 0., 0.])


def qmul(q, r, torch=False):
    """
    Multiply quaternion(s) q with quaternion(s) r.  qmul = q * r
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    if torch:
        if len(r.shape) == 2:
            r = tc.unsqueeze(r, dim=2)
            q = tc.unsqueeze(q, dim=1)
            terms = tc.bmm(r, q)
        else:
            terms = tc.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    else:
        terms = np.matmul(np.reshape(r, (-1, 4, 1)), np.reshape(q, (-1, 1, 4)))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    if torch:
        return tc.stack((w, x, y, z), dim=1).view(original_shape)
    else:
        dat = np.stack((w, x, y, z), axis=1)
        return np.reshape(dat, original_shape)


def qdifangle(q, r, cosine_dist=True):  # only for torch
    """
    Compute the difference of rotaions by q and r by rotaional angle
    :param q: quaternion
    :param r: referenced quaternion
    :param cosine_dist: true if consine distance is evaluated
    :return: rotational angle for rotaing r to q
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    assert q.shape == r.shape
    original_shape = list(q.shape)

    q = q.view(-1, 4)
    r = r.view(-1, 4)

    q[:, 1:] = - q[:, 1:]  # take conjugate (or inverse)
    difq = qmul(q, r, torch=True)
    if cosine_dist:
        return 1. - difq[:, 0].contiguous().view(original_shape[:-1])
    else:
        mask = difq[:, 0] > 1.0
        difq[mask, 0] = 1.0
        mask = difq[:, 0] < -1.0
        difq[mask, 0] = -1.0
        rot_dif = 2. * tc.acos(difq[:, 0])

        return rot_dif.view(original_shape[:-1])


def qrot(q, v, torch=False):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    if torch:
        q = q.view(-1, 4)
        v = v.view(-1, 3)
    else:
        q = np.reshape(q, (-1, 4))
        v = np.reshape(v, (-1, 3))

    qvec = q[:, 1:]
    if torch:
        uv = tc.cross(qvec, v, dim=1)
        uuv = tc.cross(qvec, uv, dim=1)
    else:
        uv = np.cross(qvec, v, axis=1)
        uuv = np.cross(qvec, uv, axis=1)

    if torch:
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
    else:
        return np.reshape((v + 2 * (q[:, :1] * uv + uuv)), original_shape)


def _qeuler(q, order, atan2, asin, clamp, epsilon=0):
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    order = order.lower()
    if order == 'xyz':
        e0 = atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        e1 = asin(clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        e2 = atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        e2 = atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        e0 = atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        e1 = asin(clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        e1 = asin(clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        e2 = atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        e0 = atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        e0 = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        e2 = atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        e1 = asin(clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        e1 = asin(clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        e0 = atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        e2 = atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        e2 = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        e1 = asin(clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        e0 = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        print('The order of axis for euler angles', order, 'is not permitted.')
        raise Exception

    return e0, e1, e2


def quat_euler(q, order, epsilon=0, torch=False):
    """
    Convert quaternion(s) to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3) of the given order in degrees
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    if torch:
        q = q.view(-1, 4)
        _euler = _qeuler(q, order, epsilon=epsilon, atan2=tc.atan2, asin=tc.asin, clamp=tc.clamp)
        return 180. * tc.stack(_euler, dim=1).view(original_shape) / np.pi
    else:
        q = np.reshape(q, (-1, 4))
        _euler = _qeuler(q, order, epsilon=epsilon, atan2=np.arctan2, asin=np.arcsin, clamp=np.clip)
        return 180. * _eulerfix(np.reshape(np.stack(_euler, axis=1), original_shape)) / np.pi


def _eulerfix(euler):
    """
    Enforce euler continuity across the time dimension by selecting
    the representation with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 3) or (L, 3), where L is the sequence length.
    Returns a tensor of the same shape.
    """
    assert len(euler.shape) == 2 or len(euler.shape) == 3
    assert euler.shape[-1] == 3
    original_shape = list(euler.shape)
    # Kanamori's Algorithm
    e = euler.copy()
    diff = e[1:, :] - e[:-1, :]
    plus2minus_mask = diff < -np.pi
    plus2minus_counts = np.cumsum(plus2minus_mask, axis=0)  # 符号が急に正から負に反転した回数を計算
    e[1:, :] += 2. * np.pi * plus2minus_counts           # 符号が反転した回数に応じて 2πを足す
    minus2plus_mask = diff > np.pi
    minus2plus_counts = np.cumsum(minus2plus_mask, axis=0)  # 符号が急に負から正に反転した回数を計算
    e[1:, :] -= 2. * np.pi * minus2plus_counts           # 符号が反転した回数に応じて 2πを引く

    return e.reshape(original_shape)


def expq(e, torch=False, epsilon=0.0001):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    if torch:
        e = e.contiguous()
        e = e.view(-1, 3)
        theta = tc.norm(e, p=2, dim=1).view(-1, 1)
        val = 0.5 * theta
        w = tc.cos(val).view(-1, 1)
        xyz = tc.sin(val) * e / (theta + epsilon)
        return tc.cat((w, xyz), dim=1).view(original_shape)
    else:
        e = e.reshape(-1, 3)
        theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
        val = 0.5 * theta
        w = np.cos(val).reshape(-1, 1)
        xyz = np.sin(val) * e / (theta + epsilon)
        return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def qexp(q, axis_angle=False, epsilon=0.0001):
    """
    Convert quaternions to axis-angle rotations (aka exponential maps).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.reshape(-1, 4)

    q[q[:, 0] < -1.0, 0] = -1.0  # clip invalid values
    q[q[:, 0] > 1.0, 0] = 1.0  # clip invalid values
    theta = 2.0 * np.arccos(q[:, 0])  # 0 <= theta <= 2 pi
    axis_sin = np.sin(0.5 * theta)  # 0 <= axis_sin <= 1
    axis_sin = np.expand_dims(axis_sin, axis=1)
    axis = q[:, 1:] / (axis_sin + epsilon)
    theta = np.expand_dims(theta, axis=1)
    flip_flag = theta > np.pi
    theta[flip_flag] -= 2.0 * np.pi  # -pi <= theta <= pi
    if axis_angle:
        theta_shape = original_shape.copy()
        theta_shape[-1] = 1
        return axis.reshape(original_shape), theta.reshape(theta_shape)
    else:
        expmap = np.multiply(theta, axis)
        return expmap.reshape(original_shape)


def euler_quat(euler, order):
    """
    Convert Euler angles to quaternions.
    Order of euler angles is given order, e.g., 'xyz', 'zxy', etc.
    """
    assert euler.shape[-1] == 3

    original_shape = list(euler.shape)
    original_shape[-1] = 4
    euler = euler.reshape(-1, 3)

    _rad_euler = np.pi * euler / 180.  # from degrees to radians
    rot = np.repeat(np.expand_dims(identity(), axis=0), euler.shape[0], axis=0)
    for e, axis in zip(_rad_euler.T, order.lower()):
        if axis == 'x':
            tmp = np.stack((np.cos(e/2), np.sin(e/2), np.zeros_like(e), np.zeros_like(e)), axis=1)
            rot = qmul(rot, np.stack((np.cos(e/2), np.sin(e/2), np.zeros_like(e), np.zeros_like(e)), axis=1))
        elif axis == 'y':
            rot = qmul(rot, np.stack((np.cos(e/2), np.zeros_like(e), np.sin(e/2), np.zeros_like(e)), axis=1))
        elif axis == 'z':
            rot = qmul(rot, np.stack((np.cos(e/2), np.zeros_like(e), np.zeros_like(e), np.sin(e/2)), axis=1))
        else:
            print('Axis symbol', axis, 'is not permitted !')

    if order in ['xyz', 'yzx', 'zxy']:  # Reverse antipodal representation to have a non-negative "w"
        rot *= -1

    rot = _qfix(rot)
    return rot.reshape(original_shape)


def _qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4) or (L, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3 or len(q.shape) == 2
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=len(q.shape) - 1)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def slerp(v0, v1, ratio):
    assert v0.shape[-1] == 4
    assert v1.shape[-1] == 4
    assert len(v0.shape) > 1
    assert len(v1.shape) > 1

    original_shape = list(v0.shape)
    v0 = v0.reshape(-1, 4)
    v1 = v1.reshape(-1, 4)

    dot = np.sum(np.multiply(v0, v1), axis=1)
    reverse_flag = dot < 0.0
    v1[reverse_flag, :] = -v1[reverse_flag, :]
    dot[reverse_flag] = -dot[reverse_flag]

    _threshold = 0.9995
    same_flag = dot > _threshold
    result = np.zeros(v0.shape)
    result[same_flag, :] = v0[same_flag, :] + ratio*(v1[same_flag, :] - v0[same_flag, :])

    dif_flag = np.logical_not(same_flag)
    dot[dot > 1.0] = 1.0  # correct illegal values
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    sin_theta_0[same_flag] = 1.0  # avoid zero division
    theta = ratio * theta_0
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    _res = s0[:, np.newaxis] * v0 + s1[:, np.newaxis] * v1
    result[dif_flag, :] = _res[dif_flag, :]

    return result.reshape(original_shape)
