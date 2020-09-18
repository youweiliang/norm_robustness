"""
Original license:
    Copyright 2018 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
Changes：
    Use torch format kernel as inputs instead of tensorflow format.
    Add some new functions for clipping the singular values of CNN layers.
"""

import torch
# import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for dev in gpu_devices:
    tf.config.experimental.set_memory_growth(dev, True)  # make tf not to occupy all GPU memory

"""
The 4D convolutional kernels in the input of these functions are in torch format,
i.e., out_channels, in_channels, kernel_size[0], kernel_size[1]
"""


def SVD_Conv_Tensor(filter, inp_shape, device='/gpu:0'):
    """ Find the singular values of the linear transformation
    corresponding to the convolution represented by conv on
    an n x n x depth input. """
    with tf.device(device):
        conv_tr = tf.constant(filter, dtype=tf.complex64)
        conv_shape = conv_tr.get_shape().as_list()
        padding = tf.constant([[0, 0], [0, 0],
                              [0, inp_shape[0] - conv_shape[2]],  # height
                              [0, inp_shape[1] - conv_shape[3]]]) # width
        if inp_shape[0] - conv_shape[2] < 0 or inp_shape[1] - conv_shape[3] < 0:
            return  # the input is smaller than kernel...
        transform_coeff = tf.signal.fft2d(tf.pad(conv_tr, padding))
        singular_values = tf.linalg.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]), compute_uv=False)
        sv = singular_values.numpy()
    
    return sv


def Clip_OperatorNorm(filter, inp_shape, clip_to, device='/gpu:0'):
    # filter = np.transpose(filter, (2, 3, 0, 1))

    with tf.device(device):
        conv_tr = tf.constant(filter, dtype=tf.complex64)
        conv_shape = conv_tr.get_shape().as_list()
        padding = tf.constant([[0, 0], [0, 0],
                              [0, inp_shape[0] - conv_shape[2]],  # height
                              [0, inp_shape[1] - conv_shape[3]]]) # width
        if inp_shape[0] - conv_shape[2] < 0 or inp_shape[1] - conv_shape[3] < 0:
            return  # the input is smaller than kernel...
        transform_coeff = tf.signal.fft2d(tf.pad(conv_tr, padding))
        D, U, V = tf.linalg.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]))
        norm = tf.reduce_max(D)
        D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex64)
        clipped_coeff = tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped),
                                         V, adjoint_b=True))
        clipped_conv_padded = tf.math.real(tf.signal.ifft2d(
                tf.transpose(clipped_coeff, perm=[2, 3, 0, 1])))
        clipped_filter = tf.slice(clipped_conv_padded,
                                  [0] * len(conv_shape), conv_shape)
        clipped_filter = clipped_filter.numpy()

    return clipped_filter


def clip_conv(net, clip_to):
    for m in net.modules():
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            _, _, k1, k2 = m.weight.size()
            h, w = m.inp_shape
            if h < k1 or w < k2:
                continue
            kernel = m.weight.data.cpu().numpy()
            kernel = Clip_OperatorNorm(kernel, m.inp_shape, clip_to)
            if kernel is not None:
                kernel = torch.from_numpy(kernel)
                m.weight.data.copy_(kernel)


def record_input_size(net):
    for m in net.modules():
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m._original_forward_ = m.forward
            bind(m, forward)


def forward(self, x):
    if not hasattr(self, 'inp_shape'):
        _, _, h, w = x.size()
        self.inp_shape = (h, w)
    return self._original_forward_(x)


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method

