import os
import torch
import timeit
import pickle
import torch.nn as nn
import tensorflow as tf
from lip.add_lip import bind_lip
from regularization.conv2d_sv import SVD_Conv_Tensor
from timing.max_eigenvalue import generic_power_method

assert tf.config.experimental.get_visible_devices("GPU") # run on GPU

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = 'cuda:0'  # for torch

stride = 1  # SVD_Conv_Tensor only support stride == 1

test_set = [((3, 3), 32, 32),
            ((3, 3), 32, 128),
            ((3, 3), 128, 256),
            ((3, 3), 256, 512),
            # ((3, 3), 512, 1024),
            ((5, 5), 256, 128),
            ((5, 5), 512, 256),]
            # ((5, 5), 1024, 512)]

input_shape = (32, 32)  # image size

result = []

for (h, w), d_in, d_out in test_set:
    print(f'Testing: ', (h, w), d_in, d_out)
    kernel = torch.randn(d_out, d_in, h, w)
    kernel_np = kernel.numpy()
    conv = nn.Conv2d(d_in, d_out, (h, w), stride=stride)
    conv.weight.data.copy_(kernel)

    input_size = (1, d_in, *input_shape)
    conv.to(device)

    powerit = timeit.timeit('generic_power_method(conv, '
        'input_size, max_iter=500, device="cuda:0")', 
        globals=globals(), number=100)

    svc_t = timeit.timeit('SVD_Conv_Tensor(kernel_np, input_shape)', globals=globals(), number=100)
    
    bind_lip(conv, norm='1-norm')
    image = torch.rand(1,  d_in, *input_shape).to(device)
    
    conv(image)
    conv.to(device)
    L1 = timeit.timeit('conv.lip()', globals=globals(), number=100)

    conv.norm = 'inf-norm'
    Linf = timeit.timeit('conv.lip()', globals=globals(), number=100)

    conv.norm = '1-norm'
    conv.to('cpu')
    L1_cpu = timeit.timeit('conv.lip()', globals=globals(), number=100)

    conv.norm = 'inf-norm'
    Linf_cpu = timeit.timeit('conv.lip()', globals=globals(), number=100)

    result.append((powerit, svc_t, L1, Linf, L1_cpu, Linf_cpu))
    print('powerit, svc_t, L1, Linf, L1_cpu, Linf_cpu:', result[-1])

    with open('./timing/timing.pkl', 'wb') as f:
        pickle.dump(result, f)

    