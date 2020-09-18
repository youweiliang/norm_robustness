import torch
# import torch.nn as nn


def generic_power_method(affine_fun, input_size, eps=1e-8,
                         max_iter=500, device='cuda'):
    """ Return the highest singular value of the linear part of
    `affine_fun` and it's associated left / right singular vectors.

    INPUT:
        * `affine_fun`: an affine function
        * `input_size`: size of the input
        * `eps`: stop condition for power iteration
        * `max_iter`: maximum number of iterations
        * `use_cuda`: set to True if CUDA is present

    OUTPUT:
        * `eigenvalue`: maximum singular value of `affine_fun`
        * `v`: the associated left singular vector
        * `u`: the associated right singular vector

    NOTE:
        This algorithm is not deterministic, depending of the random
        initialisation, the returned eigenvectors are defined up to the sign.

        If affine_fun is a PyTorch model, beware of setting to `False` all
        parameters.requires_grad.

    TEST::
        >>> conv = nn.Conv2d(3, 8, 5)
        >>> for p in conv.parameters(): p.requires_grad = False
        >>> s, u, v = generic_power_method(conv, [1, 3, 28, 28])
        >>> bias = conv(torch.zeros([1, 3, 28, 28]))
        >>> linear_fun = lambda x: conv(x) - bias
        >>> torch.norm(linear_fun(v) - s * u) # should be very small

    TODO: more tests with CUDA
    """
    affine_fun.to(device)
    zeros = torch.zeros(input_size).to(device)
    affine_fun.eval()
    affine_fun.weight.requires_grad = False
    affine_fun.bias.requires_grad = False
    bias = affine_fun(zeros)
    linear_fun = lambda x: affine_fun(x) - bias

    # Initialise with random values
    v = torch.randn(input_size).to(device)
    previous = torch.randn(input_size).to(device)
    

    v.div_(torch.norm(v))

    stop_criterion = False
    it = 0
    while not stop_criterion:
        v.requires_grad = True
        loss = torch.norm(linear_fun(v))**2
        loss.backward()
        v.requires_grad = False
        previous.copy_(v.data)
        v.copy_(v.grad.data)
        v.div_(torch.norm(v))

        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    
    u = linear_fun(v)  # unormalized left singular vector
    eigenvalue = torch.norm(u)
    # u.div_(eigenvalue)
    return eigenvalue  # , u, v

