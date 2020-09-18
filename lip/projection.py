import torch


def proj_l1_ball(y, c):
    if y.dim() > 1:
        shape = y.size()
        y = y.flatten()
    else:
        shape = None
    p = y.sign()
    y = y.abs()

    sorted_y = torch.sort(y, descending=True)[0]
    cum_y = torch.cumsum(sorted_y, dim=0)

    m = y.numel()
    idx = torch.arange(1, m+1, device=y.device)
    zero = torch.zeros(1, device=y.device)
    tmp = (c - cum_y).div_(idx.float());
    sig = (sorted_y + tmp) > zero;
    rho = torch.max(idx[sig]) - 1

    zeta = tmp[rho]
    x = torch.max(y + zeta, zero)

    x = x.mul_(p)
    if shape is not None:
        x = torch.reshape(x, shape)

    return x


if __name__ == '__main__':
    y = (torch.rand(10) * (torch.rand(10) - 0.5).sign()).reshape(2,5)
    print(y)
    x = proj_l1_ball(y, 1)
    print(x)
    print(x.abs().sum())