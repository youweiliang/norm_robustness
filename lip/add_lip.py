import torch
import math
import warnings
from .partition_kernel import partition
from .projection import proj_l1_ball

#######################################################
# Note: all valuable `alpha` in this file is deprecated
NORM = '1-norm'
MMT = 0.5
ALPHA = 1
BETA = 0.01


def bind_lip(model, norm=NORM, mmt=MMT, beta=BETA, verbose=False):
    for m in model.modules():
        classname = m.__class__.__name__
        if is_conv2d(m):
            add_lip(m, norm, mmt=mmt, beta=beta)
            m._verbose_ = verbose
        elif classname == 'Linear':
            add_lip_linear(m, norm, mmt=mmt, beta=beta)
            if verbose:
                print(f'Added norm tracking to {m}')
        elif classname.find('BatchNorm') != -1:
            add_lip_bn(m, norm, mmt=mmt, beta=beta)
            if verbose:
                print(f'Added norm tracking to {m}')

    # bind(model, config_lip)
    bind(model, add_lip_grad)
    bind(model, project)
    bind(model, project_bn)
    bind(model, calc_lip)
    bind(model, lip_param)

'''
def config_lip(self):
    for m in self.modules():
        classname = m.__class__.__name__
        if is_conv2d(m):
            m.create_view()
'''


def add_lip_grad(self, linear=True, conv=True, bn=False):
    lipc = 0
    all_lip = []
    for m in self.modules():
        classname = m.__class__.__name__
        if classname == 'Linear' or is_conv2d(m) or classname.find('BatchNorm') != -1:
            if not check_conv(m):
                continue
            a = m.lip()
            lipc += a
            all_lip.append(a)
            if linear and classname == 'Linear':
                m.update_grad()
            if conv and is_conv2d(m):
                m.update_grad()
            if bn and classname.find('BatchNorm') != -1:
                m.update_grad()
    return lipc, all_lip


def project(self, proj_to=10):
    lipc = 0
    all_lip = []
    for m in self.modules():
        classname = m.__class__.__name__
        if classname == 'Linear' or is_conv2d(m) or classname.find('BatchNorm') != -1:
            if not check_conv(m):
                continue
            a = m.lip()
            lipc += a
            all_lip.append(a)
            m.proj(proj_to)
    return lipc, all_lip


def project_bn(self, proj_to=5):
    lipc = 0
    all_lip = []
    for m in self.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            a = m.lip()
            lipc += a
            all_lip.append(a)
            m.proj(proj_to)
    return lipc, all_lip


def calc_lip(self):
    lipc = 0
    all_lip = []
    for m in self.modules():
        classname = m.__class__.__name__
        if classname == 'Linear' or is_conv2d(m) or classname.find('BatchNorm') != -1:
            if not check_conv(m):
                continue
            a = m.lip()
            lipc += a
            all_lip.append(a)
    return lipc, all_lip


def lip_param(self, mmt=MMT, alpha=ALPHA, beta=BETA, factor=None):
    for m in self.modules():
        classname = m.__class__.__name__
        if classname == 'Linear' or is_conv2d(m) or classname.find('BatchNorm') != -1:
            if factor is not None:
                if type(factor) is not tuple:
                    factor = (factor, factor, factor)
                m.mmt *= factor[0]
                m.alpha *= factor[1]
                m.beta *= factor[2]
            else:
                m.mmt = mmt
                m.alpha = alpha
                m.beta = beta


def add_lip(conv, norm, mmt=MMT, alpha=ALPHA, beta=BETA):
    # s = conv.stride
    # p = conv.padding
    # k = conv.kernel_size
    # d = conv.dilation

    conv.mod = None
    conv.mod_idx = None
    conv.chn_idx = None
    conv.norm = norm
    conv.mmt = mmt
    conv.alpha = alpha
    conv.beta = beta

    conv.grad_bound = torch.zeros_like(conv.weight, device=conv.weight.device)
    # conv.register_buffer('grad_bound', grad_bound)
    # conv.weight.grad_bound = conv.grad_bound  # for communication with optimizer

    conv._original_forward = conv.forward
    bind(conv, forward)

    func = [create_view, lip, lip_1norm, lip_1norm_all_idx, lip_1norm_single_idx, proj, update_grad]

    for fun in func:
        bind(conv, fun)


def check_size(sz, s, k, p):
    c = math.ceil(p/s)
    assert k + c * s - p <= sz


def forward(self, x):
    _, _, h, w = x.size()
    if self.mod is None and self.is_conv2d:
        self.h_w = (h, w)
        s = self.stride
        p = self.padding
        k = self.kernel_size
        try:
            check_size(h, s[0], k[0], p[0])
            check_size(w, s[1], k[1], p[1])
            # self.is_conv2d = True

            idx_set = partition(h, w, k, s, p)
            
            self.all_idx_in1mod = False
            self.single_idx_mod = False
            if len(idx_set) == 1 and len(idx_set[0]) == k[0] * k[1]:
                self.all_idx_in1mod = True
            if len(idx_set) == k[0] * k[1] == sum([len(c) for c in idx_set]):
                self.single_idx_mod = True

            if x.is_cuda:
                self.mod = [torch.cuda.LongTensor(idx) for idx in idx_set]
            else:
                self.mod = [torch.LongTensor(idx) for idx in idx_set]
            # for i, m in enumerate(self.mod):
            #     self.register_buffer(f'mod{i}', m)

            self.create_view()

            if self._verbose_:
                print(f'Added norm tracking to {self}')

        except AssertionError as e:
            print(f'Skip {self} because the input of size {(h, w)} cannot be covered by the kernel.')
            self.is_conv2d = False

    if self.is_conv2d:
        assert self.h_w == (h, w)  # make sure height and width of the input remain unchanged

    return self._original_forward(x)


def create_view(self):
    if (self.norm == "1-norm" and not self.all_idx_in1mod) or self.single_idx_mod:
        self.weight3d = self.weight.view(self.out_channels, self.in_channels, -1)
        self.grad_bound3d = self.grad_bound.view(self.out_channels, self.in_channels, -1)


@torch.no_grad()
def lip(self):
    
    if self.norm == "1-norm":
        if self.all_idx_in1mod:
            return self.lip_1norm_all_idx()
        elif self.single_idx_mod:
            return self.lip_1norm_single_idx()
        else:
            return self.lip_1norm()
    # elif self.norm == "2-norm":
    #     self.lip_bound = torch.sqrt(self.weight.square().sum() * self.n_out_el)
    elif self.norm == "inf-norm":
        self.lip_bound, self.lip_idx = torch.max(self.weight.abs().sum(dim=[1, 2, 3]), dim=0)
    return self.lip_bound.item()


def lip_1norm(self):
    a = torch.abs(self.weight3d)
    lipc = -1
    for mode in self.mod:
        tmp = a[:, :, mode]  # torch.index_select(a, dim=2, index=mode)
        tmp = torch.sum(tmp, dim=[0, 2])
        m, chn_idx = torch.max(tmp, dim=0)
        if m > lipc:
            lipc = m
            m_idx = mode
            c_idx = chn_idx
    
    if lipc == -1:  # error occurs, probably due to NaN in the weight
        self.lip_bound = lipc
        warnings.warn('An error occurs when computing norms, \
            probably due to NaN in the weight. May skip regularization.')
        return lipc

    self.mod_idx = m_idx
    self.chn_idx = c_idx  # index into in_channels
    self.lip_bound = lipc
    return lipc.item()


def lip_1norm_all_idx(self):
    a = torch.sum(self.weight.abs(), dim=[0, 2, 3])
    self.lip_bound, self.chn_idx = torch.max(a, dim=0)
    return self.lip_bound.item()


def lip_1norm_single_idx(self):
    _, in_c, kk = self.weight3d.size()
    a = torch.sum(self.weight3d.abs(), dim=0).flatten()  # torch tensor is row-major
    self.lip_bound, idx = torch.max(a, dim=0)
    self.chn_idx = math.floor(idx.item() / kk)
    self.mod_idx = idx.item() % kk
    return self.lip_bound.item()


@torch.no_grad()
def proj(self, proj_to):
    if self.lip_bound <= proj_to:
        return
    if self.norm == "1-norm":
        if self.all_idx_in1mod:
            y = self.weight[:, self.chn_idx]
            self.weight[:, self.chn_idx] = proj_l1_ball(y, proj_to)
        else:
            y = self.weight3d[:, self.chn_idx, self.mod_idx]
            self.weight3d[:, self.chn_idx, self.mod_idx] = proj_l1_ball(y, proj_to)
    else:
        y = self.weight[self.lip_idx, :, :, :]
        self.weight[self.lip_idx, :, :, :] = proj_l1_ball(y, proj_to)


@torch.no_grad()
def update_grad(self):

    if self.norm in ["1-norm",  "inf-norm"]:
        if self.lip_bound == -1:  # skip due to error
            return
        self.grad_bound *= self.mmt
    
    if self.norm == "1-norm":
        if self.all_idx_in1mod:
            g = self.grad_bound[:, self.chn_idx] 
            g.copy_(self.weight[:, self.chn_idx])
            g.sign_()
        else:
            self.grad_bound3d[:, self.chn_idx, self.mod_idx] = self.weight3d[:, self.chn_idx, self.mod_idx].sign_()
    # elif self.norm == "2-norm":
    #     scale = grad_wrt_self / self.lip_bound * self.n_out_el
    #     torch.mul(self.weight, scale, out=self.grad_bound)
    else:
        g = self.grad_bound[self.lip_idx, :, :, :] 
        g.copy_(self.weight[self.lip_idx, :, :, :])
        g.sign_()
    
    # weight_grad = self.weight.grad.data
    # coef = weight_grad.mul_(self.alpha).exp_().mul_(self.beta)

    self.weight.grad.data += self.grad_bound.mul(self.beta)


# The following functions are for Conv2d that acts like a linear map
'''
def add_lip_linear_conv(linear_conv, norm, mmt=MMT, alpha=ALPHA, beta=BETA):

    linear_conv.norm = norm
    linear_conv.mmt = mmt
    linear_conv.alpha = alpha
    linear_conv.beta = beta

    linear_conv.grad_bound = torch.zeros_like(linear_conv.weight, device=linear_conv.weight.device)
    # linear_conv.weight.grad_bound = linear_conv.grad_bound  # for communication with optimizer

    func = [create_view_linear_conv, lip_linear_conv, update_grad_linear_conv]
    as_name = ['create_view', 'lip', 'update_grad']

    for fun, name in zip(func, as_name):
        bind(conv, fun, name)


def create_view_linear_conv(self):
    if self.norm == "1-norm":
        self.g = self.grad_bound.view(-1)
        self.w = self.weight.view(-1)


def lip_linear_conv(self):
    if self.norm == "1-norm":
        self.lip_bound, self.lip_idx = torch.max(self.w.abs(), dim=0)
    # elif self.norm == "2-norm":
    #     tmp = torch.square(self.weight) 
    #     self.lip_bound = tmp.sum().sqrt()
    elif self.norm == "inf-norm":
        self.lip_bound = self.weight.abs().sum()
    return self.lip_bound


def update_grad_linear_conv(self, grad_wrt_self):
    if self.norm == "1-norm":
        self.grad_bound *= self.mmt

    if self.norm == "1-norm":
        self.g[self.lip_idx] = self.w[self.lip_idx].sign()
    # elif self.norm == "2-norm":
    #     scale = grad_wrt_self / self.lip_bound
    #     torch.mul(self.weight, scale, out=self.grad_bound)
    elif self.norm == "inf-norm":
        torch.sign(self.weight, out=self.grad_bound)

    # weight_grad = self.weight.grad.data
    # coef = weight_grad.mul_(self.alpha).exp_().mul_(self.beta)

    self.weight.grad.data += self.grad_bound.mul(self.beta)
'''

# The following functions are for linear maps
def add_lip_linear(linear, norm, mmt=MMT, alpha=ALPHA, beta=BETA):

    linear.norm = norm
    linear.mmt = mmt
    linear.alpha = alpha
    linear.beta = beta

    linear.grad_bound = torch.zeros_like(linear.weight, device=linear.weight.device)
    # linear.weight.grad_bound = linear.grad_bound  # for communication with optimizer

    func = [lip_linear, proj_linear, update_grad_linear]
    as_name = ['lip', 'proj', 'update_grad']

    for fun, name in zip(func, as_name):
        bind(linear, fun, name)


@torch.no_grad()
def lip_linear(self):
    if self.norm == "1-norm":
        col_sum = torch.sum(self.weight.abs(), dim=0)
        self.lip_bound, self.lip_idx = torch.max(col_sum, dim=0)
    # elif self.norm == "2-norm":
    #     tmp = torch.square(self.weight)
    #     self.lip_bound = tmp.sum().sqrt()
    elif self.norm == "inf-norm":
        row_sum = torch.sum(self.weight.abs(), dim=1)
        self.lip_bound, self.lip_idx = torch.max(row_sum, dim=0)
    return self.lip_bound.item()


@torch.no_grad()
def proj_linear(self, proj_to):
    if self.lip_bound <= proj_to:
        return
    if self.norm == "1-norm":
        y = self.weight[:, self.lip_idx]
        self.weight[:, self.lip_idx] = proj_l1_ball(y, proj_to)
    else:
        y = self.weight[self.lip_idx, :]
        self.weight[self.lip_idx, :] = proj_l1_ball(y, proj_to)


@torch.no_grad()
def update_grad_linear(self):
    if self.norm in ["1-norm",  "inf-norm"]:
        self.grad_bound *= self.mmt

    if self.norm == "1-norm":
        self.grad_bound[:, self.lip_idx] = self.weight[:, self.lip_idx]
        self.grad_bound[:, self.lip_idx].sign_()
    # elif self.norm == "2-norm":
    #     scale = grad_wrt_self / self.lip_bound
    #     torch.mul(self.weight, scale, out=self.grad_bound)
    else:
        self.grad_bound[self.lip_idx, :] = self.weight[self.lip_idx, :]
        self.grad_bound[self.lip_idx, :].sign_()

    self.weight.grad.data += self.grad_bound.mul(self.beta)


# The following functions are for batch normalization
def add_lip_bn(batch_norm, norm, mmt=MMT, alpha=ALPHA, beta=BETA):

    batch_norm.norm = norm
    batch_norm.mmt = mmt
    batch_norm.alpha = alpha
    batch_norm.beta = beta

    batch_norm.grad_bound = torch.zeros_like(batch_norm.weight, device=batch_norm.weight.device)
    # batch_norm.weight.grad_bound = batch_norm.grad_bound  # for communication with optimizer

    func = [lip_bn, proj_bn, update_grad_bn]
    as_name = ['lip', 'proj', 'update_grad']

    for fun, name in zip(func, as_name):
        bind(batch_norm, fun, name)


@torch.no_grad()
def lip_bn(self):
    tmp = self.weight / torch.sqrt(self.running_var + self.eps)
    self.lip_bound, self.lip_idx = torch.max(tmp.abs(), dim=0)
    return self.lip_bound.item()


@torch.no_grad()
def proj_bn(self, proj_to):
    if self.lip_bound > proj_to:
        coeff = torch.sqrt(self.running_var[self.lip_idx] + self.eps)
        self.weight[self.lip_idx] = proj_to * coeff * torch.sign(self.weight[self.lip_idx])


@torch.no_grad()
def update_grad_bn(self):
    self.grad_bound *= self.mmt
    coeff = torch.sqrt(self.running_var[self.lip_idx] + self.eps)
    self.grad_bound[self.lip_idx] = torch.sign(self.weight[self.lip_idx]) / coeff


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


def is_conv2d(layer):
    if hasattr(layer, 'is_conv2d'):
        return layer.is_conv2d
    if hasattr(layer, 'weight') and hasattr(layer, 'stride') and \
       hasattr(layer, 'padding') and hasattr(layer, 'kernel_size'):
        if layer.weight.dim() == 4:
            try:
                # assert type(layer) is torch.nn.Conv2d
                d = layer.dilation
                assert d[0] == d[1] == 1
                assert layer.padding_mode == 'zeros'
                assert layer.groups == 1
                layer.is_conv2d = True
                return True
            except (AssertionError, AttributeError) as e:
                info = [
                    ('dilation == 1', d[0] == d[1] == 1, d[0], d[1]),
                    ('padding_mode == zeros', layer.padding_mode == 'zeros', layer.padding_mode),
                    ('groups == 1', layer.groups == 1, layer.groups)
                ]
                err = [m for m in info if not m[1]]
                print("Skip a Conv2d because {0}".format(err))
                layer.is_conv2d = False
                return False
    layer.is_conv2d = False
    return False

def check_conv(m):
    if hasattr(m, 'mod') and m.mod is None:
        warnings.warn(
            'The network structure seems to change during training. '
            'The returned `all_lip` may be inconsistent across different calls.', 
            RuntimeWarning
        )
        return False
    return True


if __name__ == '__main__':
    import torchvision
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets, transforms as T

    device = torch.device('cuda:0')

    resnet = torchvision.models.resnet18(pretrained=True).to(device)

    bind_lip(resnet)
    print(next(resnet.parameters()).shape)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = datasets.ImageNet("L:/ImageNet/", split="train", download=False, transform=transform)
    trainloader = DataLoader(dataset, batch_size=64, num_workers=2)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)

    resnet.train()

    # out = resnet(images)

    # resnet.config_lip()
    
    print(images.shape)
    print(labels.shape)

    optimizer = optim.Adam(resnet.parameters())

    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()

    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)

    out = resnet(images)
    print(out.shape)

    loss = criterion(out, labels)
    loss.backward()

    weight = next(resnet.parameters())
    grad0 = torch.zeros_like(weight)
    grad0.copy_(weight.grad)

    resnet.add_lip_grad()

    idx_ = grad0 != weight.grad

    print(weight[idx_].flatten()[:10])
    print(grad0[idx_].flatten()[:10])
    print(weight.grad[idx_].flatten()[:10])

