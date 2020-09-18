import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from models import *
from lip.add_lip import bind_lip


nets = {
    'vgg': VGG,
    'regnet': RegNetX_200MF,
    'resnet': ResNet18,
    'preact_resnet': PreActResNet18,
    'googlenet': GoogLeNet,
    'densenet': DenseNet121,
    'resnetxt': ResNeXt29_2x64d,
    'mobilenet': MobileNet,
    'mobilenet2': MobileNetV2,
    'dpn': DPN92,
    'shefflenet': ShuffleNetG2,
    'senet': SENet18,
    'shefflenet2': ShuffleNetV2,
    'efficientnet': EfficientNetB0
}


def get_module_name(net):
    bind_lip(net)
    net.eval()
    x = torch.rand(1, 3, 32, 32)
    net.cpu()
    net(x)
    module_name = []
    n_conv = 0
    n_linear = 0
    n_bn = 0
    for m in net.modules():
        classname = m.__class__.__name__
        if hasattr(m, 'is_conv2d') and m.is_conv2d:
            module_name.append((f'{classname}-#{n_conv}', f'{m}'))
            n_conv += 1
        elif classname == 'Linear':
            module_name.append((f'{classname}-#{n_linear}', f'{m}'))
            n_linear += 1
        elif classname.find('BatchNorm') != -1:
            module_name.append((f'{classname}-#{n_bn}', f'{m}'))
            n_bn += 1

    return module_name


def get_last_lips(file):
    with open(file, 'rb') as f:
        record = pickle.load(f)
    return record.lip[-1][1]  # the last record is (epoch, lips)


def plot_bar(ax, start, x, y, width, label=None):
    rects = ax.bar(x + start, y, width=width, align='center', label=label)
    autolabel(rects, ax)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        offset = -1  # 5 points vertical offset
        rep = height
        va = 'bottom'
        if height < 0:
            offset = 1
            rep = -height
            va = 'top'
        ax.annotate(f'{rep:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha='center', va=va, 
                    fontsize=15)


def plot(y, xtick, x_labels, width, title):
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Norm', fontsize=15)
    # forward, inverse = get_scale(a=1.5)
    # ax.set_yscale('function', functions=(forward, inverse))
    ax.set_title(title)
    ax.set_xticks(xtick)
    ax.set_xticklabels(x_labels, fontsize=18)

    plot_bar(ax, 0, xtick, y, width)

    plt.tight_layout()
    plt.savefig(f"./{img_dir}/{title}.png")

    plt.close('all')  # to save memory


def compare_norm(files, method_names, model_name):
    record = [get_last_lips(file) for file in files]
    lens = [len(x) for x in record]
    for l in lens:
        assert l == lens[0]

    n_layers = len(record[0])
    n_records = len(files)
    xtick = np.arange(n_records)
    width = 0.7

    for i in range(n_layers):
        layer_norms = [t[i] for t in record]
        module_name = module_names[model_name][i][0]
        plot(layer_norms, xtick, method_names, width, f'{_models[model_name]}-{module_name}')


def main():
    for model in models:
        file_names = [f'{log_path}/{model}/{model}_{method}_record.pkl' for method in methods]
        found_files = []
        got_methods = []
        for i, file in enumerate(file_names):
            if os.path.exists(file):
                found_files.append(file)
                got_methods.append(_methods[methods[i]])
        if len(found_files) == 0:
            continue
        compare_norm(found_files, got_methods, model)


if __name__ == '__main__':
    log_path = './AT/log'
    img_dir = './img_compare_norm'  # path to save the produced images
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    methods = ['plain', 'alp', 'trades', 'pgd']
    _methods = {'plain':'Plain', 'alp':'ALP', 'trades':'TRADES', 'pgd':'PGD-AT'}
    models = ['vgg', 'resnet', 'senet', 'regnet']
    _models = {'vgg':'VGG', 'resnet':'ResNet', 'senet':'SENet', 'regnet':'RegNet'}
    module_names = {model:get_module_name(nets[model]()) for model in models}
    
    main()
