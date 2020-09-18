import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import warnings

warnings.filterwarnings("ignore")


def get_last_lips(file):
    with open(file, 'rb') as f:
        record = pickle.load(f)
    return record.lip[-1][1]  # the last record is (epoch, lips)


def get_acc(file):
    with open(file, 'rb') as f:
        record = pickle.load(f)
    test_acc = record.test_acc[-1][1] * 100
    robust_accuracy = record.robust_accuracy[-1][1] * 100
    return test_acc, robust_accuracy


def compare_norm(files, method_names, model_name):
    record = [get_last_lips(file) for file in files]
    lens = [len(x) for x in record]
    for l in lens:
        assert l == lens[0]

    n_layers = len(record[0])
    n_records = len(files)

    fig, ax = plt.subplots()

    for i in range(n_records):
        method_name = method_names[i]
        clean_acc, robust_acc = get_acc(files[i])
        label = _methods[method_name] + f' ({clean_acc:.0f}% | {robust_acc:.0f}%)'
        # Draw the density plot
        sns.distplot(record[i], hist = False, kde = True, rug = True,
                     kde_kws = {'linewidth': 2},
                     label = label)

    plt.legend(prop={'size': 16}, title = 'Training method')
    title = r'Distribution of $\ell_1$ norm in ' + f'{_models[model_name]} Layers'
    plt.title(title, fontsize='14')
    plt.xlabel(r'$\ell_1$ norm', fontsize='14')
    plt.ylabel('Density', fontsize='14')

    plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title

    plt.savefig(f"./{img_dir}/{model_name}.png", dpi=300)
    plt.tight_layout()

    plt.close('all')


def main():
    for model in models:
        file_names = [f'{log_path}/{model}/{model}_{method}_record.pkl' for method in methods]
        found_files = []
        got_methods = []
        for i, file in enumerate(file_names):
            if os.path.exists(file):
                found_files.append(file)
                got_methods.append(methods[i])
        if len(found_files) == 0:
            continue
        compare_norm(found_files, got_methods, model)


if __name__ == '__main__':
    log_path = './AT/log'
    img_dir = './img_den'  # path to save the produced images
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    methods = ['plain', 'alp', 'trades', 'pgd']
    _methods = {'plain':'plain', 'alp':'ALP', 'trades':'TRADES', 'pgd':'PGD-AT'}
    models = ['vgg', 'resnet', 'senet', 'regnet']
    _models = {'vgg':'VGG', 'resnet':'ResNet', 'senet':'SENet', 'regnet':'RegNet'}
    
    main()
