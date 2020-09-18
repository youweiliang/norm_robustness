import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FuncFormatter


def get_tex(y):
    s = [f'{i*100:.1f}' for i in y]
    return ' & '.join(s)


def get_acc(model):
    all_clean_acc = []
    all_robust_acc = []
    file = f'./regularization/log/{model}/log.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    record = []
    plain = None
    for i in range(len(lines)):
        # lines[i] = lines[i].replace('0_0', 'plain')
        if lines[i].find('plain') != -1:
            plain = lines[i]
            continue
        record.append(lines[i].replace('1e-05', '0.00001'))
    lines = sorted(record, reverse=True)
    if plain is not None:
        lines.insert(0, plain)

    for i in range(len(lines)):
        tmp = re.split(': |, |}', lines[i])
        if i % n_regularization_param == 1:
            t = tmp[0].split('_')[:-1]
            t = '_'.join(t)
            print(t, end=' | ')
        if tmp[0].find('plain') != -1:
            print(tmp[0], end=' | ')
        clean_acc = float(tmp[-4])
        robust_acc = float(tmp[-2])
        all_clean_acc.append(clean_acc)
        all_robust_acc.append(robust_acc)
    
    print()
    print(model, 'clean: ', end=' ')
    for i in range(len(all_clean_acc)):
        print(f'{100*all_clean_acc[i]:.1f}', end=' ')
        if i % n_regularization_param == 0:
            print('|', end=' ')
    print()
    print(model, 'robust: ', end=' ')
    for i in range(len(all_robust_acc)):
        print(f'{100*all_robust_acc[i]:.1f}', end=' ')
        if i % n_regularization_param == 0:
            print('|', end=' ')
    print()
    
    return all_clean_acc, all_robust_acc


def write_tex(model):
    all_clean_acc, all_robust_acc = get_acc(model)

    clean_acc, robust_acc = 'Clean & ' + get_tex(all_clean_acc), 'Robust & ' + get_tex(all_robust_acc)
    # print(clean_acc, end=' ')
    # print(r'\\')
    # print(robust_acc, end=' ')
    # print(r'\\')

    with open('./regularization/acc_tex_table.txt', 'w') as f:
        f.write(clean_acc + r' \\' + '\n')
        f.write(robust_acc + r' \\' + '\n')


def main():
    
    for model in models:
        try:
            write_tex(model)
        except FileNotFoundError:
            continue
        

if __name__ == '__main__':

    models = ['vgg', 'resnet', 'senet', 'regnet']

    n_regularization_param = 5  # of regularization parameters tested in each methods
    main()
