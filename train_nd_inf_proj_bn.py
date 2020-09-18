from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import *
import numpy as np
import time

from lip.add_lip import bind_lip
from lip.recorder import Recorder
import pickle

from auto_attack.autoattack import AutoAttack

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

models = [key for key, value in nets.items()]


def write2text(path, filename, name, **kwargs):
    with open(os.path.join(path, f'{filename}.txt'), 'a') as f:
        f.write(f'{name}: {kwargs}\n')


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out, target)

        loss.backward()

        train_loss += loss.item() * target.size(0)
        _, predicted = out.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if name.find('0_0') != -1:
            lipc, all_lip = model.calc_lip()
        else:
            lipc, all_lip = model.add_lip_grad(linear=linear, conv=conv, bn=bn)
            model.project_bn(proj_to=5)

        recorder.record('lip_sum', lipc)
        recorder.record('lip', all_lip)

        optimizer.step()
    
    train_loss, train_acc = train_loss / total, correct / total
    recorder.record('train_acc', train_acc)
    recorder.record('train_loss', train_loss)

    print('Training: loss: {:.4f}, Acc: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)), end=' | ')


def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss, test_acc = test_loss / total, correct / total
    recorder.record('test_acc', test_acc)
    recorder.record('test_loss', test_loss)

    print('Test: loss: {:.4f}, Acc: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    model = nets[args.model]().to(device)

    bind_lip(model, norm='inf-norm', mmt=mmt, beta=beta, verbose=False)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)   
    
    natural_acc = []
    robust_acc = []
    
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch: {epoch:3d}', end='  ')
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        train(args, model, device, train_loader, optimizer, epoch)

        test(model, test_loader)

        recorder.step()

        if epoch == args.epochs:
            adversary = AutoAttack(model, norm='Linf', eps=1/255, version='standard')
            adversary.attacks_to_run = ['apgd-ce', 'apgd-t']

            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    x_adv, robust_accuracy = adversary.run_standard_evaluation(inputs, targets, bs=128)
                    recorder.record('robust_accuracy', robust_accuracy)
                    break
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f'{name}.pt'))

    used_time = (time.time() - start_time) / 3600
    print(f'Used time: {used_time:.2f} h')

    with open(f'{log_dir}/{name}_record.pkl', 'wb') as file:
        pickle.dump(recorder, file)

    # recorder.draw('lip_sum')
    recorder.draw_many('lip')
    recorder.draw('train_acc')
    recorder.draw('test_acc')

    clean_acc = recorder.test_acc[-1][1]
    write2text(log_dir, 'log', name, clean_acc=clean_acc, robust_accuracy=robust_accuracy)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=0.,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', default='resnet',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--test-size', default=128*8, type=int, metavar='N',
                        help='test size for evaluating robustness')

    args = parser.parse_args()

    # settings
    model_dir = f'./regularization/checkpoint/{args.model}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    log_dir = f'./regularization/log/{args.model}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    img_dir = f'./regularization/img/{args.model}'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    torch.backends.cudnn.benchmark = True

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_size, shuffle=True, **kwargs)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, **kwargs)

    loss_fn = nn.CrossEntropyLoss()

    linear, conv, bn = True, True, False  # not to regularize BN, since it has only little effect on BN norms

    mmt = 0.5
    betas = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    for beta in betas:
        name = f'{args.model}_inf_nd_proj_bn_{beta}'
        print(f'Using {name}')

        recorder = Recorder(name, img_dir)
        main()
        print()
