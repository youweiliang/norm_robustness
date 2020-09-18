from __future__ import print_function
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

# from models.wideresnet import *
from models import *
from losses import alp_loss, pgd_loss, trades_loss

from lip.add_lip import bind_lip
from lip.recorder import Recorder

from auto_attack.autoattack import AutoAttack


def train(args, model, device, train_loader, optimizer, recorder, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = LOSS[args.loss](model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           loss=args.loss,
                           distance=args.distance,
                           m = args.m,
                           s = args.s)
        loss.backward()

        lipc, all_lip = model.add_lip_grad()

        recorder.record('lip_sum', lipc)
        recorder.record('lip', all_lip)

        optimizer.step()

        # print progress
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader, recorder):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: loss: {:.4f}, Acc: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)), end=' | ')
    training_accuracy = correct / len(train_loader.dataset)

    recorder.record('train_acc', training_accuracy)
    recorder.record_train(train_loss)

    return train_loss, training_accuracy


def eval_test(model, device, test_loader, recorder):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: loss: {:.4f}, Acc: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_accuracy = correct / len(test_loader.dataset)

    recorder.record('test_acc', test_accuracy)
    recorder.record_test(test_loss)

    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    if args.loss == 'alp' or args.loss == 'trades' or args.loss == 'pgd':
        print("normalize False")
        model = nets[args.model]().to(device)
    else:
        print("normalize True")
        model = nets[args.model](use_FNandWN=True).to(device)
    
    bind_lip(model, norm='1-norm', beta=args.norm_decay, verbose=False)

    recorder = Recorder(f'{name}')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        print(f'Epoch: {epoch:3d}', end='  ')
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, recorder, epoch)

        # evaluation on natural examples
        # print('==============')
        eval_train(model, device, train_loader, recorder)
        eval_test(model, device, test_loader, recorder)
        # print('==============')

        # save checkpoint
        if (epoch >= args.start_freq) and (epoch % args.save_freq == 0):
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f'{name}-epoch{epoch}.pt'))

        recorder.step()

    torch.save(model.state_dict(), os.path.join(model_dir,  f'{name}.pt'))

    with open(f'{log_dir}/{name}_record.pkl', 'wb') as file:
        pickle.dump(recorder, file)

    recorder.draw('lip_sum')
    recorder.draw_many('lip')

    recorder.draw('train_acc')
    recorder.draw('test_acc')

    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', verbose=False)
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.max(), inputs.min())

            x_adv, robust_accuracy = adversary.run_standard_evaluation(inputs, targets, bs=128)
            print(f'robust_accuracy: {robust_accuracy}')
            break

    recorder.record('robust_accuracy', robust_accuracy)

    with open(f'{log_dir}/{name}_record.pkl', 'wb') as file:
        pickle.dump(recorder, file)

if __name__ == '__main__':
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


    parser = argparse.ArgumentParser(description='Adversarial Training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--norm-decay', '--nd', default=0.,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epsilon', default=0.031,
                        help='perturbation')
    parser.add_argument('--num-steps', default=10,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.007,
                        help='perturb step size')
    parser.add_argument('--beta', type = float, default=1.0)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--snap-epoch', type=int, default=5, metavar='N',
                        help='how many batches to test')                    
    parser.add_argument('--model', default='resnet',  type=str, 
                        choices=models, help='model to use')
    parser.add_argument('--save-freq', default=10, type=int, metavar='N',
                        help='save frequency')
    parser.add_argument('--start-freq', default=1, type=int, metavar='N',
                        help='start point')
    parser.add_argument('--loss', default='pgd', type=str, 
                        choices=['pgd', 'pgd_he', 'alp', 'alp_he', 'trades', 'trades_he'])
    parser.add_argument('--distance', default='l_inf', type=str, help='distance')
    parser.add_argument('--m', default=0.2, type=float, help='angular margin')
    parser.add_argument('--s', default=15.0, type=float, help='s value')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id to use')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model_dir = "checkpoint_nd//" + args.loss
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = './log_nd'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128*8, shuffle=True, **kwargs)
    LOSS= {
            'pgd': pgd_loss,
            'pgd_he': pgd_loss,
            'alp': alp_loss,
            'alp_he': alp_loss,
            'trades': trades_loss,
            'trades_he': trades_loss,
    }

    for los in ['alp']:
        args.loss = los
        name = f'{args.model}_{args.loss}_nd'
        print(f'Using {name}')
        main()
