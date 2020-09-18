import pickle
import os
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


def filter_(s, boolean):
    x = []
    for t, b in zip(s, boolean):
        if b:
            x.append(t)
    return x


def main():
    for model in models:
        module_name = get_module_name(nets[model]())
        bn = [True if x.find('BatchNorm') != -1 else False for x, y in module_name]
        non_bn = [not x for x in bn]
        path = f'./regularization/log/{model}'
        for file in os.listdir(path):
            if not file.endswith('record.pkl'):
                continue
            if file.find('proj_bn') == -1:
                continue
            
            with open(os.path.join(path, file), 'rb') as f:
                recorder = pickle.load(f)
            try:
                lip = recorder.lip

                recorder.save_path = f'{path}/bn'
                new_lip = []
                for e, s in lip:
                    s = filter_(s, bn)
                    new_lip.append((e, s))
                recorder.lip = new_lip
                recorder.draw_many('lip')

                recorder.save_path = f'{path}/conv_linear'
                new_lip = []
                for e, s in lip:
                    s = filter_(s, non_bn)
                    new_lip.append((e, s))
                recorder.lip = new_lip
                recorder.draw_many('lip')
            except Exception as e:
                print('Error dealing with ', dirname, file, '\n', e, '\n')
            

if __name__ == '__main__':
    models = ['vgg', 'resnet', 'senet', 'regnet']
    main()
