import os
import math
import numpy as np
import matplotlib.pyplot as plt


class Recorder(object):
    """Recorder can be used to record training and testing statistics"""
    def __init__(self, name, save_path):
        super(Recorder, self).__init__()
        self.name = name
        self.save_path = save_path
        self.epoch = 1

    def record(self, name, stat):
        if hasattr(self, name):
            attr = getattr(self, name)
            attr.append((self.epoch, stat))
        else:
            setattr(self, name, [(self.epoch, stat)])

    def get_record(self, name, n_per_epoch=5):
        attr = getattr(self, name)
        
        epoches = []
        stat = []
        n = len(attr)
        step = math.ceil(n / self.epoch / n_per_epoch)  # get 5 records per epoch
        intv = 1 / (n / self.epoch)

        for i, (e, s) in enumerate(attr):
            if i % step == 0:
                epoches.append(i * intv)
                stat.append(s)
        
        return epoches, stat

    def draw(self, name, fig=None, figsize=(8,5), fontsize=20, yscale=None):
        epoches, stat = self.get_record(name)
        if fig is None:
            plt.figure(figsize=figsize)
        if yscale is not None:
            plt.yscale(yscale)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.plot(epoches, stat)
        # plt.legend(['alexnet'])
        plt.xlabel("Epoches", fontsize=fontsize)
        plt.ylabel(name, fontsize=fontsize)
        plt.tight_layout()
        # plt.show()
        save_path = self.save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        plt.savefig(f'{save_path}/{self.name}_{name}.jpg', dpi=300)
        plt.close('all')

    def draw_many(self, name, fig=None, figsize=(8,5), fontsize=20, yscale=None):
        epoches, stat = self.get_record(name)
        if fig is None:
            plt.figure(figsize=figsize)
        if yscale is not None:
            plt.yscale(yscale)

        y = np.array(stat)

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.plot(epoches, y)
        # plt.legend(['alexnet'])
        plt.xlabel("Epoches", fontsize=fontsize)
        if name == 'lip':
            plt.ylabel('Norm', fontsize=fontsize)
        else:
            plt.ylabel(name, fontsize=fontsize)
        plt.tight_layout()
        # plt.show()
        save_path = self.save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        filename = f'{save_path}/{self.name}_{name}.jpg'
        while os.path.exists(filename):
            filename = filename[:-4] + '_.jpg'
        plt.savefig(filename, dpi=300)
        plt.close('all')

    def step(self):
        self.epoch += 1

    def __len__(self):
        return self.epoch
