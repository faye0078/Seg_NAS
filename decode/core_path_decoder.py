import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict
class Decoder(object):
    def __init__(self, betas, core_path, core_path_num):
        self.core_path = core_path
        self._betas = betas
        self._num_layers = self._betas.shape[0]
        self.network_space = torch.zeros(self._num_layers, 4, 3)

        used_betas = []
        for i in range(self._num_layers):
            used_betas.append([])
            for j in range(self.core_path[i] + 1):
                used_betas[i].append([])
                if j < self.core_path[i]:
                    if j == 0:
                        used_betas[i][j].append(self._betas[i][j][:2])
                    elif j == 1:
                        used_betas[i][j].append(self._betas[i][j][:3])
                    elif j == 2 :
                        used_betas[i][j].append(self._betas[i][j][:3])
                else:
                    used_betas[i][j].append(self._betas[i][j][:core_path_num[i]])

        for i in range(len(used_betas)):
            for j in range(len(used_betas[i])):
                for k in range(len(used_betas[i][j][0])):
                    if used_betas[i][j][0].shape[0] == 0:
                        used_betas[i][j][k] = []
                    else:
                        if used_betas[i][j][0][k] > 0:
                            used_betas[i][j][0][k] = 1
                        else:
                            used_betas[i][j][0][k] = 0

        self.used_betas = used_betas

if __name__ == '__main__':
    path = '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_flexinet_alldata_second_batch24/experiment_1/betas/'
    betas_list = OrderedDict()
    used_betas_list = OrderedDict()

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[0].split('_')[0] == 'betas':
                betas_list[filename] = np.load(dirpath + filename)
                core_path = [0, 0, 1, 1, 1, 0, 1, 0, 1, 2, 2, 2]
                core_path_num = [0, 1, 2, 4, 6, 8, 9, 11, 12, 14, 17, 20]
                decoder = Decoder(betas_list[filename], core_path, core_path_num)
                used_betas_list[filename] = decoder.used_betas

    order_path_list = []
    for i in range(len(used_betas_list)):
        idx = 'betas_{}.npy'.format(str(i))
        order_path_list.append(used_betas_list[idx])
    # print(path_list)
    print(used_betas_list)
    # b = np.array(path_list['betas_52.npy'])

    # np.save(path + 'path.npy', b)
    # print(b)


