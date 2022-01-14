import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict
class Decoder(object):
    def __init__(self, alphas, active_node):
        self.active_node = active_node
        self.alphas = alphas
        self._num_layers = self._betas.shape[0]
        self.network_space = torch.zeros(self._num_layers, 4, 3)

        used_betas = []
        for i in range(self._num_layers):
            used_betas.append([])
            for j in range(self.core_path[i] + 1):
                used_betas[i].append([])
                if j < self.core_path[i]:
                    if j == 0:
                        if core_path[i-1] == 0:
                            used_betas[i][j] = self._betas[i][j][:1]
                        else:
                            used_betas[i][j] = self._betas[i][j][:2]
                    elif j == 1:
                        if core_path[i-1] == 1:
                            used_betas[i][j] = self._betas[i][j][:2]
                        else:
                            used_betas[i][j] = self._betas[i][j][:3]
                    elif j == 2 :
                        if core_path[i-1] == 2:
                            used_betas[i][j] = self._betas[i][j][:2]
                        else:
                            used_betas[i][j] = self._betas[i][j][:3]
                else:
                    used_betas[i][j] = self._betas[i][j][:core_path_num[i]]

        for i in range(len(used_betas)):
            for j in range(len(used_betas[i])):
                for k in range(len(used_betas[i][j])):
                    if used_betas[i][j][k] > 0:
                        used_betas[i][j][k] = 1
                    else:
                        used_betas[i][j][k] = 0

        self.used_betas = used_betas

if __name__ == '__main__':
    path = '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_second_batch24/experiment_1/alphas/'
    alphas_list = OrderedDict()
    used_alphas_list = OrderedDict()

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[0].split('_')[0] == 'alphas':
                alphas_list[filename] = np.load(dirpath + filename)
                active_node = np.load() # TODO: The decoder
                decoder = Decoder(alphas_list[filename], active_node)
                used_alphas_list[filename] = decoder.used_betas

    order_path_list = []
    for i in range(len(used_alphas_list)):
        idx = 'alphas_{}.npy'.format(str(i))
        order_path_list.append(used_alphas_list[idx])
    # print(path_list)
    print(used_alphas_list)
    # b = np.array(path_list['betas_52.npy'])

    # np.save(path + 'path.npy', b)
    # print(b)

