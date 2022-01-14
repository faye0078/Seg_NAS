import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict
class Decoder(object):
    def __init__(self, alphas, active_node):
        self.active_node = active_node
        self.alphas = torch.from_numpy(alphas)
        self._num_layers = self.alphas.shape[0]
        self.cell_space = torch.zeros(self._num_layers, 4, 10)

        for i in range(self._num_layers):
            for j in range(4):
                self.cell_space[i][j][self.alphas[i][j].sort()[1][-3:]] = 1


if __name__ == '__main__':
    path = '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_third_batch8_Mixed/experiment_0/alphas/'
    alphas_list = OrderedDict()
    cell_list = OrderedDict()

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            # if filename.split('.')[0].split('_')[0] == 'alphas':
            if filename.split('.')[0].split('_')[0] == 'betas':
                alphas_list[filename] = np.load(dirpath + filename)
                # active_node = np.load() # TODO: The decoder
                active_node = [[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [4, 1],
                               [5, 0], [6, 0], [6, 1], [7, 0], [8, 1], [9, 1], [9, 2],
                               [10, 2], [11, 1], [11, 2]]
                decoder = Decoder(alphas_list[filename], active_node)
                cell_list[filename] = decoder.cell_space

    order_cell_list = []
    for i in range(len(cell_list)):
        # idx = 'alphas_{}.npy'.format(str(i))
        idx = 'betas_{}.npy'.format(str(i))
        order_cell_list.append(cell_list[idx])
    # print(path_list)
    print(cell_list)
    b = np.array(cell_list['betas_59.npy'])

    np.save(path + 'cell_operations.npy', b)
    # print(b)


