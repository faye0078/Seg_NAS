import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict

def get_connections_2(core_path):
    connections = []
    connections.append([[-1, 0], [0, 0]])
    for i in range(len(core_path) - 1):
        connections.append([[i, core_path[i]], [i + 1, core_path[i + 1]]])
    return connections


class Decoder_2(object):
    def __init__(self, alphas, core_path, cell_arch):
        self.alphas = torch.from_numpy(alphas)
        self._num_layers = self.alphas.shape[0]
        self.cell_space = cell_arch
        self.core_path = core_path

    def get_n_arch(self):
        for i in range(self._num_layers):
            for j in range(4):
                if self.core_path[i] == j:
                    num = int(self.cell_space[i][j].sum() + 2)
                    if num > 11:
                        num = 11
                    self.cell_space[i][j][self.alphas[i][j].sort()[1][-num:]] = 1
        return self.cell_space
