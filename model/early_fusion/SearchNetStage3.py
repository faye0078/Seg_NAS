import numpy as np
import torch
from torch import nn
import numpy
from model.cell import ReLUConvBN, Fusion
import torch.nn.functional as F


class SearchNet3(nn.Module):

    def __init__(self, layers, depth, connections, cell, dataset, num_classes, base_multiplier=40):
        '''
        Args:
            layers: layer × depth： one or zero, one means ture
            depth: the model scale depth
            connections: the node connections
            cell: cell type
            dataset: dataset
            base_multiplier: base scale multiplier
        '''
        super(SearchNet3, self).__init__()
        self.block_multiplier = 1
        self.base_multiplier = base_multiplier
        self.depth = depth
        self.layers = layers
        self.connections = np.array(connections)
        self.cell_connect = cell(512, 512).ops_num
        self.node_add_num = np.zeros([len(layers), self.depth])

        half_base = int(base_multiplier // 2)
        if dataset == 'GID' or dataset == 'hps-GID':
            input_channel = 4
        else:
            input_channel = 3
        self.stem0 = nn.Sequential(
            nn.Conv2d(input_channel, half_base * self.block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_base * self.block_multiplier, half_base * self.block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_base * self.block_multiplier, self.base_multiplier * self.block_multiplier, 3, stride=2,
                      padding=1),
            nn.BatchNorm2d(self.base_multiplier * self.block_multiplier),
            nn.ReLU()
        )
        self.cells = nn.ModuleList()
        self.fusions = nn.ModuleList()
        multi_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        num_last_features = 0
        for i in range(len(self.layers)):
            self.cells.append(nn.ModuleList())
            self.fusions.append(nn.ModuleList())
            for j in range(self.depth):
                self.fusions[i].append(nn.ModuleDict())
                self.cells[i].append(cell(self.base_multiplier * multi_dict[j],
                                                                        self.base_multiplier * multi_dict[j]))
                num_connect = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        num_connect += 1
                        if connection[0][0] == -1:
                            self.fusions[i][j][str(connection[0])] = Fusion(self.base_multiplier * multi_dict[0],
                                                                        self.base_multiplier * multi_dict[
                                                                            connection[1][1]])
                        else:
                            self.fusions[i][j][str(connection[0])] = Fusion(
                                self.base_multiplier * multi_dict[connection[0][1]],
                                self.base_multiplier * multi_dict[connection[1][1]])
                self.node_add_num[i][j] = num_connect

                if i == len(self.layers) - 1 and num_connect != 0:
                    num_last_features += self.base_multiplier * multi_dict[j]

        self.last_conv = nn.Sequential(
            nn.Conv2d(num_last_features, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        print('connections number: \n' + str(self.node_add_num))
        self.initialize_alphas()

    def forward(self, x):
        features = []

        temp = self.stem0(x)
        temp = self.stem1(temp)
        pre_feature = self.stem2(temp)

        normalized_alphas = torch.randn(len(self.layers), self.depth, self.cell_connect).cuda()

        for i in range(len(self.layers)):
            for j in range(self.depth):
                normalized_alphas[i][j] = F.softmax(self.alphas[i][j], dim=-1)

        for i in range(len(self.layers)):
            features.append([])
            for j in range(self.depth):
                features[i].append(0)
                k = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        if connection[0][0] == -1:
                            features[i][j] += self.fusions[i][j][str(connection[0])](pre_feature)
                        else:
                            if isinstance(features[connection[0][0]][connection[0][1]], int):
                                continue
                            features[i][j] += self.fusions[i][j][str(connection[0])](
                                features[connection[0][0]][connection[0][1]])
                        k += 1

                if not isinstance(features[i][j], int):
                    features[i][j] = self.cells[i][j](features[i][j], normalized_alphas[i][j])

        last_features = [feature for feature in features[len(self.layers) - 1] if torch.is_tensor(feature)]
        last_features = [nn.Upsample(size=last_features[0].size()[2:], mode='bilinear', align_corners=True)(feature) for
                         feature in last_features]
        result = torch.cat(last_features, dim=1)
        result = self.last_conv(result)
        result = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)(result)
        return result

    def initialize_alphas(self):
        alphas = (1e-3 * torch.randn(len(self.layers), self.depth, self.cell_connect)).clone().detach().requires_grad_(
            True)

        self._arch_parameters = [
            alphas,
        ]
        self._arch_param_names = [
            'alphas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in
         zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if
                name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if
                name not in self._arch_param_names and 'filter' not in name]
