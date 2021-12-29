import numpy as np

def normal_connect_4(layer):
    connections = []
    connections.append([[-1, 0], [0, 0]])

    for i in range(layer):
        if i == 0:
            connections.append([[0, 0], [1, 0]])
            connections.append([[0, 0], [1, 1]])

        elif i == 1:
            connections.append([[1, 0], [2, 0]])
            connections.append([[1, 0], [2, 1]])
            connections.append([[1, 1], [2, 0]])
            connections.append([[1, 1], [2, 1]])
            connections.append([[1, 1], [2, 2]])

        elif i == 2:
            connections.append([[2, 0], [3, 0]])
            connections.append([[2, 0], [3, 1]])
            connections.append([[2, 1], [3, 0]])
            connections.append([[2, 1], [3, 1]])
            connections.append([[2, 1], [3, 2]])
            connections.append([[2, 2], [3, 1]])
            connections.append([[2, 2], [3, 2]])
            connections.append([[2, 2], [3, 3]])

        else:
            connections.append([[i, 0], [i + 1, 0]])
            connections.append([[i, 0], [i + 1, 1]])
            connections.append([[i, 1], [i + 1, 0]])
            connections.append([[i, 1], [i + 1, 1]])
            connections.append([[i, 1], [i + 1, 2]])
            connections.append([[i, 2], [i + 1, 1]])
            connections.append([[i, 2], [i + 1, 2]])
            connections.append([[i, 2], [i + 1, 3]])
            connections.append([[i, 3], [i + 1, 2]])
            connections.append([[i, 3], [i + 1, 3]])

    connections = np.array(connections)
    np.save('./normal_connect_4.npy', connections)  # 保存为.npy格式

def core_path_connect(layer, depth, path):
    connections = []
    connections.append([[-1, 0], [0, 0]])
    for i in range(layer):
        connections.append([[i, path[i]], [i + 1, path[i + 1]]])


if __name__ == "__main__":
    path = np.load('/media/dell/DATA/wy/Seg_NAS/run/hps-GID/12layers_flexinet/path.npy')
    core_path_connect(11, 4, path)
    # normal_connect_4(11)
