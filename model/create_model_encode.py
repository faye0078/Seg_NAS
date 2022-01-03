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

    return np.array(connections)


def core_path_connect(layer, depth, path):
    connections = []
    for i in range(layer):
        # connections.append([[i, path[i]], [i + 1, path[i + 1]]])
        for j in range(path[i]+1):
            if j == path[i]:
                connections.append([[-1, 0], [i, j]])
                for k in range(i):
                    for l in range(path[k]+1):
                        connections.append([[k, l], [i, j]])
                continue
            if j == 0:
                if path[i-1] == 0:
                    connections.append([[i - 1, 0], [i, 0]])
                else:
                    connections.append([[i - 1, 0], [i, 0]])
                    connections.append([[i - 1, 1], [i, 0]])

            if j == 1:
                if path[i-1] == 0:
                    connections.append([[i - 1, 0], [i, 1]])
                elif path[i-1] == 1:
                    connections.append([[i - 1, 0], [i, 1]])
                    connections.append([[i - 1, 1], [i, 1]])
                elif path[i-1] == 2:
                    connections.append([[i - 1, 0], [i, 1]])
                    connections.append([[i - 1, 1], [i, 1]])
                    connections.append([[i - 1, 2], [i, 1]])

            if j == 2:
                if path[i-1] == 1:
                    connections.append([[i - 1, 1], [i, 2]])
                elif path[i-1] == 2:
                    connections.append([[i - 1, 1], [i, 2]])
                    connections.append([[i - 1, 2], [i, 2]])
                elif path[i-1] == 3:
                    connections.append([[i - 1, 1], [i, 2]])
                    connections.append([[i - 1, 2], [i, 2]])
                    connections.append([[i - 1, 3], [i, 2]])


    return connections


def test_connections(connections):
    for connection in connections:
        if connections.count(connection) != 1:
            return False

    return True


# TODO: def 更严格的test函数（判断是否有通路）
# TODO: def 通过第二阶段的搜索结果建立相应的connections连接


if __name__ == "__main__":
    # connections = normal_connect_4(11)
    # path = np.load('/media/dell/DATA/wy/Seg_NAS/run/hps-GID/12layers_flexinet/path.npy')
    path = [0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 2, 2]
    connections = core_path_connect(12, 4, path)
    if test_connections(connections):
        np.save('./model_encode/complex_connect_4.npy', connections)  # 保存为.npy格式

