from collections import OrderedDict
def get_data_path(dataset):
    if dataset == 'hps-GID':
        Path = OrderedDict()
        Path['dir'] = "../data/512/"
        Path['train_list'] = "./data/lists/hps_train.lst"
        Path['val_list'] = "./data/lists/hps_val.lst"
        Path['test_list'] = "./data/lists/hps_val.lst"
        Path['mini_train_list'] = "./data/lists/mini_hps_train.lst"
        Path['mini_val_list'] = "./data/lists/mini_hps_val.lst"
    if dataset == 'GID':
        Path = OrderedDict()
        Path['dir'] = "../data/512/"
        Path['train_list'] = "./data/lists/rs_train.lst"
        Path['val_list'] = "./data/lists/rs_val.lst"
        Path['test_list'] = "./data/lists/rs_test.lst"
        Path['mini_train_list'] = "./data/lists/mini_hps_train.lst"
        Path['mini_val_list'] = "./data/lists/mini_hps_val.lst"
    elif dataset == 'cityscapes':
        Path = '/media/dell/DATA/wy/data/cityscapes'

    return Path