from dataloaders.datasets import GID
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("../")
from configs.Path import get_data_path

from dataloaders.datasets.GID import (
    CentralCrop,
    Normalise,
    RandomCrop,
    RandomMirror,
    ResizeScale,
    ToTensor,
)
from torchvision import transforms
def make_data_loader(args, **kwargs):
    if args.dataset == 'GID':
        data_path = get_data_path('GID')
        num_class = 5
        composed_trn = transforms.Compose(
            [
                RandomMirror(),
                RandomCrop(args.crop_size),
                Normalise(*args.normalise_params),
                ToTensor(),
            ]
        )
        composed_val = transforms.Compose(
            [

                CentralCrop(args.crop_size),
                Normalise(*args.normalise_params),
                ToTensor(),
            ]
        )
        composed_test = transforms.Compose(
            [
                CentralCrop(args.crop_size),
                Normalise(*args.normalise_params),
                ToTensor(),
            ])

        if args.nas == 'search':
            train_set = GID.GIDDataset(data_file=data_path['mini_train_list'],
                                        data_dir=data_path['dir'],
                                        transform_trn=composed_trn,
                                        transform_val=composed_val,)
            val_set = GID.GIDDataset(data_file=data_path['mini_val_list'],
                                        data_dir=data_path['dir'],
                                        transform_trn=composed_trn,
                                        transform_val=composed_val,)
        elif args.nas == 'retrain':
            train_set = GID.GIDDataset(data_file=data_path['train_list'],
                                        data_dir=data_path['dir'],
                                        transform_trn=composed_trn,
                                        transform_val=composed_val,)
            val_set = GID.GIDDataset(data_file=data_path['val_list'],
                                        data_dir=data_path['dir'],
                                        transform_trn=composed_trn,
                                        transform_val=composed_val,)
            test_set = GID.GIDDataset(data_file=data_path['test_list'],
                                        data_dir=data_path['dir'],
                                        transform_val=composed_test,)
        # elif args.nas == 'retrain':
        #     train_set = GID.GIDDataset(data_file=data_path['mini_train_list'],
        #                                 data_dir=data_path['dir'],
        #                                 transform_trn=composed_trn,
        #                                 transform_val=composed_val,)
        #     val_set = GID.GIDDataset(data_file=data_path['mini_val_list'],
        #                                 data_dir=data_path['dir'],
        #                                 transform_trn=composed_trn,
        #                                 transform_val=composed_val,)
        else:
            raise Exception('nas param not set properly')

        n_examples = len(train_set)
        n_train = int(n_examples/2)
        train_set1, train_set2 = random_split(train_set, [n_train, n_examples - n_train])

        print(" Created train setB = {} examples, train setB = {}, val set = {} examples".format(len(train_set1), len(train_set2), len(val_set)))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
        train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        if args.nas == 'retrain':
            # a = 1
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        if args.nas == 'search':
            return train_loader1, train_loader2, val_loader, num_class
        elif args.nas == 'retrain':
            return train_loader, val_loader, test_loader, num_class
            # return train_loader, val_loader, num_class

    elif args.dataset == 'cityscapes':
        if args.autodeeplab == 'search':
            train_set1, train_set2 = cityscapes.twoTrainSeg(args)
            num_class = train_set1.NUM_CLASSES
            train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, num_worker=args.num_worker, shuffle=True, **kwargs)
            train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, num_worker=args.num_worker, shuffle=True, **kwargs)
        elif args.autodeeplab == 'train':
            train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_worker=args.num_worker,  shuffle=True, **kwargs)
        else:
            raise Exception('nas param not set properly')

        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_worker=args.num_worker, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_worker=args.num_worker, shuffle=False, **kwargs)

        if args.autodeeplab == 'search':
            return train_loader1, train_loader2, val_loader, num_class
        elif args.autodeeplab == 'train':
            return train_loader, val_loader, test_loader, num_class


    else:
        raise NotImplementedError
