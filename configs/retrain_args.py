import argparse
import numpy as np

def obtain_retrain_args():
    parser = argparse.ArgumentParser(description="ReTrain the nas model")

    parser.add_argument('--use_default', type=bool, default=True,  help='if use the default arch')
    # parser.add_argument('--train', action='store_true', default=True, help='training mode')
    # parser.add_argument('--exp', type=str, default='bnlr7e-3', help='name of experiment')
    # parser.add_argument('--gpu', type=str, default='0', help='test time gpu device id')
    # parser.add_argument('--backbone', type=str, default='autodeeplab', help='resnet101')
    # parser.add_argument('--dataset', type=str, default='cityscapes', help='pascal or cityscapes')
    # parser.add_argument('--groups', type=int, default=None, help='num of groups for group normalization')
    # parser.add_argument('--epochs', type=int, default=4000, help='num of training epochs')
    # parser.add_argument('--batch_size', type=int, default=14, help='batch size')
    # parser.add_argument('--base_lr', type=float, default=0.05, help='base learning rate')
    # parser.add_argument('--warmup_start_lr', type=float, default=5e-6, help='warm up learning rate')
    # parser.add_argument('--lr-step', type=float, default=None)
    # parser.add_argument('--warmup-iters', type=int, default=1000)
    # parser.add_argument('--min-lr', type=float, default=None)
    # parser.add_argument('--last_mult', type=float, default=1.0, help='learning rate multiplier for last layers')
    # parser.add_argument('--scratch', action='store_true', default=False, help='train from scratch')
    # parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze batch normalization parameters')
    # parser.add_argument('--weight_std', action='store_true', default=False, help='weight standardization')
    # parser.add_argument('--beta', action='store_true', default=False, help='resnet101 beta')
    # parser.add_argument('--crop_size', type=int, default=769, help='image crop size')
    # parser.add_argument('--resize', type=int, default=769, help='image crop size')
    # parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    # parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    # parser.add_argument('--filter_multiplier', type=int, default=32)
    # parser.add_argument('--dist', type=bool, default=False)
    # parser.add_argument('--autodeeplab', type=str, default='train')
    # parser.add_argument('--block_multiplier', type=int, default=5)
    # parser.add_argument('--use-ABN', default=True, type=bool, help='whether use ABN')
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    # parser.add_argument('--port', default=6000, type=int)
    # parser.add_argument('--max-iteration', default=1000000, type=bool)
    # parser.add_argument('--net_arch', default=None, type=str)
    # parser.add_argument('--cell_arch', default=None, type=str)
    # parser.add_argument('--criterion', default='Ohem', type=str)
    parser.add_argument('--initial-fm', default=None, type=int)
    # parser.add_argument('--mode', default='poly', type=str, help='how lr decline')
    # parser.add_argument('--local_rank', dest='local_rank', type=int, default=-1, )
    # parser.add_argument('--train_mode', type=str, default='iter', choices=['iter', 'epoch'])

    parser.add_argument('--net_arch', type=str, default='/media/dell/DATA/wy/Seg_NAS/run/GID/path.npy')
    parser.add_argument('--cell_arch', type=str, default='/media/dell/DATA/wy/Seg_NAS/run/GID/cell.npy')

    parser.add_argument('--opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'], help='opt level for half percision training (default: O0)')

    parser.add_argument('--dataset', type=str, default='GID', choices=['pascal', 'coco', 'cityscapes', 'kd', 'GID'], help='dataset name (default: pascal)')
    parser.add_argument('--nas', type=str, default='retrain', choices=['search', 'retrain'])

    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=320, help='base image size')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--resize', type=int, default=512, help='resize image size')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--use_ABN', type=bool, default=False, help='whether to use abn (default: False)')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
    NORMALISE_PARAMS = [
        1.0 / 255,  # SCALE
        np.array([0.485, 0.456, 0.406, 0.411]).reshape((1, 1, 4)),  # MEAN
        np.array([0.229, 0.224, 0.225, 0.227]).reshape((1, 1, 4)),  # STD
    ]
    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS, help="Normalisation parameters [scale, mean, std],")
    parser.add_argument('--nclass', type=int, default=5,help='number of class')

    parser.add_argument("--dist", type=bool, default=False)
    # training hyper params
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--filter_multiplier', type=int, default=32)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--alpha_epoch', type=int, default=20,metavar='N', help='epoch to start training alphas')

    parser.add_argument('--num_worker', type=int, default=4,metavar='N', help='numer workers')
    parser.add_argument('--batch-size', type=int, default=3, metavar='N', help='input batch size for training (default: auto)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR', help='learning rate for alpha and beta in architect searching process')
    parser.add_argument('--lr-scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed and logging
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='12layers_retrain', help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--val', action='store_true', default=True, help='skip validation during training')

    args = parser.parse_args()
    return args
