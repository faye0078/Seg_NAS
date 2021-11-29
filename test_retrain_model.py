import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from configs.retrain_args import obtain_retrain_args
from engine.retrainer import Trainer


# 为每个卷积层搜索最适合它的卷积实现算法
# torch.backends.cudnn.benchmark=True

def main():
    args = obtain_retrain_args()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    print('Total Epoches:', trainer.args.epochs)

    trainer.validation(epoch)



if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
