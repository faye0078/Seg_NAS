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

    # trainer.start_epoch = 0 #暂时先设置为0，需要读取保存过的模型
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
        # if epoch % 10 ==0:
            
        #      state_dict = trainer.model.state_dict()
        #      state = {
        #         'epoch': epoch + 1,
        #         'state_dict': state_dict,
        #         'optimizer':  trainer.optimizer.state_dict(),
        #         'best_pred':  trainer.best_pred,
        #     }
        #      filename = '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_default_retrain/epoch_{}.pth.tar'.format(epoch)
        #      torch.save(state, filename)



if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
