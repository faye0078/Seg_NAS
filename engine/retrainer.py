import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
from collections import OrderedDict

import sys
sys.path.append("./apex")

import sys
sys.path.append("..")

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False
from search.loss import SegmentationLosses
from dataloaders import make_data_loader
# from decoder import Decoder
from search.lr_scheduler import LR_Scheduler
from retrain.saver import Saver
# from utils.summaries import TensorboardSummary
from search.evaluator import Evaluator
from retrain.model_onepath import Retrain_Autodeeplab as Onepath_Autodeeplab
from search.copy_state_dict import copy_state_dict

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # 可视化
        # self.summary = TensorboardSummary(self.saver.experiment_dir)
        # self.writer = self.summary.create_summary()
        # 使用amp
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        # 定义dataloader
        kwargs = {'num_workers': args.num_worker, 'pin_memory': True, 'drop_last':True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)


        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

        torch.cuda.empty_cache()
        # 定义网络
        model = Onepath_Autodeeplab(args)
        optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, 1000, min_lr=args.min_lr)

        if args.cuda:
            self.model = self.model.cuda()

        # 使用apex支持混合精度分布式训练
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            # fix for current pytorch version with opt_level 'O1'
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        # Hack to fix BN fprop without affine transformation
                        if module.weight is None:
                            module.weight = torch.nn.Parameter(
                                torch.ones(module.running_var.shape, dtype=module.running_var.dtype,
                                           device=module.running_var.device), requires_grad=False)
                        if module.bias is None:
                            module.bias = torch.nn.Parameter(
                                torch.zeros(module.running_var.shape, dtype=module.running_var.dtype,
                                            device=module.running_var.device), requires_grad=False)

            # print(keep_batchnorm_fp32)
            self.model, [self.optimizer] = amp.initialize(
                self.model, [self.optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')

        # 加载模型
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'],  strict=False)


            copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


    def training(self, epoch):
        train_loss = 0.0
        # try:
        #     self.train_loaderA.dataset.set_stage("train")
        # except AttributeError:
        #     self.train_loaderA.dataset.dataset.set_stage("train")  # for subset
        self.model.train()
        tbar = tqdm(self.train_loader)

        for i, sample in enumerate(tbar):
            image = sample["image"]
            target = sample["mask"]
            # image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda().float(), target.cuda().float()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if not self.args.val:
            # save checkpoint every epoch
            is_best = False
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

