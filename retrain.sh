CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 7 --dataset GID --checkname '12layers_retrain/pspnet' --num_worker 4\
 --epochs 200 --model_name 'pspnet' --nclass 5\
 --resume '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_retrain/pspnet/experiment_1/epoch75_checkpoint.pth.tar'