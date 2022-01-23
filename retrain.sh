CUDA_VISIBLE_DEVICES=2 python retrain_nas.py \
 --batch-size 9 --dataset GID --checkname '12layers_retrain/unet' --num_worker 4\
 --epochs 200 --model_name 'unet' --nclass 5\
 --resume '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_retrain/unet/experiment_0/epoch17_checkpoint.pth.tar'