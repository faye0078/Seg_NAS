CUDA_VISIBLE_DEVICES=2 python retrain_nas.py \
 --batch-size 28 --dataset GID --checkname 12layers_retrain --num_worker 4\
 --alpha_epoch 20 --resize 512 --crop_size 512 \
 --epochs 100  --use_default True\
 --resume /media/dell/DATA/wy/Seg_NAS/run/GID/12layers_retrain/model_best.pth.tar