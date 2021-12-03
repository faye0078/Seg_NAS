CUDA_VISIBLE_DEVICES=0 python train_search.py \
 --batch-size 5 --dataset GID --checkname 12layers_forward --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 256\
 --resume /media/dell/DATA/wy/Seg_NAS/run/GID/12layers_forward/model_best.pth.tar
