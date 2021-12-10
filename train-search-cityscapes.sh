CUDA_VISIBLE_DEVICES=2 python train_search.py \
 --batch-size 3 --dataset cityscapes --checkname 12layers_forward --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 321 \
 --resume /media/dell/DATA/wy/Seg_NAS/run/cityscapes/12layers_forward/model_best.pth.tar