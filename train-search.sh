CUDA_VISIBLE_DEVICES=2 python train_search.py \
 --batch-size 24 --dataset hps-GID --checkname 12layers_forward --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 256

