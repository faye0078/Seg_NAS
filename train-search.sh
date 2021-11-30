CUDA_VISIBLE_DEVICES=2 python train_search.py \
 --batch-size 4 --dataset GID --checkname 12layers --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 321 \
