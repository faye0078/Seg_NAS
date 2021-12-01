CUDA_VISIBLE_DEVICES=0 python train_search.py \
 --batch-size 4 --dataset GID --checkname 12layers_forward --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 321 --crop_size 256 \
