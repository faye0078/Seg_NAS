CUDA_VISIBLE_DEVICES=3 python train_loop_search.py \
 --batch-size 6 --dataset GID --checkname one_loop/search --num_worker 4\
 --alpha_epoch 10 --filter_multiplier 8 --resize 512 --crop_size 512