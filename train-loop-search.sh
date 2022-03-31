CUDA_VISIBLE_DEVICES=3 python train_loop_search.py \
 --batch-size 22 --dataset GID --checkname Ablation/max_path --num_worker 4\
 --alpha_epoch 0 --filter_multiplier 8 --resize 512 --crop_size 512\
 --resume '/media/dell/DATA/wy/Seg_NAS/run/GID/Ablation/max_path/experiment_0/2_stage1_epoch10_checkpoint.pth.tar'