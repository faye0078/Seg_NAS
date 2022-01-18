CUDA_VISIBLE_DEVICES=2 python train_search.py \
 --batch-size 24 --dataset GID-15 --checkname 12layers_first_cell --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 512\
 --model_name FlexiNet --search_stage first --model_encode_path '/media/dell/DATA/wy/Seg_NAS/model/model_encode/first_connect_4.npy'\
 --resume '/media/dell/DATA/wy/Seg_NAS/run/GID-15/12layers_first_cell/experiment_0/epoch38_checkpoint.pth.tar'