CUDA_VISIBLE_DEVICES=3 python train_search.py \
 --batch-size 2 --dataset GID --checkname 1024/14layers_third --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 1024 --crop_size 1024\
 --model_name FlexiNet --search_stage third --model_encode_path '/media/dell/DATA/wy/Seg_NAS/model/model_encode/third_connect_4.npy'