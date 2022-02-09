CUDA_VISIBLE_DEVICES=3 python train_search.py \
 --batch-size 4 --dataset GID --checkname 1024/14layers_second --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 1024 --crop_size 1024\
 --model_name FlexiNet --search_stage second --model_encode_path '/media/dell/DATA/wy/Seg_NAS/model/model_encode/second_connect_4.npy'