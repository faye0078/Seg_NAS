CUDA_VISIBLE_DEVICES=2 python train_search.py \
 --batch-size 24 --dataset GID --checkname 12layers_flexinet_alldata_first_batch24_cell5 --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 512\
 --model_name FlexiNet --search_stage first --model_encode_path '/media/dell/DATA/wy/Seg_NAS/model/model_encode/normal_connect_4.npy'