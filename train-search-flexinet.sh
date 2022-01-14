CUDA_VISIBLE_DEVICES=2 python train_search.py \
 --batch-size 8 --dataset GID-15 --checkname 12layers_third_batch8_Mixed --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 512\
 --model_name FlexiNet --search_stage third --model_encode_path '/media/dell/DATA/wy/Seg_NAS/model/model_encode/third_connect_4.npy'