CUDA_VISIBLE_DEVICES=0 python train_search.py \
 --batch-size 6 --dataset GID --checkname 14layers_third_rs --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 512\
 --model_name FlexiNet --search_stage third --model_encode_path '/media/dell/DATA/wy/Seg_NAS/model/model_encode/GID-5/14layers_mixedcell1_3operation/third_connect_4.npy'\
 --resume '/media/dell/DATA/wy/Seg_NAS/run/GID/14layers_third_rs/experiment_0/epoch9_checkpoint.pth.tar'