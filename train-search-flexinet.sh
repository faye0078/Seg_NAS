CUDA_VISIBLE_DEVICES=1 python train_search.py \
 --batch-size 5 --dataset hps-GID --checkname 12layers_flexinet_alldata_second_test --num_worker 4\
 --alpha_epoch 20 --filter_multiplier 8 --resize 512 --crop_size 512\
  --model_name FlexiNet --search_stage second