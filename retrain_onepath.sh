CUDA_VISIBLE_DEVICES=2 python retrain_nas.py \
 --batch-size 10 --dataset GID --checkname 12layers_retrain --num_worker 4\
 --alpha_epoch 10 --resize 512 --crop_size 512 \
 --epochs 100  --use_default False\
