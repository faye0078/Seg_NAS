CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 8 --dataset GID --checkname 12layers_default_retrain --num_worker 4\
 --alpha_epoch 20 --resize 512 --crop_size 512 \
 --epochs 100  --use_default True\
