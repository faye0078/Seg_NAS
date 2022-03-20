CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 7 --dataset GID --checkname '1024/14layers_retrain/flexinet' --resize 1024 --crop_size 1024 --num_worker 4\
 --epochs 200 --model_name 'flexinet' --nclass 5