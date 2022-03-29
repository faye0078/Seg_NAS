CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 32 --dataset GID --checkname 'one_loop/retrain' --resize 512 --crop_size 512 --num_worker 4\
 --epochs 200 --model_name 'flexinet' --nclass 5