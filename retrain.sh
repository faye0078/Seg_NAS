CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 8 --dataset GID --checkname 'resolution/512' --resize 512 --crop_size 512 --num_worker 4\
 --epochs 100 --model_name 'fast-nas' --nclass 5