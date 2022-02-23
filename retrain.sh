CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 50 --dataset GID --checkname 'resolution/256' --resize 256 --crop_size 256 --num_worker 4\
 --epochs 100 --model_name 'fast-nas' --nclass 5