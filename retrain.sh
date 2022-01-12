CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 128 --dataset GID --checkname '12layers_retrain/my_path' --num_worker 4\
 --epochs 200 --model_name 'flexinet' --nclass 5