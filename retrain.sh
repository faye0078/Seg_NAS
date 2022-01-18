CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 24 --dataset GID --checkname '12layers_retrain/flexinet_aspp' --num_worker 4\
 --epochs 200 --model_name 'flexinet' --nclass 5