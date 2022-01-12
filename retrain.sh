CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 22 --dataset GID --checkname '12layers_retrain/deeplabv3plus' --num_worker 4\
 --epochs 200 --model_name 'deeplabv3plus' --nclass 5