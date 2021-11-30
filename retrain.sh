CUDA_VISIBLE_DEVICES=2 python retrain_nas.py \
 --batch-size 28 --dataset GID --checkname '12layers_onepath_retrain' --num_worker 4\
 --epochs 70
