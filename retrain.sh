<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=2 python retrain_nas.py \
=======
CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
>>>>>>> 886e97f8171dc6e284e089013da92bb5eae13b99
 --batch-size 8 --dataset GID --checkname 12layers_onepath_retrain --num_worker 4\
 --alpha_epoch 20 --resize 512 --crop_size 512 \
 --epochs 70  --use_default True\
