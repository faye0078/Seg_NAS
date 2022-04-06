CUDA_VISIBLE_DEVICES=2 python retrain_nas.py \
 --batch-size 12 --dataset uadataset --checkname 'retrain/loop_2' --resize 512 --crop_size 512 --num_worker 4\
 --epochs 10 --model_name 'flexinet' --nclass 12