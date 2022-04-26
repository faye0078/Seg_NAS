CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 12 --dataset uadataset --checkname 'retrain/no_rs' --resize 512 --crop_size 512 --num_worker 8\
 --epochs 200 --model_name 'flexinet' --nclass 12