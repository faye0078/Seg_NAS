CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 12 --dataset uadataset --checkname 'retrain/loop_2' --resize 512 --crop_size 512 --num_worker 4\
 --epochs 200 --model_name 'flexinet' --nclass 12\
 --resume '/media/dell/DATA/wy/dfc2022/run/dfc2022/train/model_best.pth.tar'