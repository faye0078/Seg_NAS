CUDA_VISIBLE_DEVICES=0 python retrain_nas.py \
 --batch-size 12 --dataset uadataset --checkname 'test' --resize 512 --crop_size 512 --num_worker 8\
 --epochs 200 --model_name 'flexinet' --nclass 12\
 --resume '/media/dell/DATA/wy/Seg_NAS/run/uadataset/retrain/loop_2/experiment_1/epoch53_checkpoint.pth.tar'