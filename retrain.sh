CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 7 --dataset GID --checkname '12layers_forward_retrain/pspnet' --num_worker 4\
 --epochs 200 --model_name 'pspnet' --nclass 5\
 --net_arch '/media/dell/DATA/wy/Seg_NAS/run/hps-GID/12layers_forward/path.npy'\
 --cell_arch '/media/dell/DATA/wy/Seg_NAS/run/hps-GID/12layers_forward/cell.npy'