CUDA_VISIBLE_DEVICES=2 python retrain_nas.py \
 --batch-size 12 --dataset hps-GID --checkname '12layers_flexinet_retrain/56_nmp' --num_worker 4\
 --epochs 200 --model_name 'flexinet' --nclass 5\
 --net_arch '/media/dell/DATA/wy/Seg_NAS/run/hps-GID/12layers_forward/path.npy'\
 --cell_arch '/media/dell/DATA/wy/Seg_NAS/run/hps-GID/12layers_forward/cell.npy'