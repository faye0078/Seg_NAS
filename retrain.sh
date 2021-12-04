CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 4 --dataset GID --checkname '12layers_forward_retrain/one_path' --num_worker 4\
 --epochs 100 --model_name 'one_path'\
 --net_arch '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_forward/path.npy'\
 --cell_arch '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_forward/cell.npy'\
