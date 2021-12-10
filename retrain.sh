CUDA_VISIBLE_DEVICES=3 python retrain_nas.py \
 --batch-size 4 --dataset cityscapes --checkname '12layers_forward_retrain/multi' --num_worker 4\
 --epochs 200 --model_name 'multi' --nclass 19\
 --net_arch '/media/dell/DATA/wy/Seg_NAS/run/cityscapes/12layers_forward/path.npy'\
 --cell_arch '/media/dell/DATA/wy/Seg_NAS/run/cityscapes/12layers_forward/cell.npy'