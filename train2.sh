CUDA_VISIBLE_DEVICES=8 python run_nerf.py --N_iter 100000 --config configs/chair.txt --N_rand 512 --rgb rbg --expname fix/chair/rbg;
CUDA_VISIBLE_DEVICES=8 python run_nerf.py --N_iter 100000 --config configs/ficus.txt --N_rand 512 --rgb gbr --expname fix/ficus/gbr;
