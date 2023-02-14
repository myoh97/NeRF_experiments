CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/ficus.txt --N_rand 512 --rgb rgb --expname fix/ficus/rgb;
CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/drums.txt --N_rand 512 --rgb brg --expname fix/drums/brg;
