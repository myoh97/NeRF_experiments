CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/lego.txt --N_rand 512 --rgb rgb --expname lego/rgb_512;
CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/lego.txt --N_rand 512 --rgb rbg --expname lego/rbg_512;
CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/lego.txt --N_rand 512 --rgb bgr --expname lego/bgr_512;
CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/lego.txt --N_rand 512 --rgb brg --expname lego/brg_512;
CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/lego.txt --N_rand 512 --rgb grb --expname lego/grb_512;
CUDA_VISIBLE_DEVICES=9 python run_nerf.py --N_iter 100000 --config configs/lego.txt --N_rand 512 --rgb gbr --expname lego/gbr_512;