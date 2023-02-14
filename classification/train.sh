CUDA_VISIBLE_DEVICES=9 python train.py --model fc --exp fc;
CUDA_VISIBLE_DEVICES=9 python train.py --model fc_drop --exp fc_drop;
CUDA_VISIBLE_DEVICES=9 python train.py --model fc_bn --exp fc_bn;
