CUDA_VISIBLE_DEVICES=8 python train.py --model fc_in --exp fc_in;
CUDA_VISIBLE_DEVICES=8 python train.py --model fc_bn_drop --exp fc_bn_drop;
CUDA_VISIBLE_DEVICES=8 python train.py --model fc_in_drop --exp fc_in_drop;
