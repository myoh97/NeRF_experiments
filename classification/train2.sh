CUDA_VISIBLE_DEVICES=8 python train.py --model base --exp furniture_pop_rgb --iter 700 --pop_layer rgb
CUDA_VISIBLE_DEVICES=8 python train.py --model base --exp furniture_pop_alpha --iter 700 --pop_layer alpha
CUDA_VISIBLE_DEVICES=8 python train.py --model base --exp furniture_pop_bias --iter 700 --pop_layer bias
