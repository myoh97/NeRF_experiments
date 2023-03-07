CUDA_VISIBLE_DEVICES=7 python train.py --model base --exp furniture_pop_views --iter 700 --pop_layer views
CUDA_VISIBLE_DEVICES=7 python train.py --model base --exp furniture_pop_feature --iter 700 --pop_layer feature
CUDA_VISIBLE_DEVICES=7 python train.py --model base --exp furniture_pop_weight --iter 700 --pop_layer weight
