# run 100 random data splits, 20 random inits each split
# error when running ogbn data, needs to be fixed :<

# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset CoauthorCS  --num_splits 100
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Photo  --num_splits 100
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Citeseer  --num_splits 100
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Cora --num_splits 100
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset ogbn-arxiv --num_splits 100