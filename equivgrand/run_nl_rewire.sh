# 100 splits, rewiring using gdc (need to check again, results not good :<)
# rewiring can leads to out of memory, need to be fixed :<
# need to check cmd arguments 

# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Cora --num_splits 100 --rewiring gdc
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Citeseer  --num_splits 100 --rewiring gdc
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Photo  --num_splits 100 --rewiring gdc
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset CoauthorCS  --num_splits 100 --rewiring gdc
