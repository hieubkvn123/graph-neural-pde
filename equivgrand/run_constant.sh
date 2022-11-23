# run_GNN.py: 1 data split, 20 inits
# run_reproduce.py: specify number of split with --num_splits, 20 inits for each split
# bugs when run with datasets: PubMed, Computer

# run GRAND model with best params (linear) - can run with 5 datasets
CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Cora --block constant 
CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Citeseer --block constant 
CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Photo --block constant 
CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset CoauthorCS --block constant 
CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset ogbn-arxiv --block constant 

# run GRAND model with best params (nonlinear) - can run with 5 datasets
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Cora
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Citeseer
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Photo 
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset CoauthorCS 
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset ogbn-arxiv

# run GRAND model with best params (nonlinear, rewire)
# run with rewire can leads to out of memory, need to be fixed! ()
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Cora --rewiring gdc 
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Citeseer --rewiring gdc 
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Photo  --rewiring gdc 
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset CoauthorCS  --rewiring gdc 
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset ogbn-arxiv --rewiring gdc 


# run GRAND model with explicit block and function
# CUDA_VISIBLE_DEVICES=1 python run_GNN.py --dataset Cora --block attention --function laplacian


# 100 splits, 20 init (as reported in the paper) - unknown error when running ogbn-arxiv data :( 
CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Cora --block constant --num_splits 100
CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Citeseer --block constant --num_splits 100
CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset Photo --block constant --num_splits 100
CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset CoauthorCS --block constant --num_splits 100
# CUDA_VISIBLE_DEVICES=1 python run_reproduce.py --dataset ogbn-arxiv --block constant --num_splits 100


# to run EquivGRAND model (experimenting)
# CUDA_VISIBLE_DEVICES=5 python run_GNN.py --dataset Cora --no_early --block equiv_attention --function equiv_laplacian --equiv 