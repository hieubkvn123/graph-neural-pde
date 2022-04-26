# I. Running with 10 fixed splits + 20 random seeds
# 1. On GRAND baseline
python3 run_GNN.py --function laplacian \
                       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'GRAND baseline - 10 fixed splits + 20 seeds' \
		       --num_random_seeds 20 \
		       --geom_gcn_splits

# 2. Test run on best bounds for each alpha value
python3 run_GNN.py --function laplacian \
		       --block attention   \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
              	       --time 128.0 
		       --max_nfe 100000000 \
                       --run_name 'Alpha=1.0;bound=0.05 - 10 fixed splits + 20 seeds' \
		       --geom_gcn_splits \
		       --num_random_seeds 20 \
                       --alpha_ 1.0 \
                       --clip_bound 0.05  

python3 run_GNN.py --function laplacian \
         	       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'Alpha=2.0;bound=0.25 - 10 fixed splits + 20 seeds' \
		       --geom_gcn_splits \
		       --num_random_seeds 20 \
                       --alpha_ 1.0 \
                       --clip_bound 0.25  

python3 run_GNN.py --function laplacian \
                       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'Alpha=3.0;bound=0.4 - 10 fixed splits + 20 seeds' \
		       --geom_gcn_splits \
		       --num_random_seeds 20 \
                       --alpha_ 3.0 \
                       --clip_bound 0.4  

python3 run_GNN.py --function laplacian \
                       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'Alpha=4.0;bound=0.6 - 10 fixed splits + 20 seeds' \
		       --geom_gcn_splits \
		       --num_random_seeds 20 \
                       --alpha_ 4.0 \
                       --clip_bound 0.6

# II. Running with 10 random splits + 10 random seeds 
# 1. On GRAND baseline
python3 run_GNN.py --function laplacian \
                       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'GRAND baseline - 10 random splits + 10 seeds' \
		       --num_random_seeds 10 \
		       --num_splits 10

# 2. Test run on best bounds for each alpha value
python3 run_GNN.py --function laplacian \
		       --block attention   \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
              	       --time 128.0 
		       --max_nfe 100000000 \
                       --run_name 'Alpha=1.0;bound=0.05 - 10 random splits + 10 seeds' \
		       --num_splits 10 \
		       --num_random_seeds 10 \
                       --alpha_ 1.0 \
                       --clip_bound 0.05  

python3 run_GNN.py --function laplacian \
         	       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'Alpha=2.0;bound=0.25 - 10 random splits + 10 seeds' \
		       --num_splits 10 \
		       --num_random_seeds 10 \
                       --alpha_ 1.0 \
                       --clip_bound 0.25  

python3 run_GNN.py --function laplacian \
                       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'Alpha=3.0;bound=0.4 - 10 random splits + 10 seeds' \
		       --num_splits 10 \
		       --num_random_seeds 10 \
                       --alpha_ 3.0 \
                       --clip_bound 0.4  

python3 run_GNN.py --function laplacian \
                       --block attention \
                       --experiment \
                       --lr 0.0000001 \
                       --max_iters 1000 \
                       --time 128.0 \
                       --max_nfe 100000000 \
                       --run_name 'Alpha=4.0;bound=0.6 - 10 random splits + 10 seeds' \
		       --num_splits 10 \
		       --num_random_seeds 10 \
                       --alpha_ 4.0 \
                       --clip_bound 0.6
