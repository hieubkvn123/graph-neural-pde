import os
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--function", required=False, default="ext_laplacian3", help="The extended laplacian function to use")
parser.add_argument("--clip_low", required=False, type = float, default=0.05, help="Lower bound for clipping value")
parser.add_argument("--clip_high", required=False, type = float,  default=1.0005, help="Upper bound for clipping value")
parser.add_argument("--clip_step", required=False, default=0.1, help="Step size for clipping values")
args = vars(parser.parse_args())

alphas = [4.0, 3.0, 2.0, 1.0]
k_values = [3,4,5,6,7,8]
t_values = [128.0, 64.0, 32.0]
bounds = np.arange(args['clip_low'], args['clip_high'], args['clip_step'])

cmd = """
    python3 run_GNN.py --function {}
                       --block attention 
                       --dataset Cora
                       --experiment 
                       --max_iters 1000 
                       --max_nfe 100000000 
                       --alpha_ {} 
                       --time {}
                       --k {}
"""

for alpha in alphas:
    for t in t_values:
        for k in k_values:
            cmd_ = cmd.format(args['function'], alpha, t, k).replace("\n", "").replace("\t", "")
            
            print(cmd_)
            os.system(cmd_)

    
