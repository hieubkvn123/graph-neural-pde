import os
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--function", required=False, default="ext_laplacian3", help="The extended laplacian function to use")
parser.add_argument("--clip_low", required=False, type = float, default=0.05, help="Lower bound for clipping value")
parser.add_argument("--clip_high", required=False, type = float,  default=1.0005, help="Upper bound for clipping value")
parser.add_argument("--clip_step", required=False, default=0.1, help="Step size for clipping values")
args = vars(parser.parse_args())

bounds = np.arange(args['clip_low'], args['clip_high'], args['clip_step'])
t_values = np.array(list(range(1, 21))) * 5

cmd = """
    python3 run_GNN.py --function ext_laplacian3
                       --block attention 
                       --dataset Cora
                       --experiment 
                       --max_iters 1000 
                       --max_nfe 100000000 
                       --time {}
                       --run_name 'adaptive_grand_T={}'
"""

for t in t_values:
    cmd_ = cmd.format(t, t).replace("\n", "").replace("\t", "")
    
    print(cmd_)
    os.system(cmd_)

    
