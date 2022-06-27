import os
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--function", required=False, default="ext_laplacian3", help="The extended laplacian function to use")
parser.add_argument("--clip_low", required=False, type = float, default=0.05, help="Lower bound for clipping value")
parser.add_argument("--clip_high", required=False, type = float,  default=1.0005, help="Upper bound for clipping value")
parser.add_argument("--clip_step", required=False, default=0.1, help="Step size for clipping values")
parser.add_argument("--dataset", required=False, default='Cora', help="The dataset to tune for")
parser.add_argument("--time", required=False, default=128.0, help="T value")
args = vars(parser.parse_args())

alphas = [1.0, 2.0, 3.0, 4.0]
bounds = np.arange(args['clip_low'], args['clip_high'], args['clip_step'])

cmd = """
    python3 run_GNN.py --function {}
                       --dataset {} 
                       --time {}
                       --alpha_ {} 
                       --clip_bound {}
                       --block attention 
                       --epoch 100
                       --experiment 
                       --max_iters 1000 
                       --max_nfe 100000000 
"""

for alpha in alphas:
    for bound in bounds:
        cmd_ = cmd.format(args['function'], args["dataset"], args["time"], 
                alpha, bound).replace("\n", "").replace("\t", "")
        
        print(cmd_)
        os.system(cmd_)

    
