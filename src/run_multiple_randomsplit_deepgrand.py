import re
import os
import subprocess
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def main(opt, planetoid_split=False):
    if(not opt['non_linear']):
        cmd = """
            python3 run_GNN.py --function ext_laplacian3
                               --dataset {} 
                               --time {}
                               --alpha_ {} 
                               --epsilon_ {}
                               --log_file {}
                               --block attention 
                               --epoch 150
                               --experiment 
                               --max_iters 1000 
                               --max_nfe 100000000 
                               --l1_weight_decay 0.0
                               --decay 0.0001
        """
    else:
        cmd = """
            python3 run_GNN.py --function ext_transformer
                               --dataset {} 
                               --time {}
                               --alpha_ {} 
                               --epsilon_ {}
                               --log_file {}
                               --block constant
                               --epoch 150
                               --experiment 
                               --max_iters 1000 
                               --max_nfe 100000000 
                               --l1_weight_decay 0.0
                               --decay 0.0001
        """


    for i in range(opt["num_seeds"]):
        try:
            cmd = cmd.format(opt["dataset"], opt["time"], 
                    opt["alpha"], opt['epsilon'], opt["log_file"])
            cmd_ = cmd.replace("\n", "").replace("\t", "")
            cmd_ = re.sub(' +', ' ', cmd_).strip()
            process = subprocess.Popen(cmd_.split(' ')) # os.system(cmd_)
            
            print(f'[INFO] Running for seed #[{i+1}/{opt["num_seeds"]}], process id = ', process.pid)
            print(cmd)

            process.wait()
        except KeyboardInterrupt:
            print('=======================================================================')
            print('[INFO] Interrupted ...')
            process.kill()
            break


    print("[INFO] Reading result log file at ", opt["log_file"], " ...")
    df = pd.read_csv(opt["log_file"], names=["time", "alpha", "_1", "best_val_acc", "best_test_acc", "_2", "_3", "_4", "_5"])
    best_test_accs = df["best_test_acc"]
    mean_acc = best_test_accs.mean()
    std_acc = best_test_accs.std()

    print("    -> Mean test accuracy : ", mean_acc)
    print("    -> Std test accuracy : ", std_acc)

    print("\n[INFO] Removing log file ...")
    os.remove(opt["log_file"])

    return mean_acc, std_acc

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=False, type=str, default='Cora', help="The dataset to tune for")
    parser.add_argument("--time", required=False, type=float, default=16.0, help="T value")
    parser.add_argument("--alpha", required=False, type=float, default=0.001, help="The exponential alpha for DeepGRAND")
    parser.add_argument("--epsilon", required=False, type=float, default=1e-6, help='The epsilon value for DeepGRAND')
    parser.add_argument("--log_file", required=True, type=str, help="The path to the CSV result file")
    parser.add_argument("--num_seeds", required=False, type=int, default=20, help="Number of random seeds to test")
    parser.add_argument("--non_linear", required=False, action="store_true", help="Linear or non-linear DeepGRAND")
    args = vars(parser.parse_args())
    
    main(args)
