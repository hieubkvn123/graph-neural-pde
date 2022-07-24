import os
import re
import itertools
import subprocess
import numpy as np
import pandas as pd

def grand_ablation_study_T_value(opt):
    cmd = """
        python3 run_GNN.py --function laplacian 
                           --dataset {} 
                           --time {}
                           --log_file {}
                           --planetoid_split
                           --block attention 
                           --epoch 100
                           --experiment 
    """

    for i in range(opt["num_seeds"]):
        try:
            cmd = cmd.format(opt["dataset"], opt["time"], opt["log_file"])
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


def main():
    num_seeds = 5
    result_file = 'tests/grand_ablation_study.csv'
    datasets = ['Cora', 'Citeseer']
    times = [4.0, 16.0, 32.0, 64.0, 128.0]
    columns = ['dataset', 'time', 'mean_acc', 'std_acc']

    all_perm = list(itertools.product(datasets, times))
    df = pd.DataFrame(columns=columns)


    for i, params in enumerate(all_perm):
        ds, t = params
        opt = {
            'dataset' : ds,
            'time' : t,
            'log_file' : f'tests/log_grand_T={t}_{ds}.csv',
            'num_seeds' : num_seeds
        }

        mean_acc, std_acc = grand_ablation_study_T_value(opt)
        df.loc[i] = [ds, t, mean_acc, std_acc]

        print(f'--> Done! Storing results in {result_file}... ')
        df.to_csv(result_file)


if __name__ == '__main__':
    main()
