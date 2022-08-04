import os
import re
import sys
import traceback
import itertools
import subprocess
import numpy as np
import pandas as pd

def grand_ablation_study_T_value(opt):
    if(opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']):
        cmd = """
            python3 run_GNN.py --function transformer 
                               --dataset {} 
                               --time {}
                               --log_file {}
                               --planetoid_split
                               --block constant 
                               --epoch 100
                               --experiment 
        """
    else:
        cmd = """
            python3 run_GNN.py --function transformer
                               --dataset {} 
                               --time {}
                               --log_file {}
                               --block constant 
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
    datasets = ['Cora', 'Citeseer', 'Pubmed']
    times = [4.0, 16.0, 32.0, 64.0, 128.0]
    # datasets = ['Computers', 'Photo', 'CoauthorCS']
    # times = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    columns = ['dataset', 'time', 'mean_acc', 'std_acc']
    result_file = f'tests/grand_nlr_ablation_study_{"_".join(datasets)}.csv'
    all_perm = list(itertools.product(datasets, times))
    df = pd.DataFrame(columns=columns)

    print('Result file : ', result_file)

    if(os.path.exists(result_file)):
        df = pd.read_csv(result_file)
        for col in df.columns:
            if(col not in columns): df = df.drop(col, axis=1)

    for i, params in enumerate(all_perm):
        ds, t = params

        if(((df['time'] == t)&(df['dataset']==ds)).any()):
            print('--> Experiment result exists, skipping...')
            continue

        opt = {
            'dataset' : ds,
            'time' : t,
            'log_file' : f'tests/log_grand_T={t}_{ds}.csv',
            'num_seeds' : num_seeds
        }

        try:
            mean_acc, std_acc = grand_ablation_study_T_value(opt)
        except:
            print(f'--> Run for setting {params} failed ...')
            traceback.print_exc(file=sys.stdout)
            continue

        df.loc[i] = [ds, t, mean_acc, std_acc]

        print(f'--> Done! Storing results in {result_file}... ')
        try:
            df.to_csv(result_file)
        except:
            print(f'--> Saving to {result_file} failed...')
            continue


if __name__ == '__main__':
    main()
