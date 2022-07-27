import os
import sys
import traceback
import time as time_
import itertools
import pandas as pd
from run_multiple_geomsplit_deepgrand import main as run1
from run_multiple_randomsplit_deepgrand import main as run2

# dataset = ['Cora', 'Citeseer', 'Pubmed']
dataset = ['Computers', 'Photo', 'CoauthorCS']
time = [16.0, 32.0, 64.0, 4.0, 128.0]
alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
epsilon = [1e-3, 1e-8]
num_seeds = 5 # 20
log_folder = './tests'
columns = ['time', 'alpha', 'epsilon', 'mean_acc', 'std_acc', 'exec_time']
all_perm = list(itertools.product(time, alpha, epsilon))

for d in dataset:
    print(f'[INFO] Running tune for {d}...')
    if(d in ['Cora', 'Citeseer', 'Pubmed']):
        result_file = os.path.join(log_folder, f'geom_split_results_{d}.csv')
    else:
        result_file = os.path.join(log_folder, f'rand_split_results_{d}.csv')

    df = pd.DataFrame(columns=columns)

    # load result file if existed
    if(os.path.exists(result_file)):
        df = pd.read_csv(result_file)
        for col in df.columns:
            if(col not in columns): df = df.drop(col, axis=1)

    for i, params in enumerate(all_perm):
        t, a, e = params
        print(f'--> Current setting : time = {t}, alpha = {a}, epsilon = {e}')

        if(((df['time'] == t)&(df['alpha']==a)&(df['epsilon']==e)).any()):
            print('--> Experiment result exists, skipping...')
            continue

        opt = {
            'dataset' : d,
            'time' : t,
            'alpha' : a,
            'epsilon' : e,
            'log_file' : f'{log_folder}/log_{d}_alp-{a}_eps-{e}_T-{t}.csv',
            'num_seeds' : num_seeds,
            'non_linear' : False
        }
    
        try:
            start = time_.time()
            if(d in ['Cora', 'Citeseer', 'Pubmed']):
                opt['block'] = 'attention'
                mean_acc, std_acc = run1(opt, planetoid_split=True)
            elif(d in ['Computers', 'Photo', 'CoauthorCS']):
                opt['block'] = 'hard_attention'
                mean_acc, std_acc = run2(opt, planetoid_split=False)
            end = time_.time()
        except:
            print(f'--> Run for setting {params} failed ...')
            traceback.print_exc(file=sys.stdout)
            continue

        # Insert row
        df.loc[i] = [t, a, e, mean_acc, std_acc, '{:.2f}'.format((end - start)/60)]

        print(f'--> Done! Storing results in {result_file}... ')

        try:
            df.to_csv(result_file)
        except:
            print(f'--> Saving to {result_file} failed...')
            continue
