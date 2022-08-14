import os
import time as time_
import itertools
import pandas as pd
from best_params import best_params_dict
from run_multiple_geomsplit_deepgrand import main as run1
from run_multiple_randomsplit_deepgrand import main as run2

# dataset = ['Cora', 'Citeseer', 'Pubmed']
dataset = ['Computers', 'Photo', 'CoauthorCS']
labelrates = [1, 2, 5, 10, 20]
alpha = [1e-6, 1e-8]
epsilon = [1e-3, 1e-8]
num_seeds = 5 # 20
log_folder = './tests'
columns = ['lbr', 'alpha', 'epsilon', 'mean_acc', 'std_acc', 'exec_time']
all_perm = list(itertools.product(labelrates, alpha, epsilon))

for d in dataset:
    print(f'[INFO] Running tune for {d}...')
    if(d in ['Cora', 'Citeseer', 'Pubmed']):
        result_file = os.path.join(log_folder, f'geom_split_results_{d}_lowlabel.csv')
    else:
        result_file = os.path.join(log_folder, f'rand_split_results_{d}_lowlabel.csv')

    t = best_params_dict[d]['time']
    df = pd.DataFrame(columns=columns)

    # load result file if existed
    if(os.path.exists(result_file)):
        df = pd.read_csv(result_file)
        for col in df.columns:
            if(col not in columns): df = df.drop(col, axis=1)

    for i, params in enumerate(all_perm):
        lbr, a, e = params
        print(f'--> Current setting : time = {t}, alpha = {a}, epsilon = {e}')

        if(((df['lbr'] == lbr)&(df['alpha']==a)&(df['epsilon']==e)).any()):
            print('--> Experiment result exists, skipping...')
            continue

        opt = {
            'dataset' : d,
            'time' : t,
            'alpha' : a,
            'epsilon' : e,
            'num_per_class' : lbr,
            'log_file' : f'{log_folder}/log_{d}_alp-{a}_eps-{e}_T-{t}.csv',
            'num_seeds' : num_seeds,
            'non_linear' : False
        }
    
        try:
            start = time_.time()
            if(d in ['Cora', 'Citeseer', 'Pubmed']):
                mean_acc, std_acc = run1(opt, planetoid_split=False)
            elif(d in ['Computers', 'Photo', 'CoauthorCS']):
                mean_acc, std_acc = run2(opt, planetoid_split=False)
            end = time_.time()
        except:
            print(f'--> Run for setting {params} failed ...')
            continue

        # Insert row
        df.loc[i] = [lbr, a, e, mean_acc, std_acc, '{:.2f}'.format((end - start)/60)]

        print(f'--> Done! Storing results in {result_file}... ')

        try:
            df.to_csv(result_file)
        except:
            print(f'--> Saving to {result_file} failed...')
            continue
