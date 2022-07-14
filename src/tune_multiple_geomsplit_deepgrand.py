import os
import time as time_
import itertools
import pandas as pd
from run_multiple_geomsplit_deepgrand import main as run

dataset = ['Cora', 'Citeseer', 'Pubmed']
time = [16.0, 32.0, 64.0]
alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
epsilon = [1e-3, 1e-8]
num_seeds = 5 # 20
log_folder = './tests'
all_perm = list(itertools.product(time, alpha, epsilon))

for d in dataset:
    print(f'[INFO] Running tune for {d}...')
    result_file = os.path.join(log_folder, f'geom_split_results_{d}.csv')
    df = pd.DataFrame(columns=['time', 'alpha', 'epsilon', 'mean_acc', 'std_acc', 'exec_time'])    

    # Remove result file if existed
    if(os.path.exists(result_file)):
        os.remove(result_file)

    for i, params in enumerate(all_perm):
        t, a, e = params
        print(f'--> Current setting : time = {t}, alpha = {a}, epsilon = {e}')

        opt = {
            'dataset' : d,
            'time' : t,
            'alpha' : a,
            'epsilon' : e,
            'log_file' : f'{log_folder}/log_{d}_alp-{a}_eps-{e}_T-{t}.csv',
            'num_seeds' : num_seeds,
            'non_linear' : False
        }
    
        start = time_.time()
        mean_acc, std_acc = run(opt)
        end = time_.time()

        # Insert row
        df.loc[i] = [t, a, e, mean_acc, std_acc, '{:.2f}'.format((end - start)/60)]

        print(f'--> Done! Storing results in {result_file}... ')
        df.to_csv(result_file)
