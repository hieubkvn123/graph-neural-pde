import os
import pathlib

rebuttal_dir = 'tests/rebuttals'
pathlib.Path(rebuttal_dir).mkdir(exist_ok=True, parents=True)

template = '''
python3 run_GNN.py --function ext_laplacian3 \
--block {} \
--time {} \
--alpha_ {} \
--epsilon_ {} \
--log_file 'tests/rebuttals/iclr_{}_lowlabel.csv' \
--num_per_class {} \
--epoch 250 \
--experiment \
--max_iters 1000 \
--max_nfe 100000000 \
--l1_weight_decay 0.0 \
--decay 0.0001 \
--dataset {} \
--threshold {}
'''

template = '''
python3 run_GNN.py --function ext_laplacian3 --block {} --time {} --alpha_ {} --epsilon_ {} --log_file 'tests/rebuttals/iclr_{}_lowlabel.csv' --num_per_class {} --epoch 250 --experiment --max_iters 1000 --max_nfe 100000000 --l1_weight_decay 0.0 --decay 0.0001 --dataset {} --threshold {}
'''

datasets = [
    # 'Photo', 'CoauthorCS', 'Computers',
    'Cora', 'Citeseer', 'Pubmed'
]

label_rates = [1, 2, 5, 10, 20]

hyperparams = {
    'Computers': {
        'time' : 6.2566016177876166, 
        'alpha_' : [1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
        'epsilon_' : [1e-8, 0.001, 0.001, 1e-8, 1e-8]
    },
    'Photo' : {
        'time' : 8.586016177876166,
        'alpha_' : [1e-8, 1e-6, 1e-6, 1e-8, 1e-8],
        'epsilon_' : [1e-8, 0.001, 0.001, 0.001, 1e-8]
    },
    'CoauthorCS' : {
        'time' : 6.586016177876166,
        'alpha_' : [1e-8, 1e-6, 1e-6, 1e-6, 1e-8],
        'epsilon_' : [1e-8, 0.001, 0.001, 1e-8, 1e-8]
    },
    'Pubmed' : {
        'time' : 14.942327880200853,
        'alpha_' : [1e-8, 1e-8, 1e-6, 1e-8, 1e-6],
        'epsilon_' : [0.001, 1e-8, 1e-8, 0.001, 1e-8]
    },
    'Citeseer' : {
        'time' : 9.874113442879092,
        'alpha_' : [1e-8, 1e-6, 1e-6, 1e-6, 1e-6],
        'epsilon_' : [1e-8, 0.001, 1e-8, 0.001, 0.001]
    },
    'Cora' : {
        'time' : 20.294754260552843,
        'alpha_' : [1e-6, 1e-6, 1e-6, 1e-8, 1e-6],
        'epsilon_' : [1e-8, 1e-8, 0.001, 1e-8, 1e-8]
    }
}

thresholds = {
    'Computers' : {
        20 : 86.27,
        10 : 82.79,
        5 : 81.64,
        2 : 75.90,
        1 : 66.65
    },
    'Photo' : {
        20 : 93.0,
        10 : 90.0,
        5 : 88.0,
        2 : 84.0,
        1 : 82.0
    },
    'CoauthorCS' : {
        20 : 91.0,
        10 : 89.0,
        5 : 87.0,
        2 : 80.0,
        1 : 66.0
    },
    'Cora' : {
        20 : 83.0,
        10 : 82.0,
        5 : 79.5,
        2 : 73.0,
        1 : 63.0
    },
    'Citeseer' : {
        20 : 74.0,
        10 : 72.5,
        5 : 70.5,
        2 : 64.0,
        1 : 57.0
    },
    'Pubmed' : {
        20 : 79.0,
        10 : 76.5,
        5 : 73.0,
        2 : 71.0,
        1 : 65.5
    }
}

print('rm tests/rebuttals/*.csv')
num_runs = 5
for dataset in datasets:
    opt = hyperparams[dataset]
    for i, lbr in enumerate(label_rates):
        time = opt['time']
        alpha_ = opt['alpha_'][i]
        epsilon_ = opt['epsilon_'][i]
        block = 'attention' if dataset in ['Cora', 'Citeseer', 'Pubmed', 'Computers'] else 'hard_attention'
        threshold = thresholds[dataset][lbr]

        cmd = template.format(block, time, alpha_, epsilon_, dataset, lbr, dataset, threshold)

        for i in range(num_runs):
            print(f'echo "{cmd}"')
            print(cmd)

