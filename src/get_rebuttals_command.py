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
--epoch 150 \
--experiment \
--max_iters 1000 \
--max_nfe 100000000 \
--l1_weight_decay 0.0 \
--decay 0.0001 \
--gnp \
--trusted_mask \
--dataset Photo \
--alpha_learnable \
'''

datasets = [
    'Computers', 'Photo', 'CoauthorCS', 'Cora', 'Citeseer', 'Pubmed'
]

label_rates = [1, 2, 5, 10, 20]

hyperparams = {
    'Computers': {
        'time' : 3.249016177876166,
        'alpha_' : [1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
        'epsilon_' : [1e-8, 0.001, 0.001, 1e-8, 1e-8]
    },
    'Photo' : {
        'time' : 3.5824027975386623,
        'alpha_' : [1e-8, 1e-6, 1e-6, 1e-8, 1e-8],
        'epsilon_' : [1e-8, 0.001, 0.001, 0.001, 1e-8]
    },
    'CoauthorCS' : {
        'time' : 3.126400580172773,
        'alpha_' : [1e-8, 1e-6, 1e-6, 1e-6, 1e-8],
        'epsilon_' : [1e-8, 0.001, 0.001, 1e-8, 1e-8]
    },
    'Pubmed' : {
        'time' : 12.942327880200853,
        'alpha_' : [1e-8, 1e-8, 1e-6, 1e-8, 1e-6],
        'epsilon_' : [0.001, 1e-8, 1e-8, 0.001, 1e-8]
    },
    'Citeseer' : {
        'time' : 7.874113442879092,
        'alpha_' : [1e-8, 1e-6, 1e-6, 1e-6, 1e-6],
        'epsilon_' : [1e-8, 0.001, 1e-8, 0.001, 0.001]
    },
    'Cora' : {
        'time' : 18.294754260552843,
        'alpha_' : [1e-6, 1e-6, 1e-6, 1e-8, 1e-6],
        'epsilon_' : [1e-8, 1e-8, 0.001, 1e-8, 1e-8]
    }
}

num_runs = 10
for dataset in datasets:
    opt = hyperparams[dataset]
    for i, lbr in enumerate(label_rates):
        time = opt['time']
        alpha_ = opt['alpha_'][i]
        epsilon_ = opt['epsilon_'][i]
        block = 'attention' if dataset in ['Cora', 'Citeseer', 'Pubmed'] else 'hard_attention'

        cmd = template.format(block, time, alpha_, epsilon_, dataset, lbr)
        for i in range(num_runs):
            print(cmd)

