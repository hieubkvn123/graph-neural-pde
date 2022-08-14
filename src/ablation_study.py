import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change the settings of matplotlib fonts
font = {
    'weight' : 'bold',
    'size'   : 22
}

matplotlib.rc('font', **font)
COLORS = ['red', 'green', 'blue']
experiment_sets = {
    'set_1' : {
        'result_files' : {
            'Cora' : 'tests/geom_split_results_Cora.csv',
            'Citeseer' : 'tests/geom_split_results_Citeseer.csv',
            'Pubmed' : 'tests/geom_split_results_Pubmed.csv'
        },
        'grand_file' : 'tests/grand_ablation_study.csv'
    },
    'set_2' : {
        'result_files' : {
            'Computers' : 'tests/rand_split_results_Computers.csv',
            'Photo' : 'tests/rand_split_results_Photo.csv',
            'CoauthorCS' : 'tests/rand_split_results_CoauthorCS.csv'
        },
        'grand_file' : 'tests/grand_ablation_study_Computers_Photo_CoauthorCS.csv'
    }
}

num_ds = 3
fig, ax = plt.subplots(len(experiment_sets.keys()), num_ds, 
        figsize=(9 * num_ds, 6 * len(experiment_sets.keys())))
for e, (k, opt) in enumerate(experiment_sets.items()):
    # Extract constants
    result_files = opt['result_files']
    grand_file = opt['grand_file']

    # Read from baseline file
    df_grand = pd.read_csv(grand_file)

    # Study the effect of depth on DeepGRAND
    num_ds = len(result_files.keys())

    for i, ds in enumerate(result_files.keys()):
        if(num_ds > 1):
            ax_ = ax[e][i]
        else:
            ax_ = ax[e]

        df = pd.read_csv(result_files[ds])
        print('DeepGRAND results for ', ds, ': ')
        print(df.sort_values('mean_acc', ascending=False).drop_duplicates(['time']))

        df = df.groupby('time').max('mean_acc')
        df_ = df_grand[df_grand['dataset']==ds].groupby('time').max('mean_acc')

        times = list(df.index)
        mean_acc = df['mean_acc'].values
        std = df['std_acc'].values

        ax_.set_title(ds)
        ax_.grid()

        # Visualize DeepGRAND result
        ax_.plot(times, mean_acc, color=COLORS[0], marker='o', label='DeepGRAND')
        ax_.fill_between(times, mean_acc - std, mean_acc + std, alpha=0.1, color=COLORS[0])

        # Visualize GRAND result
        times = df_.index
        ax_.plot(times, df_['mean_acc'], color=COLORS[1], marker='o', label='GRAND')
        ax_.fill_between(times, df_['mean_acc'] - df_['std_acc'], 
                            df_['mean_acc'] + df_['std_acc'], alpha=0.1, color=COLORS[1])
        ax_.legend(loc='lower left')

print('[INFO] Saving ablation study results ...')
fig.supylabel('Mean/std accuracies')
fig.supxlabel('Depth (T)')
plt.tight_layout()
plt.savefig('tests/ablation_study_effect_of_T.png', bbox_inches='tight')
