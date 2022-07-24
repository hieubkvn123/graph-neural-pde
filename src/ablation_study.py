import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = ['red', 'green', 'blue']
result_files = {
    'Cora' : 'tests/geom_split_results_Cora.csv',
    'Citeseer' : 'tests/geom_split_results_Citeseer.csv',
    'Pubmed' : 'tests/geom_split_results_Pubmed.csv'
}
grand_file = 'tests/grand_ablation_study.csv'
df_grand = pd.read_csv(grand_file)

# Study the effect of depth on DeepGRAND
num_ds = len(result_files.keys())
fig, ax = plt.subplots(1, num_ds, figsize=(10 * num_ds, 6))

for i, ds in enumerate(result_files.keys()):
    if(num_ds > 1):
        ax_ = ax[i]
    else:
        ax_ = ax

    df = pd.read_csv(result_files[ds])
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
    ax_.legend()

print('[INFO] Saving ablation study results ...')
plt.savefig('tests/ablation_study_effect_of_T.png', bbox_inches='tight')
# plt.show()
