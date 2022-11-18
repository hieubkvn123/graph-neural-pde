import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change the settings of matplotlib fonts
font = {
    'weight' : 'bold',
    'size'   : 14
}

matplotlib.rc('font', **font)

COLORS = ['red', 'green', 'blue', 'cyan', 'tab:blue']
result_files = {
    'Cora' : 'tests/geom_split_results_Cora.csv',
    'Citeseer' : 'tests/geom_split_results_Citeseer.csv'
}
fig, ax = plt.subplots(1, len(result_files.keys()), figsize=(15, 7))

# Study the effect of depth on DeepGRAND
for i, ds in enumerate(result_files.keys()):
    df = pd.read_csv(result_files[ds])

    # For each alpha, plot the accuracy change over time
    for j, a_ in enumerate([1e-4, 1e-3, 1e-2, 0.1, 0.5]):
        df_ = df[df['alpha'] == a_]
        mean_acc = df_.groupby('time').mean()['mean_acc']
        std_acc = df_.groupby('time').std()['std_acc']
        ax[i].plot(mean_acc.index, mean_acc, label=f'{a_:.2E}', color=COLORS[j], marker='o')
        ax[i].fill_between(mean_acc.index, mean_acc - std_acc, 
                            mean_acc + std_acc, alpha=0.1, color=COLORS[j])
        ax[i].set_title(ds)
        ax[i].grid()

        if(ds == 'Cora'):
            ax[i].legend(loc='lower left')
        else:
            ax[i].legend(loc='upper right')

print('Saving figure to tests/ablation_study_effect_of_alpha.png')
fig.supxlabel('Depth (T)')
fig.supylabel('Mean/std of test accuracies')

plt.tight_layout()
plt.legend()
plt.savefig('tests/ablation_study_effect_of_alpha.png')
