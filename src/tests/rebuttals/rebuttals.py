import pandas as pd

rebuttal_files = [
    'final/iclr_Photo_lowlabel.csv',
    'final/iclr_CoauthorCS_lowlabel.csv',
    'final/iclr_Computers_lowlabel.csv',
    'final/iclr_Cora_lowlabel.csv',
    'final/iclr_Citeseer_lowlabel.csv'
]

for _file in rebuttal_files:
    df = pd.read_csv(_file)
    df.columns = ['time', 'alpha', 'epsilon', 'lbr', 'test_acc']
    mean_acc = df.groupby('lbr').mean()['test_acc'].values
    std_acc = df.groupby('lbr').std()['test_acc'].values

    results = pd.DataFrame({'mean_test_acc':mean_acc, 'std_test_acc':std_acc})
    print('\n', _file)
    print(results)
