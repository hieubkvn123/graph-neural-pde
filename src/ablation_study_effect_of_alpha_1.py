import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T = 128.0
epsilon = 1e-8
cora = 'tests/geom_split_results_Cora.csv'
citeseer = 'tests/geom_split_results_Citeseer.csv'

df = pd.read_csv(cora)
df = df[df['time'] == T]
df = df[df['epsilon'] == epsilon]
print(df)
