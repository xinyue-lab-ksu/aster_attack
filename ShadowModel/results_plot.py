import pandas as pd
import matplotlib.pyplot as plt

d = {'model': ['LR', 'LR', 'LR', 'LR'], 'epsilon': [1.000000e-03, 1.000000e-04, 1.000000e-05, 1.000000e-07], 'precision':[0.78, 0.83, 0.83, 0.82], 'recall': [0.92, 0.91, 0.92, 0.93]}
df = pd.DataFrame(data=d)

mean_df = df[['model', 'epsilon', 'precision', 'recall']]
mean_df = mean_df.set_index(['model', 'epsilon'])


mean_ax = mean_df.plot.bar(rot=45)

plt.show()