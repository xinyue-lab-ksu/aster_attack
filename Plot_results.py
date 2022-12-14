import pandas as pd
import  matplotlib.pyplot as plt

recall = {'epsilon': [1e-03, 1e-06, 1e-08, 1e-09], 'aster': [0.68, 0.35, 0.72, 0.43], 'new_method':[0.88, 0.91, 0.90, 0.89]}
recall_df = pd.DataFrame(data=recall).set_index(['epsilon'])
plt.figure()
plt.rcParams.update({'font.size': 22}) # must set in top
recall_df.plot.bar(rot=45, title="Recall")
plt.show()

precision = {'epsilon': [1e-03, 1e-06, 1e-08, 1e-09], 'aster': [0.53, 0.54 , 0.51, 0.46], 'new_method':[0.72, 0.79, 0.68, 0.66]}
precision_df = pd.DataFrame(data=precision).set_index(['epsilon'])

precision_df.plot.bar(rot=45, title="Precision")
plt.show()
