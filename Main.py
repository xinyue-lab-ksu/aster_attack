from random import randrange

from Evaluate import run
import statistics
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# Diabetes
epsilon = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
training_samples = [100]
results = []
datasets = ["heartbeat"]
target_models = ["NN"]

for eps in epsilon:
    for training_sample in training_samples:
        for dataset in datasets:
            for model in target_models:
                precisions = []
                recalls = []
                f1_scores = []
                dict = {}
                for i in range(1, 11):
                    randseed = randrange(32, 256)
                    res = run(dataset, model, eps, training_sample, randseed)
                    precisions.append(res[0])
                    recalls.append(res[1])
                    f1_scores.append(res[2])
                dict['model'] = model
                dict['epsilon'] = eps
                dict['precision'] = statistics.mean(precisions)
                dict['recall'] = statistics.mean(recalls)
                dict['precision_var'] = statistics.variance(precisions)
                dict['recall_var'] = statistics.variance(recalls)
                results.append(dict)

# Save the results
df = pd.DataFrame(results)

mean_df = df[['model', 'epsilon', 'precision', 'recall']]
mean_df = mean_df.set_index(['model', 'epsilon'])

var_df = df[['model', 'epsilon', 'precision_var', 'recall_var']]
var_df = var_df.set_index(['model', 'epsilon'])
print(mean_df)
mean_ax = mean_df.plot.bar(rot=45)
var_ax = var_df.plot.bar(rot=45)
plt.show()