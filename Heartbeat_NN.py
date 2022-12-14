from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
train_df = pd.read_csv("data/heartbeat/heartbeat.csv")
test_df = pd.read_csv("data/heartbeat/heartbeat_test.csv")

# Clean, split the data into test & train
# Use a utility from sklearn to split and shuffle your dataset.
# train_df, test_df = train_test_split(df, test_size=0.2)
train_df = np.array(train_df)
test_df = np.array(test_df)

train_df = np.array(train_df)
test_df = np.array(test_df)

train_labels = np.array(train_df[:, -1])
test_labels = np.array(test_df[:, -1])

train_features = np.array(train_df[:, :-1])
test_features = np.array(test_df[:, :-1])

# Normalize the data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
train_features = np.clip(train_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print('Training labels shape:', train_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Test features shape:', test_features.shape)

clf = MLPClassifier(hidden_layer_sizes = (64, 32, 16), max_iter = 16, activation = 'relu').fit(train_features, train_labels)
predict_proba = clf.predict_proba(test_features)
predictions= clf.predict(test_features)
print("Test Accuracy: ",  accuracy_score(predictions, test_labels))
train_predictions= clf.predict(train_features)
print("Train Accuracy: ",  accuracy_score(train_predictions, train_labels))


# save
with open('target_models/heartbeat_NN.pkl', 'wb') as f:
    pickle.dump(clf, f)
