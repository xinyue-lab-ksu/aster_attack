import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
train_df = pd.read_csv("data/diabetes/diabetes_train.csv")

# Load the dataset
test_df = pd.read_csv("data/diabetes/diabetes_test.csv")


train_labels = np.array(train_df.pop('Outcome'))
train_labels = train_labels != 0
test_labels = np.array(test_df.pop('Outcome'))

train_features = np.array(train_df)
test_features = np.array(test_df)

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

clf = RandomForestClassifier(n_estimators=128, max_depth=12, random_state=0).fit(train_features, train_labels)
predict_proba = clf.predict_proba(test_features)
predictions= clf.predict(test_features)
print("Test Accuracy: ",  accuracy_score(predictions, test_labels))

# save
with open('target_models/diabetes_RF.pkl', 'wb') as f:
    pickle.dump(clf, f)
