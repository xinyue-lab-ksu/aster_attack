import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statistics
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
train_df_1 = pd.read_csv("C:/Users/prasa/Desktop/FALL GRA/Aster-main/Aster-main/ShadowModel/attack_data/train.csv")
test_df_1 = pd.read_csv("C:/Users/prasa/Desktop/FALL GRA/Aster-main/Aster-main/ShadowModel/attack_data/test.csv")

def run_lr_model(times=10):
    precisions = []
    recalls = []
    accuracy_scores = []
    train_accuracy_scores = []
    for i in range(times):
        # Clean, split the data into test & train
        # Use a utility from sklearn to split and shuffle your dataset.
        #train_df, test_df = train_test_split(df, test_size=0.1)
        train_df = np.array(train_df_1)
        test_df = np.array(test_df_1)

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

        clf = LogisticRegression().fit(train_features, train_labels)
        predict_proba = clf.predict_proba(test_features)
        predictions = clf.predict(test_features)
        accuracy_scores.append(accuracy_score(predictions, test_labels))
        precisions.append(precision_score(test_labels, predictions))
        recalls.append(recall_score(test_labels, predictions))
        train_predictions = clf.predict(train_features)
        train_accuracy_scores.append(accuracy_score(train_predictions, train_labels))

    lr_precision = statistics.mean(precisions)
    lr_recall = statistics.mean(recalls)
    accuracy = statistics.mean(accuracy_scores)
    train_accuracy = statistics.mean(train_accuracy_scores)
    print("LR Model - Precisions {} Recalls {} Test_Accuracy {} Train_Accuracy", precisions, recalls, accuracy_scores,
          train_accuracy_scores)
    return lr_precision, lr_recall, accuracy, train_accuracy


def run_rf_model(times=10):
    precisions = []
    recalls = []
    accuracy_scores = []
    train_accuracy_scores = []
    for i in range(times):
        # Clean, split the data into test & train
        # Use a utility from sklearn to split and shuffle your dataset.
        # train_df, test_df = train_test_split(df, test_size=0.1)
        train_df = np.array(train_df_1)
        test_df = np.array(test_df_1)

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

        clf = RandomForestClassifier(n_estimators=32, max_depth=64, random_state=13).fit(train_features, train_labels)
        predict_proba = clf.predict_proba(test_features)
        predictions = clf.predict(test_features)
        accuracy_scores.append(accuracy_score(predictions, test_labels))
        precisions.append(precision_score(test_labels, predictions))
        recalls.append(recall_score(test_labels, predictions))
        train_predictions = clf.predict(train_features)
        train_accuracy_scores.append(accuracy_score(train_predictions, train_labels))

    rf_precision = statistics.mean(precisions)
    rf_recall = statistics.mean(recalls)
    accuracy = statistics.mean(accuracy_scores)
    train_accuracy = statistics.mean(train_accuracy_scores)

    print("RF Model - Precisions {} Recalls {} Test_Accuracy {} Train_Accuracy", precisions, recalls, accuracy_scores,
          train_accuracy_scores)
    return rf_precision, rf_recall, accuracy, train_accuracy


result = run_lr_model()
print("LR model- Precision {}, Recall {}, Test Accuracy {} Train Accuracy {}", result)

result = run_rf_model()
print("RF model- Precision {}, Recall {}, Test Accuracy {} Train Accuracy {}", result)