# -*- coding: utf-8 -*-
"""
Spyder Editor
# Author: ismailozgenc
"""
import numpy as np
import pandas as pd
import collections


X_train = pd.read_csv('X_train.csv').iloc[1:].reset_index(drop=True).values
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test.csv', skiprows=1).values
y_test = pd.read_csv('y_test.csv').values.ravel()

def calc_prior(y_data):
    classes, counts = np.unique(y_data, return_counts=True)
    priors = counts / len(y_data)
    return dict(zip(classes, priors))  # prior knowledge

def calc_likelihoods(X_data,y_data):
    classes = np.unique(y_data)
    counted_features = X_data.shape[1]  # Total number of features
    likelihoods = {}
    for classtype in classes:
        class_indices = np.where(y_data == classtype)  # Indices of rows where the class matches
        X_class = X_data[class_indices]  # Extract rows belonging to the class
        likelihoods[classtype] = {}
        for feature in range(counted_features):
            values, counts = np.unique(X_class[:, feature], return_counts=True)
            total_count = counts.sum()
            zipped = zip(values, counts)
            likelihoods[classtype][feature] = { value: (count) / (total_count) 
                for value, count in zip(values, counts)}    
    return likelihoods

def calc_posterior(X_data,priors,likelihoods):
    predictions = []
    for x in X_data:
        posteriors = {}
        for cls, prior in priors.items():
            posterior = np.log(prior)
            for feature_index, feature_value in enumerate(x):
                if feature_value in likelihoods[cls][feature_index]:
                    posterior += np.log(likelihoods[cls][feature_index][feature_value]) # to avoid underflow
                else:
                    posterior += np.log(1e-18)  # To handle zero value problem
            posteriors[cls] = posterior
        predictions.append(max(posteriors, key=posteriors.get))
    return predictions

priors = calc_prior(y_train)
likelihoods = calc_likelihoods(X_train, y_train)
predictions = calc_posterior(X_test,priors,likelihoods)

accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
print("Confusion matrix:")
print(confusion_matrix)

#part 4.3
total_unique_values = 0
print("\nNumber of parameters needed for")
for cls, features in likelihoods.items():
    print(f"Class: {cls}")
    for feature_index, feature_likelihoods in features.items():
        unique_values_count = len(feature_likelihoods)
        total_unique_values += unique_values_count
        print(f"  Feature {feature_index}: {unique_values_count}")

print(f"Total unique values across all features and classes: {total_unique_values}")

# Q4.4
def top_feature_values(likelihoods):
    scores = {}
    for feature in likelihoods[True]:
        for val, p_true in likelihoods[True][feature].items():
            p_false = likelihoods[False][feature].get(val, 1e-12)
            scores[(feature, val)] = p_true / p_false  # feature importance is related to ratio of seeing a feature across the classes 
    top_true = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_false = sorted(scores.items(), key=lambda x: x[1])[:3]
    print("\nTop 3 feature values for True (galaxy):")
    for (feat, val), ratio in top_true:
        print(f"Feature {feat}, Value {val} => Ratio {ratio:.3f}")
    print("Top 3 feature values for False (otherwise):")
    for (feat, val), ratio in top_false:
        print(f"Feature {feat}, Value {val} => Ratio {ratio:.3f}")

top_feature_values(likelihoods)

# Q4.5: 
def compute_mi(likelihoods, priors):
    mi = {}
    for feature in likelihoods[True]:
        value_specific_info = 0
        # Get all unique values for this feature from both classes.
        values = set(likelihoods[True][feature].keys()).union(likelihoods[False][feature].keys())
        for v in values:
            # Compute marginal probability p(v)
            p_v = sum(priors[cls] * likelihoods[cls][feature].get(v, 0) for cls in [True, False])
            for cls in [True, False]:
                p_joint = priors[cls] * likelihoods[cls][feature].get(v, 0)
                if p_joint > 0 and p_v > 0:
                    value_specific_info += p_joint * np.log(p_joint / (priors[cls] * p_v))
        mi[feature] = value_specific_info
    return mi

mi_scores = compute_mi(likelihoods, priors)
mi_df = pd.DataFrame({'Feature': list(mi_scores.keys()), 'Mutual Information': list(mi_scores.values())})
mi_df = mi_df.sort_values('Mutual Information', ascending=False)
print("\nMutual Information Ranking:")
print(mi_df.to_string(index=False))

def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == True) & (y_pred == True))
    fp = np.sum((y_true == False) & (y_pred == True))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

precisions = []
n_features = X_train.shape[1]
sorted_features = mi_df['Feature'].values
for k in range(1, n_features + 1):
    top_features = sorted_features[:k]
    X_train_k = X_train[:, top_features]
    X_test_k = X_test[:, top_features]
    likelihoods_k = calc_likelihoods(X_train_k, y_train)
    preds_k = calc_posterior(X_test_k, priors, likelihoods_k)
    precisions.append(precision(y_test, preds_k))
precision_table = pd.DataFrame({'Top k Features': np.arange(1, n_features + 1),
                                'Precision': precisions})
print("\nPrecision vs. Top-k Features:")
print(precision_table.to_string(index=False))


def calc_hybrid_likelihoods(X_data, y_data, group_features=[0, 1, 2, 4]):
    classes = np.unique(y_data)
    n_features = X_data.shape[1]
    likelihoods = {}
    for cls in classes:
        idx = np.where(y_data == cls)
        X_class = X_data[idx]
        likelihoods[cls] = {}
        group_data = [tuple(row) for row in X_class[:, group_features]]
        counter = collections.Counter(group_data)
        total = X_class.shape[0]
        likelihoods[cls]['group'] = {val: count / total for val, count in counter.items()}
        likelihoods[cls]['indep'] = {}
        for feature in range(n_features):
            if feature in group_features:
                continue
            values, counts = np.unique(X_class[:, feature], return_counts=True)
            total_count = counts.sum()
            likelihoods[cls]['indep'][feature] = {value: count / total_count for value, count in zip(values, counts)}
    return likelihoods

def calc_hybrid_posterior(X_data, priors, likelihoods, group_features=[0, 1, 2, 4]):
    predictions = []
    for x in X_data:
        posteriors = {}
        for cls, prior in priors.items():
            p = np.log(prior)
            group_tuple = tuple(x[group_features])
            p += np.log(likelihoods[cls]['group'].get(group_tuple, 1e-18))
            for feature in range(len(x)):
                if feature in group_features:
                    continue
                p += np.log(likelihoods[cls]['indep'][feature].get(x[feature], 1e-18))
            posteriors[cls] = p
        predictions.append(max(posteriors, key=posteriors.get))
    return predictions

partially_naive_likelihoods = calc_hybrid_likelihoods(X_train, y_train)
partially_naive_predictions = calc_hybrid_posterior(X_test, priors, partially_naive_likelihoods)
partially_naive_accuracy = np.mean(partially_naive_predictions == y_test)
print("\nHybrid Naive Bayes Accuracy:", partially_naive_accuracy)
partially_naive_confusion_matrix = pd.crosstab(y_test, partially_naive_predictions, rownames=['Actual'], colnames=['Predicted'])
print("Confusion matrix:")
print(partially_naive_confusion_matrix)
