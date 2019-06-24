import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
sys.path.append("../Preprocessing")
import Prep as prep


label = np.loadtxt('label.csv', delimiter=',', skiprows=0)
train = np.genfromtxt('train.csv', delimiter=',', skip_header = True)

train = prep.missing_point(train)

#print(np.isnan(train).sum())

x_data, label = prep.mutual_info(train, label)
print(x_data.shape)

X_train, X_test, y_train, y_test = prep.split_train_test(x_data, label)

class_weights = prep.get_weight(y_train)
sample_weight = prep.get_sample('balanced',y_train)

X_train, y_train = prep.Outlier(X_train, y_train)
print(X_train.shape)

class_weight = {0: class_weights[0], 1: class_weights[1]}

clf = LogisticRegression(class_weight=class_weight).fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
print('acc_train = ', accuracy_score(y_pred_train, y_train))
target = ['class 1', 'class 2']
print(classification_report(y_train, y_pred_train, target_names=target))
print('auc = ', roc_auc_score(y_train, y_pred_train, average='weighted'))
y_pred_test = clf.predict(X_test)
y_prob_pred_test = clf.predict_proba(X_test)
print('acc_test = ',accuracy_score(y_pred_test, y_test))
print(classification_report(y_test, y_pred_test, target_names=target))
print('f1 score =', f1_score(y_test, y_pred_test, average='weighted'))

clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.7)
clf.fit(X_train, y_train, sample_weight)
y_pred_train = clf.predict(X_train)
print('acc_train = ', accuracy_score(y_pred_train, y_train))
target = ['class 1', 'class 2']
print(classification_report(y_train, y_pred_train, target_names=target))
print('auc = ', roc_auc_score(y_train, y_pred_train, average='weighted'))
y_pred_test = clf.predict(X_test)
y_prob_pred_test = clf.predict_proba(X_test)
print('acc_test = ',accuracy_score(y_pred_test, y_test))
print(classification_report(y_test, y_pred_test, target_names=target))
print('f1 score =', f1_score(y_test, y_pred_test, average='weighted'))
fpr, tpr, thresholds = roc_curve(y_test, y_prob_pred_test[:, 1])
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=32, max_depth=8,class_weight=class_weight).fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
print('acc_train = ', accuracy_score(y_pred_train, y_train))
target = ['class 1', 'class 2']
print(classification_report(y_train, y_pred_train, target_names=target))
print('auc = ', roc_auc_score(y_train, y_pred_train, average='weighted'))
y_pred_test = clf.predict(X_test)
y_prob_pred_test = clf.predict_proba(X_test)
print('acc_test = ',accuracy_score(y_pred_test, y_test))
print(classification_report(y_test, y_pred_test, target_names=target))
print('f1 score =', f1_score(y_test, y_pred_test, average='weighted'))
fpr, tpr, thresholds = roc_curve(y_test, y_prob_pred_test[:, 1])
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()