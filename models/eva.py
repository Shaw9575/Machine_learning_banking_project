import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt

def EVA(y_pred_train,y_train,y_pred_test,y_prob_pred_test,y_test):
    print('acc_train = ', accuracy_score(y_pred_train, y_train))
    target = ['class 1', 'class 2']
    print(classification_report(y_train, y_pred_train, target_names=target))
    print('auc = ', roc_auc_score(y_train, y_pred_train, average='weighted'))
    print('acc_test = ',accuracy_score(y_pred_test, y_test))
    print(classification_report(y_test, y_pred_test, target_names=target))
    print('f1 score =', f1_score(y_test, y_pred_test, average='weighted'))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_pred_test[:, 1])
    plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()