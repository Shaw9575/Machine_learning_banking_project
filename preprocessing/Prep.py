import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Imputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif, SelectFromModel
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.impute import SimpleImputer

    
def split_train_test(x_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)
    return X_train, X_test, y_train, y_test
    
def StdScaler(x_data):
    scaler = StandardScaler()
    scaler.fit(x_data)
    res = scaler.transform(x_data)
    return res
    
def MMScaler(x_data):
    scaler = MinMaxScaler()
    scaler.fit(x_data)
    res = scaler.transform(x_data)
    return res
    
def mutual_info(x_train, y_train):
    x_train = SelectPercentile(mutual_info_classif, percentile=10).fit_transform(x_train, y_train)
    return x_train, y_train
    
def ExtraTrees(x_train, y_train):
    clf = ExtraTreesClassifier()
    clf = clf.fit(x_train, y_train)
    model = SelectFromModel(clf, prefit=True)
    x_train = model.transform(x_train)
    return x_train, y_train

def fclassif(x_train, y_train):
    x_train = SelectPercentile(f_classif, percentile=10).fit_transform(x_train, y_train)
    return x_train, y_train

def missing_point(x_train):
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    x = imp.fit_transform(x_train)
    return x
    
def get_weight(y_train):
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    return class_weights

def get_sample(class_weight, y_train):
    sample_weights = compute_sample_weight(class_weight, y_train)
    return sample_weights
    
def balance_data():
    # should have one
    pass
    
def Outlier(x_train, y_train):
    # should have better result
    LOF = LocalOutlierFactor(n_neighbors=80)
    Outlier = LOF.fit_predict(x_train, y_train)
    (samples, feature) = x_train.shape
    Train = np.zeros((samples, feature))
    Tlabel = np.zeros(samples)
    count3 = 0
    for c in range(0, samples):
        if Outlier[c] == 1:
            Train[count3, :] = x_train[c, :]
            Tlabel[count3] = y_train[c]
            count3 = count3 + 1

    return Train, Tlabel