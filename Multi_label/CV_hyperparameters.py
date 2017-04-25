from trace import *
from feature_extraction import *
import numpy as np
from time import strftime
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split, ParameterGrid
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, classification_report

import itertools
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix

FEATURE = 'both' # use burst features or size_IAT ('size_IAT', 'burst' or 'both')
METHOD = 'RF' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
TEST_SIZE = 0.20


# Logging function
def log(s):
    print '[INFO] ' + s


if __name__ == "__main__":
    mode = sys.argv[1]
    all_traces = load_pickled_traces(mode)
    windowed_traces = window_all_traces(all_traces)

    # Split test set
    labels = [x.labels for x in windowed_traces]
    multi_single = [len(x) for x in labels]
    X_train_val, X_test, y_train_val, y_test = train_test_split(windowed_traces,labels, stratify=multi_single, test_size=TEST_SIZE, random_state=0)


    clf = RandomForestClassifier(random_state = 0)
    parameters = ParameterGrid({'n_estimators': range(15,50)})


    results = [{},{},{},{}]
    fold = 0
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    for train, val in kf.split(X_train_val):
        log('Started testing hyperparameters for fold ' + str(fold+1)+'.')
        # Seperate train list from val list
        train_list = [windowed_traces[i] for i in train]
        val_list = [windowed_traces[i] for i in val]

        if FEATURE == 'size_IAT':
            feature_matrix, classes, train_range = build_feature_matrix_size_IAT(train_list)
            feature_matrix_val, classes_val, val_range = build_feature_matrix_size_IAT(val_list, train_range)
        elif FEATURE == 'burst':
            feature_matrix, classes, train_range = build_feature_matrix_burst(train_list)
            feature_matrix_val, classes_val, val_range = build_feature_matrix_burst(val_list, train_range)
        elif FEATURE == 'both':
            feature_matrix, classes, train_range = build_feature_matrix_both(X_train_val)
            feature_matrix_val, classes_val, val_range = build_feature_matrix_both(val_list, train_range)  

        mlb = MultiLabelBinarizer(classes=['HTTP', 'Skype', 'Torrent', 'Youtube'])
        classes_bit = mlb.fit_transform(classes)
        classes_val_bit = mlb.fit_transform(classes_val)

        for par in list(parameters):
            clf.set_params(**par)
            clf.fit(feature_matrix, classes_bit)
            predictions = clf.predict(feature_matrix_val)
            loss = hamming_loss(predictions, classes_val_bit)
            results[fold][str(par)] = loss

        fold +=1

    N = float(len(results))
    averaged = { k : sum(t[k] for t in results)/N for k in results[0] }

    best = min(averaged, key=averaged.get)

    for idx, val in enumerate(results):
        print 'fold: ' + str(idx+1) +'; loss: ' + str(val[best])
    print 'Best parameter setting: ' + best + '; with average loss: ' + str(averaged[best])








