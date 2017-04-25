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

import itertools
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix

#FEATURE = 'size_IAT' # use burst features or size_IAT ('size_IAT', 'burst' or 'both')
#METHOD = 'LR' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
TEST_SIZE = 0.20


# Logging function
def log(s):
    print '[INFO] ' + s


if __name__ == "__main__":
    mode = sys.argv[1]
    FEATURE = sys.argv[2] # use burst features or size_IAT ('size_IAT', 'burst' or 'both')
    METHOD = sys.argv[3] # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
    all_traces = load_pickled_traces(mode)
    #windowed_traces = window_all_traces(all_traces)

    # Split test set but keep windows from different traces seperated from eachother
    labels = [x.label for x in all_traces]
    X_train_val, X_test, y_train_val, y_test = train_test_split(all_traces,labels, stratify=np.array(labels), test_size=TEST_SIZE, random_state=0)

    X_train_val = window_all_traces(X_train_val)

    if METHOD == 'NB':
        clf = MultinomialNB()
        parameters = ParameterGrid({'alpha': np.logspace(-5.0, 5.0, num=11)})
    elif METHOD == 'RF':
        clf = RandomForestClassifier(random_state = 0)
        parameters = ParameterGrid({'n_estimators': range(10,25)})
    elif METHOD == 'MLP':
        clf = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive', random_state = 0)
        parameters = ParameterGrid({'alpha': np.logspace(-6.0, -2.0, num=5), 'max_iter' : [100,200,300], 'hidden_layer_sizes' : [(100,),(100,100),(100,100,100)]})
    elif METHOD == 'LR':
        clf = LogisticRegression(solver = 'sag', random_state = 0)
        parameters = ParameterGrid({'C': np.logspace(-5.0, 5.0, num=11), 'tol':np.logspace(-5.0, 5.0, num=11)})

    results = [{},{},{},{}]
    fold = 0
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    for train, val in kf.split(X_train_val):
        log('Started testing hyperparameters for fold ' + str(fold+1)+'.')
        # Seperate train list from val list
        train_list = [X_train_val[i] for i in train]
        val_list = [X_train_val[i] for i in val]

        if FEATURE == 'size_IAT':
            feature_matrix, classes, train_range = build_feature_matrix_size_IAT(train_list)
            feature_matrix_val, classes_val, val_range = build_feature_matrix_size_IAT(val_list, train_range)
        elif FEATURE == 'burst':
            feature_matrix, classes, train_range = build_feature_matrix_burst(train_list)
            feature_matrix_val, classes_val, val_range = build_feature_matrix_burst(val_list, train_range)
        elif FEATURE == 'both':
            feature_matrix, classes, train_range = build_feature_matrix_both(train_list)
            feature_matrix_val, classes_val, val_range = build_feature_matrix_both(val_list, train_range)  

        for par in list(parameters):
            clf.set_params(**par)
            clf.fit(feature_matrix, classes)
            prec = clf.score(feature_matrix_val, classes_val)
            results[fold][str(par)] = prec

        fold +=1

    N = float(len(results))
    averaged = { k : sum(t[k] for t in results)/N for k in results[0] }

    best = max(averaged, key=averaged.get)

    for idx, val in enumerate(results):
        print 'fold: ' + str(idx+1) +'; score: ' + str(val[best])
    print 'Best parameter setting: ' + best + '; with average score: ' + str(averaged[best])








