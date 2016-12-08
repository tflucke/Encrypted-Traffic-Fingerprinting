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

from sklearn.metrics import confusion_matrix

FEATURE = 'both' # use burst features or size_IAT ('size_IAT' or 'burst')
METHOD = 'MLP' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
TEST_SIZE = 0.20


# Logging function
def log(s):
    print '[INFO] ' + s


if __name__ == "__main__":
    all_traces = load_pickled_traces()
    windowed_traces = window_all_traces(all_traces)

    # Split test set
    labels = [x.label for x in windowed_traces]
    X_train_val, X_test, y_train_val, y_test = train_test_split(windowed_traces,labels, stratify=np.array(labels), test_size=TEST_SIZE, random_state=0)


    if METHOD == 'NB':
        clf = MultinomialNB()
        parameters = ParameterGrid({'alpha': np.logspace(-5.0, 0, num=6)})
    elif METHOD == 'RF':
        clf = RandomForestClassifier(random_state = 0)
        parameters = ParameterGrid({'n_estimators': range(5,16)})
    elif METHOD == 'MLP':
        clf = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive', random_state = 0)
        parameters = ParameterGrid({'alpha': np.logspace(-6.0, -2.0, num=5), 'max_iter' : [100,200,300], 'hidden_layer_sizes' : [(100,),(100,100),(100,100,100)]})
    elif METHOD == 'LR':
        clf = LogisticRegression(solver = 'sag', random_state = 0)
        parameters = ParameterGrid({'C': np.logspace(-5.0, 5.0, num=11), 'tol':np.logspace(-5.0, 5.0, num=11)})

    best = {0 : {'score':0}, 1 : {'score':0}, 2 : {'score':0}, 3 : {'score':0}}
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

        for par in list(parameters):
            clf.set_params(**par)
            clf.fit(feature_matrix, classes)
            prec = clf.score(feature_matrix_val, classes_val)
            if prec > best[fold]['score']:
                best[fold] = par
                best[fold]['score'] = prec

        fold +=1

    print best




