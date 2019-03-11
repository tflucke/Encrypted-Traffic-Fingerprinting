#!/usr/bin/python2
from trace import *
from feature_extraction import *
import numpy as np
import sys
import os
from time import strftime
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split

import itertools
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz



#FEATURE = 'burst' # use burst features, size_IAT or both ('size_IAT', 'burst' or 'both')
#METHOD = 'LR' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
TEST_SIZE = 0.20
modes = ['unencr','ipsec_ns','ipsec_def',
    'ipsec_20ps','ipsec_50ps','ipsec_100ps','ipsec_150ps','ipsec_200ps','ipsec_300ps','ipsec_400ps',
    'ipsec_20_ud', 'ipsec_50_ud','ipsec_100_ud','ipsec_150_ud','ipsec_200_ud','ipsec_300_ud','ipsec_400_ud','tor']


if __name__ == "__main__":
    mode = sys.argv[1]
    FEATURE = sys.argv[2] # use burst features or size_IAT ('size_IAT', 'burst' or 'both')
    METHOD = 'RF' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression

    if mode == 'unencr':
        parameters = {'size_IAT' : 
            {'NB':{'alpha': 0.1}, 
             'RF': {'n_estimators': 11},
             'MLP': {'alpha': 0.01, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'NB':{'alpha': 0.1}, 
             'RF': {'n_estimators': 20},
             'MLP': {'alpha': 0.0001, 'max_iter' : 300, 'hidden_layer_sizes' : (100,)},
             'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {
             'NB':{'alpha': 0.01}, 
             'RF': {'n_estimators': 21},
             'MLP': {'alpha': 0.0001, 'max_iter' : 300, 'hidden_layer_sizes' : (100,)},
             'LR': {'C': 10000, 'tol': 0.01}}}
    elif mode == 'ipsec_ns':
        parameters = {'size_IAT' : 
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 20},
             'MLP': {'alpha': 1.0000000000000001e-06, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 23},
             'MLP': {'alpha': 0.001, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {
             'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 13},
             'MLP': {'alpha': 0.01, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_def':
        parameters = {'size_IAT' : 
            {'NB':{'alpha': 0.01}, 
             'RF': {'n_estimators': 24},
             'MLP': {'alpha': 0.01, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 18},
             'MLP': {'alpha': 0.0001, 'max_iter' : 300, 'hidden_layer_sizes' : (100,)},
             'LR': {'C': 100, 'tol': 0.0001}}, 
         'both': {
             'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 20},
             'MLP': {'alpha': 0.01, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_20ps':
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_50ps':
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_100ps':
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 1000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_150ps':  
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 10000, 'tol': 0.01}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_200ps' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}} 
    elif mode =='ipsec_300ps' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}} 
    elif mode =='ipsec_400ps' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}}, # actually 100, but this is probably an error
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_20_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_50_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_100_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_150_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_200_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_300_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100, 'tol': 0.0001}}}
    elif mode =='ipsec_400_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}}, # actually 100, but this is probably an error
         'burst':
            {'LR': {'C': 10000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='tor':
        parameters = {'size_IAT' : 
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 22},
             'MLP': {'alpha': 1.0000000000000001e-06, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 24},
             'MLP': {'alpha': 1.0000000000000001e-05, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {
             'NB':{'alpha': 0.001}, 
             'RF': {'n_estimators': 16},
             'MLP': {'alpha': 0.01, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}}} 
    else:
        sys.exit("Execute as: python trace_classification.py 'mode' \n mode = " + str(modes))
    
    all_traces = load_pickled_traces(load_mode=mode)
    #windowed_traces = window_all_traces(all_traces, window_size = 1024)

    # Split test set but keep windows from different traces seperated from eachother
    labels = [x.label for x in all_traces]
    X_train_val, X_test, y_train_val, y_test = train_test_split(all_traces,labels, stratify=np.array(labels), test_size=TEST_SIZE, random_state=0)
    
    X_train_val = window_all_traces(X_train_val)
    X_test = window_all_traces(X_test)

    #print X_train_val, y_train_val, X_test, y_test
    # Use parameters as obtained from CV_hyperparameters

    clf = RandomForestClassifier()
    
    # Set parameters
    clf.set_params(**parameters[FEATURE][METHOD])
    
    if FEATURE == 'size_IAT':
        feature_matrix, classes, train_range = build_feature_matrix_size_IAT(X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_size_IAT(X_test, train_range)
    elif FEATURE == 'burst':
        feature_matrix, classes, train_range = build_feature_matrix_burst(X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_burst(X_test, train_range)
    elif FEATURE == 'both':
        feature_matrix, classes, train_range = build_feature_matrix_both(X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_both(X_test, train_range)   

    clf.fit(feature_matrix, classes)
    
    #get edges

    if FEATURE == 'size_IAT':
        range_windowed_traces = determine_histogram_edges_size_IAT(X_train_val)
    elif FEATURE == 'burst':
        range_windowed_traces = determine_histogram_edges_burst(X_train_val)
    else:
        print 'Not a valid feature space!'

    # Save the trees
    for i in range(0,len(clf.estimators_)):
        filename = 'trees/'+mode+'/'+'tree_'+FEATURE+'_'+str(i)+'.dot'
        print "Generating tree "+str(i)+"."
        export_graphviz(clf.estimators_[i],out_file=filename, filled=True,rounded=True)
        os.system('dot -Tpng '+filename+' -o '+filename)
