from trace import *
from feature_extraction import *
import numpy as np
import sys
import argparse as ap
from time import strftime
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split

import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

#FEATURE = 'burst' # use burst features, size_IAT or both ('size_IAT', 'burst' or 'both')
#METHOD = 'LR' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
TEST_SIZE = 0.20
modes = ['unencr','ipsec_ns','ipsec_def',
    'ipsec_20ps','ipsec_50ps','ipsec_100ps','ipsec_150ps','ipsec_200ps','ipsec_300ps','ipsec_400ps',
    'ipsec_20_ud', 'ipsec_50_ud','ipsec_100_ud','ipsec_150_ud','ipsec_200_ud','ipsec_300_ud','ipsec_400_ud','tor']

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":

    parser = ap.ArgumentParser(description = "Create visualizations from data")

    # Mode will almost always be Tor.
    parser.add_argument('mode', metavar = 'mode', type = str,
                        action = 'store', help = 'The mode to use. Usally "tor".')

    # Feature may be burst, size_IAT, or both.
    parser.add_argument('feature', metavar = 'feature', type = str,
                        action = 'store', help = 'The feature to use. "burst", "size_IAT", or "both"')

    # Method may be NB (Naive Bayes), RF (random forests),
    # LR (logistic regression), or MLP (not explained so I don't know).
    parser.add_argument('method', metavar = 'method', type = str,
                        action = 'store', help = 'The method to use. "NB" for Naive Bayes, "LR" for logistic regression, "RF" for random forests, and MLP for ???')

    # Optional option to specify the trace file to use.
    parser.add_argument('--trace', dest = 'trace_file', help = 'The trace file to load')

    # Parse arguments.
    args = parser.parse_args()

    if args.mode == 'unencr':
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
    elif args.mode == 'ipsec_ns':
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
    elif args.mode =='ipsec_def':
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
    elif args.mode =='ipsec_20ps':
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_50ps':
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_100ps':
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 1000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_150ps':  
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 10000, 'tol': 0.01}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_200ps' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}} 
    elif args.mode =='ipsec_300ps' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}} 
    elif args.mode =='ipsec_400ps' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}}, # actually 100, but this is probably an error
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_20_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_50_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_100_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_150_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_200_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='ipsec_300_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100, 'tol': 0.0001}}}
    elif args.mode =='ipsec_400_ud' : 
        parameters = {'size_IAT' : 
            {'LR': {'C': 100000, 'tol': 0.0001}}, # actually 100, but this is probably an error
         'burst':
            {'LR': {'C': 10000, 'tol': 0.0001}}, 
         'both': {'LR': {'C': 100000, 'tol': 0.0001}}}
    elif args.mode =='tor':
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
         'rtt':
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
    
    # Load traces from trace file.
    if args.trace_file:
        try:
            all_traces = load_pickled_traces(load_mode=mode, o_file=args.trace_file)
        except IOError:
            sys.exit("Could not open trace file '{}'".format(args.trace_file))
    else:
        try:
            all_traces = load_pickled_traces(load_mode=mode)
        except IOError:
            sys.exit("Could not open default trace file")

    # Split test set but keep windows from different traces seperated from eachother
    labels = [x.label for x in all_traces]
    X_train_val, X_test, y_train_val, y_test = train_test_split(all_traces,labels, stratify=np.array(labels), test_size=TEST_SIZE, random_state=0)
    
    X_train_val = window_all_traces(X_train_val)
    X_test = window_all_traces(X_test)

    #print X_train_val, y_train_val, X_test, y_test
    # Use parameters as obtained from CV_hyperparameters
    if args.method == 'NB':
        clf = MultinomialNB()
    elif args.method == 'RF':
        clf = RandomForestClassifier()
    elif args.method == 'MLP':
        clf = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive', random_state = 0)
    elif args.method == 'LR':
        clf = LogisticRegression()
    
    # Set parameters
    clf.set_params(**parameters[args.feature][args.method])
    
    if args.feature == 'size_IAT':
        feature_matrix, classes, train_range = build_feature_matrix_size_IAT(X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_size_IAT(X_test, train_range)
    elif args.feature == 'burst':
        feature_matrix, classes, train_range = build_feature_matrix_burst(X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_burst(X_test, train_range)
    elif args.feature == 'rtt':
        feature_matrix, classes, train_range = build_feature_matrix('rtt', X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix('rtt', X_test, train_range)
    elif args.feature == 'both':
        feature_matrix, classes, train_range = build_feature_matrix_both(X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_both(X_test, train_range)   

    clf.fit(feature_matrix, classes)
    
    print clf.score(feature_matrix_test, classes_test)
    
    if args.trace_file:
        trace_file = args.trace_file.split('/')[1][:-4]
    else:
        trace_file = 'default'

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(classes_test, clf.predict(feature_matrix_test))
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix    
    plt.figure()
    #title = 'Normalized confusion matrix for ' + trace_file + ' ' + args.
    title = trace_file.replace('_', ' ') + ' ' + args.feature + ' using ' + args.method
    plot_confusion_matrix(cnf_matrix, classes=traffic_types, title=title)
    plt.tight_layout()

    filename = "confusion_matrix_" + trace_file + '_'
    filename += args.mode + '_' + args.feature + '_' + args.method + '.png'

    plt.savefig(filename)
    plt.show()
