from trace import *
from feature_extraction import *
import numpy as np
import sys
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

FEATURE = 'both' # use burst features, size_IAT or both ('size_IAT', 'burst' or 'both')
METHOD = 'MLP' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
TEST_SIZE = 0.20

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
    mode = sys.argv[1]

    if mode == 'unencr':
        parameters = {'size_IAT' : 
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 15},
             'MLP': {'alpha': 1.0000000000000001e-06, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 13},
             'MLP': {'alpha': 1.0000000000000001e-06, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100,100)},
             'LR': {'C': 10000, 'tol': 0.01}}, 
         'both': {
             'NB':{'alpha': 0.1}, 
             'RF': {'n_estimators': 15},
             'MLP': {'alpha': 0.0001, 'max_iter' : 300, 'hidden_layer_sizes' : (100,)},
             'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode == 'ipsec_ns':
        parameters = {'size_IAT' : 
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 15},
             'MLP': {'alpha': 0.01, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}},
         'burst':
            {'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 15},
             'MLP': {'alpha': 0.01, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}}, 
         'both': {
             'NB':{'alpha': 1.0000000000000001e-05}, 
             'RF': {'n_estimators': 12},
             'MLP': {'alpha': 1.0000000000000001e-06, 'max_iter' : 300, 'hidden_layer_sizes' : (100,100,100)},
             'LR': {'C': 100000, 'tol': 0.0001}}}
    elif mode =='ipsec_def':
        pass
    else:
        sys.exit("Execute as: python trace_classification.py 'mode' \n mode = unencr or ipsec_ns or ipsec_def")
    
    all_traces = load_pickled_traces(load_mode=mode)
    windowed_traces = window_all_traces(all_traces, window_size = 1024)

    # Split test set
    labels = [x.label for x in windowed_traces]
    X_train_val, X_test, y_train_val, y_test = train_test_split(windowed_traces,labels, stratify=np.array(labels), test_size=TEST_SIZE, random_state=0)
    #print X_train_val, y_train_val, X_test, y_test
    # Use parameters as obtained from CV_hyperparameters
    if METHOD == 'NB':
        clf = MultinomialNB()
    elif METHOD == 'RF':
        clf = RandomForestClassifier()
    elif METHOD == 'MLP':
        clf = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive', random_state = 0)
    elif METHOD == 'LR':
        clf = LogisticRegression()
    
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
    
    print clf.score(feature_matrix_test, classes_test)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(classes_test, clf.predict(feature_matrix_test))
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=traffic_types, title='Normalized confusion matrix')

    plt.show()