from trace import *
from feature_extraction import *
import numpy as np
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

FEATURE = 'size_IAT' # use burst features or size_IAT ('size_IAT' or 'burst')
METHOD = 'MLP' # options: 'NB' : Naive Bayes, 'RF' : random forest, 'MLP' : , 'LR': logistic regression
TEST_SIZE = 0.20

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
    all_traces = load_pickled_traces()
    windowed_traces = window_all_traces(all_traces)

    # kf = KFold(n_splits=4, shuffle=True)
    # for train, test in kf.split(windowed_traces):
    #     # Seperate train list from test list
    #     train_list = [windowed_traces[i] for i in train]
    #     print 'Number of training samples: ' + str(len(train_list))
    #     test_list = [windowed_traces[i] for i in test]
    #     print 'Number of validation samples: ' + str(len(test_list))

    #     if FEATURE == 'size_IAT':
    #         feature_matrix, classes, train_range = build_feature_matrix_size_IAT(train_list)
    #         feature_matrix_test, classes_test, test_range = build_feature_matrix_size_IAT(test_list, train_range)
    #     elif FEATURE == 'burst':
    #         feature_matrix, classes, train_range = build_feature_matrix_burst(train_list)
    #         feature_matrix_test, classes_test, test_range = build_feature_matrix_burst(test_list, train_range)

    #     if METHOD == 'NB':
    #         clf = MultinomialNB()
    #     elif METHOD == 'RF':
    #         clf = RandomForestClassifier()
    #     elif METHOD == 'MLP':
    #         clf = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive')
    #     elif METHOD == 'LR':
    #         clf = LogisticRegression()
    #     clf.fit(feature_matrix, classes)

    #     # Show probabilities of the prediction
    #     #print clf.predict_proba(feature_matrix_test)

    #     print clf.score(feature_matrix_test, classes_test)

    #     # Compute confusion matrix
    #     cnf_matrix = confusion_matrix(classes_test, clf.predict(feature_matrix_test))
    #     np.set_printoptions(precision=2)
    #     # Plot normalized confusion matrix
        
    #     plt.figure()
    #     plot_confusion_matrix(cnf_matrix, classes=traffic_types, title='Normalized confusion matrix')

    #     plt.show()

    # Split test set
    labels = [x.label for x in windowed_traces]
    X_train_val, X_test, y_train_val, y_test = train_test_split(windowed_traces,labels, stratify=np.array(labels), test_size=TEST_SIZE, random_state=0)
    #print X_train_val, y_train_val, X_test, y_test
    # Use parameters as obtained from CV_hyperparameters
    if METHOD == 'NB':
        clf = MultinomialNB(alpha = 1.0000000000000001e-05)
    elif METHOD == 'RF':
        clf = RandomForestClassifier(n_estimators = 5)
    elif METHOD == 'MLP':
        clf = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive', random_state = 0, alpha = 1.0000000000000001e-06, max_iter = 300, hidden_layer_sizes = (100, 100))
    elif METHOD == 'LR':
        clf = LogisticRegression(C = 1000, tol = 1.0000000000000001e-05)
    
    if FEATURE == 'size_IAT':
        feature_matrix, classes, train_range = build_feature_matrix_size_IAT(X_train_val)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_size_IAT(X_test, train_range)
    elif FEATURE == 'burst':
        feature_matrix, classes, train_range = build_feature_matrix_burst(train_list)
        feature_matrix_test, classes_test, test_range = build_feature_matrix_burst(X_test, train_range)

    clf.fit(feature_matrix, classes)
    
    print clf.score(feature_matrix_test, classes_test)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(classes_test, clf.predict(feature_matrix_test))
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=traffic_types, title='Normalized confusion matrix')

    plt.show()