from trace import *
from feature_extraction import *
import numpy as np
from time import strftime
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, KFold

if __name__ == "__main__":
	all_traces = load_traces()

	kf = KFold(n_splits=4, shuffle=True)
	for train, test in kf.split(all_traces):
		# Seperate train list from test list
		train_list = [all_traces[i] for i in train]
		test_list = [all_traces[i] for i in test]

		feature_matrix, classes = build_feature_matrix(train_list)
		feature_matrix_test, classes_test = build_feature_matrix(test_list)

		clf = MultinomialNB()
		clf.fit(feature_matrix, classes)

		print clf.score(feature_matrix_test, classes_test)




