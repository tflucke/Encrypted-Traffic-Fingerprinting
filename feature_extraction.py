from trace import *
import glob
import random
import cPickle as pickle
import numpy as np
from math import log
from time import strftime
from scipy.sparse import csr_matrix, vstack

traffic_types = ['HTTP', 'Skype', 'Torrent', 'Youtube']
suffixes = ['','','_part','']
path = 'traces/'
object_file = 'traces/pickled_traces.dat'

# Load all traces that match the reg exp
def load_traces():
	all_traces = []
	for i in range(0,len(traffic_types)):
		files = glob.glob(path+traffic_types[i]+'/'+traffic_types[i]+'_*'+suffixes[i]+'.pcap')
		for f in files:
			print strftime("%H:%M:%S") +': ' + f
			t = Trace()
			t.load_pcap(f,traffic_types[i])
			all_traces.append(t)
	return all_traces

# Get the iat and size features for the first 'nr' of packets corresponding to a certain label
def first_x_packets(all_traces, label, nr):
	first_iat = []
	first_size = []
	for a in range(0, len(all_traces)):
		if all_traces[a].label == label:
			first_iat += all_traces[a].get_IAT()[0:nr]
			first_size += all_traces[a].packetsizes[0:nr]
	return first_size, first_iat

# Generate a 2D log histogram (x-axis is packetsize, y-axis is IAT)
def generate_histogram(trace, fixed_range=None):
	log_packetsize = [np.sign(i)*log(abs(i)) for i in trace.packetsizes]
	log_IAT = [log(i) for i in trace.get_IAT()]
	hist,x,y = np.histogram2d(log_packetsize, log_IAT, bins = (20, 20), normed = True, range= fixed_range)
	return hist, x,y

# Generate a feature matrix with the 2D histogram data
def build_feature_matrix(traces, given_range = None):
	if given_range is None:
		fixed_range = determine_histogram_edges(traces)
	else:
		fixed_range = given_range
	classes = []
	feature_matrix = None
	for x in traces:
		hist, xedges, yedges = generate_histogram(x, fixed_range)
		row = csr_matrix(hist.flatten())
		if feature_matrix is None:
			feature_matrix = row
		else:
			feature_matrix = csr_vappend(feature_matrix, row)
		classes.append(traffic_types.index(x.label))

	return feature_matrix, classes, fixed_range

# Append a row to a csr matrix
def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a

# Find global max and min values in all training data to fix edges
def determine_histogram_edges(traces):
	all_IAT_values = [x.get_IAT() for x in traces]
	all_size_values = [x.packetsizes for x in traces]

	min_IAT = log(np.amin([item for sublist in all_IAT_values for item in sublist]))
	max_IAT = log(np.amax([item for sublist in all_IAT_values for item in sublist]))

	min_size_temp = log(abs(np.amin([item for sublist in all_size_values for item in sublist])))
	max_size_temp = log(abs(np.amax([item for sublist in all_size_values for item in sublist])))

	min_size = -1*np.max([min_size_temp, max_size_temp])
	max_size = np.max([min_size_temp, max_size_temp])

	return [[min_size, max_size], [min_IAT, max_IAT]]

# Window all given traces
def window_all_traces(traces):
	all_windowed = []
	for trace in traces:
		all_windowed.extend(trace.get_windowed())

	return all_windowed

# Pickle the traces to avoid reading pcaps	
def pickle_traces(traces):
	with open(object_file, "wb") as f:
	    pickle.dump(len(traces), f)
	    for value in traces:
	        pickle.dump(value, f)


def load_pickled_traces():
	traces = []
	with open(object_file, "rb") as f:
		for _ in range(pickle.load(f)):
			traces.append(pickle.load(f))
	return traces