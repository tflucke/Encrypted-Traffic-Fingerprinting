from trace import *
import glob
import random
import cPickle as pickle
import numpy as np
from math import log
from time import strftime
from scipy.sparse import csr_matrix, vstack, hstack
import sys

traffic_types = ['HTTP', 'Skype', 'Torrent', 'Youtube']
BINS = 32
nan_hist =  np.empty((BINS,BINS), np.float64)
nan_hist[:] = np.NAN
pars = {
	'ipsec':{
		'path': 'ipsec_traces/',
		'object_file': 'ipsec_traces/pickled_traces.dat',
		'ip': '192.168.0.2'	
	},
	'ipsec_20':{
		'path': 'ipsec_20/',
		'object_file': 'ipsec_20/pickled_traces.dat',
		'ip': '192.168.0.2'	
	},
	'ipsec_50':{
		'path': 'ipsec_50/',
		'object_file': 'ipsec_50/pickled_traces.dat',
		'ip': '192.168.0.2'	
	},
	'ipsec_100':{
		'path': 'ipsec_100/',
		'object_file': 'ipsec_100/pickled_traces.dat',
		'ip': '192.168.0.2'	
	},
	'ipsec_200':{
		'path': 'ipsec_200/',
		'object_file': 'ipsec_200/pickled_traces.dat',
		'ip': '192.168.0.2'	
	},
	'ipsec_300':{
		'path': 'ipsec_300/',
		'object_file': 'ipsec_300/pickled_traces.dat',
		'ip': '192.168.0.2'	
	},
	'ipsec_400':{
		'path': 'ipsec_400/',
		'object_file': 'ipsec_400/pickled_traces.dat',
		'ip': '192.168.0.2'	
	}
}

# Fill this in to determine which kind of traffic to work on
mode = 'ipsec_400'

# Load all traces that match the reg exp
def load_traces():
	all_traces = []
	files = glob.glob(pars[mode]['path']+'/'+'*'+'.pcap')
	for f in files:
		print strftime("%H:%M:%S") +': ' + f
		start_index = f.rfind('/')
		stop_index = f.rfind('.')
		labels = f[start_index+1:stop_index].split('_')
		labels = [x for x in labels if x in traffic_types]
		t = Trace(pars[mode]['ip'])
		t.load_pcap(f,labels)
		all_traces.append(t)
	return all_traces

# Generate a 2D log histogram (x-axis is packetsize, y-axis is IAT)
def generate_histogram_size_IAT(trace, fixed_range=None):
	log_packetsize = [np.sign(i)*log(abs(i)) for i in trace.packetsizes]
	log_IAT = [log(i) for i in trace.get_IAT()]
	hist,x,y = np.histogram2d(log_packetsize, log_IAT, bins = (BINS, BINS), normed = True, range= fixed_range)
	return hist, x,y

# Generate a 2D log histogram (x-axis is burst total packetsize, y-axis is burst times)
def generate_histogram_burst(trace, fixed_range=None):
	size, time = trace.get_burst_info()
	log_size = [np.sign(i)*log(abs(i)) for i in size]
	log_time = [log(i) for i in time]
	hist,x,y = np.histogram2d(log_size, log_time, bins = (BINS, BINS), normed = True, range= fixed_range)
	if np.isnan(np.min(hist)):
		hist[np.isnan(hist)] = 0
	return hist, x,y

# Generate a feature matrix with the 2D histogram data
def build_feature_matrix_burst(traces, given_range = None):
	if given_range is None:
		fixed_range = determine_histogram_edges_burst(traces)
	else:
		fixed_range = given_range
	classes = []
	feature_matrix = None
	for x in traces:
		hist, xedges, yedges = generate_histogram_burst(x, fixed_range)
		row = csr_matrix(hist.flatten())
		if feature_matrix is None:
			feature_matrix = row
		else:
			feature_matrix = csr_vappend(feature_matrix, row)
		
		classes.append(set(x.labels))

	return feature_matrix, classes, fixed_range

# Generate a feature matrix with the 2D histogram data
def build_feature_matrix_size_IAT(traces, given_range = None):
	if given_range is None:
		fixed_range = determine_histogram_edges_size_IAT(traces)
	else:
		fixed_range = given_range
	classes = []
	feature_matrix = None
	for x in traces:
		hist, xedges, yedges = generate_histogram_size_IAT(x, fixed_range)
		row = csr_matrix(hist.flatten())
		if feature_matrix is None:
			feature_matrix = row
		else:
			feature_matrix = csr_vappend(feature_matrix, row)
		classes.append(set(x.labels))

	return feature_matrix, classes, fixed_range

# Generate a feature matrix with the 2D histogram data
def build_feature_matrix_both(traces, given_range = (None, None)):
	feature_matrix_size_IAT, classes, size_IAT_range = build_feature_matrix_size_IAT(traces, given_range[0])
	feature_matrix_burst, classes, burst_range = build_feature_matrix_burst(traces, given_range[1])

	feature_matrix = hstack([feature_matrix_size_IAT, feature_matrix_burst])

	return csr_matrix(feature_matrix), classes, (size_IAT_range, burst_range)

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
def determine_histogram_edges_size_IAT(traces):
	all_IAT_values = [x.get_IAT() for x in traces]
	all_size_values = [x.get_packetsizes() for x in traces]

	min_IAT = log(np.amin([item for sublist in all_IAT_values for item in sublist]))
	max_IAT = log(np.amax([item for sublist in all_IAT_values for item in sublist]))

	min_size_temp = log(abs(np.amin([item for sublist in all_size_values for item in sublist])))
	max_size_temp = log(abs(np.amax([item for sublist in all_size_values for item in sublist])))

	min_size = -1*np.max([min_size_temp, max_size_temp])
	max_size = np.max([min_size_temp, max_size_temp])

	return [[min_size, max_size], [min_IAT, max_IAT]]

def determine_histogram_edges_burst(traces):
	all_burst_size_values = []
	all_burst_time_values = []
	for x in traces:
		a,b = x.get_burst_info()
		all_burst_size_values.append(a)
		all_burst_time_values.append(b)

	min_burst_time = log(np.amin([item for sublist in all_burst_time_values for item in sublist]))
	max_burst_time = log(np.amax([item for sublist in all_burst_time_values for item in sublist]))

	min_burst_size_temp = log(abs(np.amin([item for sublist in all_burst_size_values for item in sublist])))
	max_burst_size_temp = log(abs(np.amax([item for sublist in all_burst_size_values for item in sublist])))

	min_burst_size = -1*np.max([min_burst_size_temp, max_burst_size_temp])
	max_burst_size = np.max([min_burst_size_temp, max_burst_size_temp])

	return [[min_burst_size, max_burst_size], [min_burst_time, max_burst_time]]

# Window all given traces
def window_all_traces(traces, window_size = 1024):
	all_windowed = []
	for trace in traces:
		all_windowed.extend(trace.get_windowed(window_size = window_size))

	return all_windowed

# Pickle the traces to avoid reading pcaps	
def pickle_traces(traces):

	o_file = pars[mode]['object_file']
	with open(o_file, "wb") as f:
	    pickle.dump(len(traces), f)
	    for value in traces:
	        pickle.dump(value, f)


def load_pickled_traces(load_mode=mode):
	traces = []
	o_file = pars[load_mode]['object_file']
	with open(o_file, "rb") as f:
		for _ in range(pickle.load(f)):
			traces.append(pickle.load(f))
	return traces