from trace import *
import glob
import random
import cPickle as pickle
import numpy as np
from math import log
from time import strftime
from scipy.sparse import csr_matrix, vstack, hstack
import sys

# traffic_types = ['HTTP', 'Skype', 'Torrent', 'Youtube']
traffic_types = ['HTTP', 'SSH', 'Dominion', 'Youtube']
BINS = 32
nan_hist =  np.empty((BINS,BINS), np.float64)
nan_hist[:] = np.NAN
pars = {
	'unencr':{
		'path': 'traces/',
		'object_file': 'traces/pickled_traces.dat',
		'ip': '10.200.1.2'
	},
	'ipsec_ns':{
		'path': 'ipsec_traces/',
		'object_file': 'ipsec_traces/pickled_traces.dat',
		'ip': '192.168.1.2'	
	},
	'ipsec_def':{
		'path': 'ipsec_def_traces/',
		'object_file': 'ipsec_def_traces/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_20ps':{
		'path': 'traces_20ps/',
		'object_file': 'traces_20ps/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_50ps':{
		'path': 'traces_50ps/',
		'object_file': 'traces_50ps/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_100ps':{
		'path': 'traces_100ps/',
		'object_file': 'traces_100ps/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_150ps':{
		'path': 'traces_150ps/',
		'object_file': 'traces_150ps/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_200ps':{
		'path': 'traces_200ps/',
		'object_file': 'traces_200ps/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_300ps':{
		'path': 'traces_300ps/',
		'object_file': 'traces_300ps/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_400ps':{
		'path': 'traces_400ps/',
		'object_file': 'traces_400ps/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_20_ud':{
		'path': 'traces_20_ud/',
		'object_file': 'traces_20_ud/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_50_ud':{
		'path': 'traces_50_ud/',
		'object_file': 'traces_50_ud/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_100_ud':{
		'path': 'traces_100_ud/',
		'object_file': 'traces_100_ud/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_150_ud':{
		'path': 'traces_150_ud/',
		'object_file': 'traces_150_ud/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_200_ud':{
		'path': 'traces_200_ud/',
		'object_file': 'traces_200_ud/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_300_ud':{
		'path': 'traces_300_ud/',
		'object_file': 'traces_300_ud/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'ipsec_400_ud':{
		'path': 'traces_400_ud/',
		'object_file': 'traces_400_ud/pickled_traces.dat',
		'ip': '192.168.0.2'		
	},
	'tor':{
		'path': 'tor_traces/',
		'object_file': 'tor_traces/pickled_traces.dat',
		#'ip': '192.168.2 .2'
                'ip': '10.0.2.15' # IP address of virtual machine
	}
}

# Fill this in to determine which kind of traffic to work on
mode = 'tor'

# Load all traces that match the reg exp
def load_traces():
	all_traces = []
	for i in range(0,len(traffic_types)):
		files = glob.glob(pars[mode]['path']+traffic_types[i]+'/'+traffic_types[i]+'_*'+'.pcap')
		for f in files:
			print strftime("%H:%M:%S") +': ' + f
			t = Trace(pars[mode]['ip'])
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

def generate_histogram(feature, trace, fixed_range=None):
    """
    Generates histograms of the specified type.

    @param feature: May be 'size_IAT, burst, or rtt.
    @param trace: The trace to operate on.
    @param fixed_range: A fixed range. Defaults to None.
    """
    if feature == 'size_IAT':
        log_packetsize = [np.sign(i)*log(abs(i)) for i in trace.packetsizes]
        log_IAT = [log(i) for i in trace.get_IAT()]
        hist, x, y = np.histogram2d(log_packetsize, log_IAT, bins=(BINS, BINS),
                                    normed=True, range=fixed_range)
    if feature == 'burst':
        size, time = trace.get_burst_info()
        log_size = [np.sign(i)*log(abs(i)) for i in size]
        log_time = [log(i) for i in time]
        hist, x, y = np.histogram2d(log_size, log_time, bins=(BINS, BINS),
                                    normed=True, range=fixed_range)
        if np.isnan(np.min(hist)):
            hist[np.isnan(hist)] = 0
    if feature == 'rtt':
        log_rtt = [log(rtt.rtt) for rtt in trace.rtts]
        log_serv = [log(i) for i in range(1, len(log_rtt) + 1)]
        hist, x, y = np.histogram2d(log_serv, log_rtt, bins=(BINS, BINS),
                                    normed=True, range=fixed_range)
        if np.isnan(np.min(hist)):
            hist[np.isnan(hist)] = 0

    return hist, x, y

def build_feature_matrix(feature, traces, given_range=None):
    """
    Builds a feature matrix.

    @param feature: The feature to use. May be 'size_IAT', 'burst', 'rtt', or 'all'.
    @param traces: A list of Trace objects.
    @param given_range: A predefined range. Defaults to None.
    """
    fixed_range = given_range
    classes = []
    feature_matrix = None

    if feature == 'size_IAT':
        raise NotImplementedError
    elif feature == 'burst':
        raise NotImplementedError
    elif feature == 'rtt':
        if given_range is None:
            fixed_range = determine_histogram_edges('rtt', traces)

        for trace in traces:
            hist, xedges, yedges = generate_histogram('rtt', trace, fixed_range)
            row = csr_matrix(hist.flatten())
            if feature_matrix is None:
                feature_matrix = row
            else:
                feature_matrix = csr_vappend(feature_matrix, row)
            classes.append(traffic_types.index(trace.label))
    elif feature == 'all':
        raise NotImplementedError
    else:
        raise ValueError("Invalid feature: {}".format(feature))

    return feature_matrix, classes, fixed_range

# Generate a feature matrix with the 2D histogram data
def build_feature_matrix_burst(traces, given_range = None):
	if given_range is None:
		fixed_range = determine_histogram_edges('burst', traces)
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
		classes.append(traffic_types.index(x.label))

	return feature_matrix, classes, fixed_range

# Generate a feature matrix with the 2D histogram data
def build_feature_matrix_size_IAT(traces, given_range = None):
	if given_range is None:
		fixed_range = determine_histogram_edges('size_IAT', traces)
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
		classes.append(traffic_types.index(x.label))

	return feature_matrix, classes, fixed_range

# Generate a feature matrix with the 2D histogram data
def build_feature_matrix_both(traces, given_range = (None, None)):
	feature_matrix_size_IAT, classes, size_IAT_range = build_feature_matrix_size_IAT(traces, given_range[0])
	feature_matrix_burst, classes, burst_range = build_feature_matrix_burst(traces, given_range[1])

	feature_matrix = hstack([feature_matrix_size_IAT, feature_matrix_burst])

	return csr_matrix(feature_matrix), classes, (size_IAT_range, burst_range)

# Append a row to a csr matrix
def csr_vappend(a,b):
    """
    Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied.
    """

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a

def determine_histogram_edges(feature, traces):
    """
    Determines min and max values for histogram variables.

    @param feature: The feature set to operate on. May be 'size_IAT', 'burst', or 'rtt'.
    @param traces: A list of trace objects.
    """

    if feature == 'size_IAT':
        all_IAT_values = [x.get_IAT() for x in traces]
        all_size_values = [x.get_packetsizes() for x in traces]

        # Get min and max IATs.
        IATs = [item for sublist in all_IAT_values for item in sublist]
        min_y = log(np.amin(IATs))
        max_y = log(np.amax(IATs))

        # Get min and max packet sizes.
        sizes = [item for sublist in all_size_values for item in sublist]
        min_size_temp = log(abs(np.amin(sizes))) #FIXME necessary?
        max_size_temp = log(abs(np.amax(sizes)))
        min_x = -1*np.max([min_size_temp, max_size_temp])
        max_x = np.max([min_size_temp, max_size_temp])

    elif feature == 'burst':
        all_burst_size_values = []
        all_burst_time_values = []
        for trace in traces:
            size, time = trace.get_burst_info()
            all_burst_size_values.append(size)
            all_burst_time_values.append(time)

        # Get min and max burst times.
        times = [item for sublist in all_burst_time_values for item in sublist]
        min_y = log(np.amin(times))
        max_y = log(np.amax(times))

        # Get min and max bursts.
        bursts = [item for sublist in all_burst_size_values for item in sublist]
        min_burst_size_temp = log(abs(np.amin(bursts))) #FIXME Necessary?
        max_burst_size_temp = log(abs(np.amax(bursts)))
        max_size = np.max([min_burst_size_temp, max_burst_size_temp]) #FIXME Necessary?
        min_x = -1 * max_size
        max_x = max_size

    elif feature == 'rtt':
        # Get min and max RTTs
        rtts = [rtt.rtt for trace in traces for rtt in trace.rtts]
        min_x = log(np.amin(rtts)) # Find minimum RTT among all traces.
        max_x = log(np.amax(rtts)) # Find maximum RTT among all traces.

        # Min and max of services will be 1 to len(rtts) + 1
        min_y = 0
        max_y = log(1000)

    else:
        raise ValueError("Invalid feature: {}".format(feature))

    return [[min_x, max_x], [min_y, max_y]]

# Window all given traces
def window_all_traces(traces, window_size = 1024):
	all_windowed = []
	for trace in traces:
		all_windowed.extend(trace.get_windowed(window_size = window_size))

	return all_windowed

# Pickle the traces to avoid reading pcaps	
def pickle_traces(traces, o_file=None):
        if o_file is None:
		o_file = pars[load_mode]['object_file']
	with open(o_file, "wb") as f:
	    pickle.dump(len(traces), f)
	    for value in traces:
	        pickle.dump(value, f)


def load_pickled_traces(load_mode=mode, o_file=None):
	traces = []
        if o_file is None:
		o_file = pars[load_mode]['object_file']
	with open(o_file, "rb") as f:
                # The first pickled object represents the total number of objects.
		for _ in range(pickle.load(f)):
			traces.append(pickle.load(f))
	return traces

def consolidate_traces(traces):
    """
    Consolidates a list of traces into a smaller list of in which each trace
    has a different label.

    @param traces: The list of traces to consolidate.
    """
    consolidated = {}

    for t in traces:
        if t.label in consolidated:
            packets = consolidated[t.label].get_packetsizes() + t.get_packetsizes()
            sizes = consolidated[t.label].get_timestamps() + t.get_timestamps()
            new_trace = Trace()
            new_trace.construct_trace(packets, sizes, t.label)
            new_trace.rtts = consolidated[t.label].rtts + t.rtts
            consolidated[t.label] = new_trace
        else:
            consolidated[t.label] = t

    for key in consolidated:
        consolidated[key].packetsizes = sorted(consolidated[key].get_packetsizes())
        consolidated[key].timestamps = sorted(consolidated[key].get_timestamps())

    return [v for _, v in consolidated.items()]
