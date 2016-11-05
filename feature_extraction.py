from trace import *
import glob
import random
import numpy as np
from math import log
from time import strftime
from scipy.sparse import csr_matrix, vstack

traffic_types = ['HTTP', 'Skype', 'Torrent', 'Youtube']
suffixes = ['','','_part','']
path = 'traces/'

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


def first_x_packets(all_traces, label, nr):
	first_iat = []
	first_size = []
	for a in range(0, len(all_traces)):
		if all_traces[a].label == label:
			first_iat += all_traces[a].get_IAT()[0:nr]
			first_size += all_traces[a].packetsizes[0:nr]
	return first_size, first_iat


def generate_histogram(trace):
	#xedges  = [-1600, -500, -75, -65,-55,-45, 60, 70, 90, 100, 120, 140,150, 370, 390, 490, 510, 1240, 1260, 1600]
	#yedges = list(np.arange(0,10.5,0.5))
	log_packetsize = [np.sign(i)*log(abs(i)) for i in trace.packetsizes]
	log_IAT = [log(i) for i in trace.get_IAT()]
	hist,x,y = np.histogram2d(log_packetsize, log_IAT, bins = (20, 20), normed = True)
	return hist, x,y

def build_feature_matrix(traces):
	classes = []
	feature_matrix = None
	for x in traces:
		hist, xedges, yedges = generate_histogram(x)
		row = csr_matrix(hist.flatten())
		if feature_matrix is None:
			feature_matrix = row
		else:
			feature_matrix = csr_vappend(feature_matrix, row)
		classes.append(traffic_types.index(x.label))

	return feature_matrix, classes

def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a





