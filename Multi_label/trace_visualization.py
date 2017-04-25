from trace import *
from feature_extraction import *
import numpy as np
import sys
from time import strftime
import matplotlib.pyplot as plt
import numpy.random as nprnd

FEATURE = 'burst' # use burst features or size_IAT ('size_IAT' or 'burst')
modes = ['ipsec', 'ipsec_20','ipsec_50','ipsec_100','ipsec_200','ipsec_300','ipsec_400']

if __name__ == "__main__":
	
	mode = sys.argv[1]

	if mode not in modes:
		sys.exit("Execute as: python trace_visualization.py 'mode' \n mode = " + str(modes))

	all_traces = load_pickled_traces(load_mode=mode)
	if FEATURE == 'size_IAT':
		overall_range = determine_histogram_edges_size_IAT(all_traces)
	elif FEATURE == 'burst':
		overall_range = determine_histogram_edges_burst(all_traces)
	else:
		print 'Not a valid feature space!'

	# Plot all traces as a whole
	for i in all_traces:
		if FEATURE == 'size_IAT':
			H, xedges, yedges = generate_histogram_size_IAT(i, overall_range)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(packetsizes)')
			plt.ylabel('log(IAT)')
			plt.title(str(i.labels) + '_full')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		elif FEATURE == 'burst':
			H, xedges, yedges = generate_histogram_burst(i, overall_range)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(burst size (bytes))')
			plt.ylabel('log(burst time)')
			plt.title(str(i.labels) + '_full')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		else:
			print 'Not a valid feature space!'
		

	windowed_traces = window_all_traces(all_traces)
	if FEATURE == 'size_IAT':
		range_windowed_traces = determine_histogram_edges_size_IAT(windowed_traces)
	elif FEATURE == 'burst':
		range_windowed_traces = determine_histogram_edges_burst(windowed_traces)
	else:
		print 'Not a valid feature space!'

	chosen = np.random.choice(windowed_traces, 20)

	for i in chosen:
		if FEATURE == 'size_IAT':
			H, xedges, yedges = generate_histogram_size_IAT(i, overall_range)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(packetsizes)')
			plt.ylabel('log(IAT)')
			plt.title(str(i.labels) + '_window')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		elif FEATURE == 'burst':
			H, xedges, yedges = generate_histogram_burst(i, overall_range)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(burst size (bytes))')
			plt.ylabel('log(burst time)')
			plt.title(str(i.labels) + '_window')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		else:
			print 'Not a valid feature space!'