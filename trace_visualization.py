from trace import *
from feature_extraction import *
import numpy as np
from time import strftime
import matplotlib.pyplot as plt
import numpy.random as nprnd

FEATURE = 'burst' # use burst features or size_IAT ('size_IAT' or 'burst')


if __name__ == "__main__":
	
	all_traces = load_pickled_traces()
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
			plt.title(str(i.label) + '_full')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		elif FEATURE == 'burst':
			H, xedges, yedges = generate_histogram_burst(i, overall_range)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(burst size (bytes))')
			plt.ylabel('log(burst time)')
			plt.title(str(i.label) + '_full')
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

	#Split by label, in order to plot some random windows for each label

	Skype = [x for x in windowed_traces if x.label == 'Skype']
	rnd_skype = nprnd.randint(len(Skype), size=5)
	Torrent = [x for x in windowed_traces if x.label == 'Torrent']
	rnd_torrent = nprnd.randint(len(Torrent), size=5)
	Http = [x for x in windowed_traces if x.label == 'HTTP']
	rnd_http = nprnd.randint(len(Http), size=5)
	YouTube = [x for x in windowed_traces if x.label == 'Youtube']
	rnd_youtube = nprnd.randint(len(YouTube), size=5)

	for i in rnd_skype:
		if FEATURE == 'size_IAT':
			H, xedges, yedges = generate_histogram_size_IAT(Skype[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(packetsizes)')
			plt.ylabel('log(IAT)')
			plt.title('Skype_window')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		elif FEATURE == 'burst':
			H, xedges, yedges = generate_histogram_burst(Skype[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(burst size (bytes))')
			plt.ylabel('log(burst time)')
			plt.title('Skype_window')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		else:
			print 'Not a valid feature space!'


	for i in rnd_torrent:
		if FEATURE == 'size_IAT':
			H, xedges, yedges = generate_histogram_size_IAT(Torrent[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(packetsizes)')
			plt.ylabel('log(IAT)')
			plt.title('Torrent_window')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		elif FEATURE == 'burst':
			H, xedges, yedges = generate_histogram_burst(Torrent[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.title('Torrent_window')
			plt.xlabel('log(burst size (bytes))')
			plt.ylabel('log(burst time)')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		else:
			print 'Not a valid feature space!'
		

	for i in rnd_http:
		if FEATURE == 'size_IAT':
			H, xedges, yedges = generate_histogram_size_IAT(Http[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(packetsizes)')
			plt.ylabel('log(IAT)')
			plt.title('Http_window')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		elif FEATURE == 'burst':
			H, xedges, yedges = generate_histogram_burst(Http[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.title('Http_window')
			plt.xlabel('log(burst size (bytes))')
			plt.ylabel('log(burst time)')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		else:
			print 'Not a valid feature space!'

	for i in rnd_youtube:
		if FEATURE == 'size_IAT':
			H, xedges, yedges = generate_histogram_size_IAT(YouTube[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.xlabel('log(packetsizes)')
			plt.ylabel('log(IAT)')
			plt.title('YouTube_window')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		elif FEATURE == 'burst':
			H, xedges, yedges = generate_histogram_burst(YouTube[i], range_windowed_traces)
			X, Y = np.meshgrid(xedges, yedges)
			plt.title('YouTube_window')
			plt.xlabel('log(burst size (bytes))')
			plt.ylabel('log(burst time)')
			a = plt.pcolormesh(X, Y, H.transpose()[::-1])
			plt.colorbar()
			a.set_clim([0,0.6])
			plt.show()
		else:
			print 'Not a valid feature space!'