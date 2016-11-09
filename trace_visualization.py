from trace import *
from feature_extraction import *
import numpy as np
from time import strftime
import matplotlib.pyplot as plt
import numpy.random as nprnd

if __name__ == "__main__":
	
	all_traces = load_pickled_traces()
	overall_range = determine_histogram_edges(all_traces)

	# Plot all traces as a whole
	for i in all_traces:
		H, xedges, yedges = generate_histogram(i, overall_range)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title(str(i.label) + '_full')
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()

	windowed_traces = window_all_traces(all_traces)
	range_windowed_traces = determine_histogram_edges(windowed_traces)

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
		H, xedges, yedges = generate_histogram(Skype[i], range_windowed_traces)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Skype_window')
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()

	for i in rnd_torrent:
		H, xedges, yedges = generate_histogram(Torrent[i], range_windowed_traces)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Torrent_window')
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()

	for i in rnd_http:
		H, xedges, yedges = generate_histogram(Http[i], range_windowed_traces)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('Http_window')
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()

	for i in rnd_youtube:
		H, xedges, yedges = generate_histogram(YouTube[i], range_windowed_traces)
		X, Y = np.meshgrid(xedges, yedges)
		plt.xlabel('log(packetsizes)')
		plt.ylabel('log(IAT)')
		plt.title('YouTube_window')
		a = plt.pcolormesh(X, Y, H.transpose()[::-1])
		plt.colorbar()
		a.set_clim([0,0.6])
		plt.show()